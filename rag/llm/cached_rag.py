import os
import pypdf
import logging

from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from rag.config import chat, get_root_dir
from rag.utils import load_llm_chat
from rag.llm.state import AgentConfig, RetrievalAction, State

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LLMAgent:
    def __init__(self):
        self.config = AgentConfig()
        self.search = TavilySearchResults(max_results=self.config.max_search_results)
        self.llm = load_llm_chat(chat)
        self.pdf_path = get_root_dir() + "/data/pdf.pdf"
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        self.vectorstore = None
        self.vstore_location = get_root_dir() + "/data/vectorstore"
        self.memory = MemorySaver()
        self.state_config = {"configurable": {"thread_id": "1"}}
        self.graph = self.build_graph()
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # self.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


    def determine_action(self, state: State) -> State:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Determine the best action to be made by the following rules:
            - If the query is asking for a factual or current information: SEARCH
            - If its a normal conversation that does not involve fact checking or
             it is about something you have good knowledge on: NONE
            - If the query does not make sense: ERROR
            Return SEARCH, NONE or ERROR, nothing more.
            Context: {summary}"""),
            MessagesPlaceholder(variable_name="messages"),
            ("system", """DO NOT GIVE EXPLANATIONS!""")
        ])

        chain = prompt | self.llm | StrOutputParser()
        try:
            result = chain.invoke({
                "messages": state["messages"],
                "summary": state.get("summary", "")
            })
            state["action_plan"] = (
                RetrievalAction.SEARCH if "SEARCH" in result.upper()
                else RetrievalAction.NONE if "NONE" in result.upper()
                else RetrievalAction.ERROR
            )

            if state["action_plan"] == RetrievalAction.ERROR:
                self.memory.clear()
                state['messages'].append(AIMessage(content=f"Sorry. But I couldn't understand you."))

        except Exception as e:
            logging.error(f"Action determination failed: {e}")
            state['action_plan'] = RetrievalAction.ERROR
            state['messages'].append(AIMessage(content=f"System error: Error in {e}"))

        return state
    

    def curate_query(self, state: State) -> State:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Curate the query to make it ready for RAG search context. 
             If the query is already clear, just repeat it.
             Query: {query}
             Don't give explanations, just curate the query.""")
        ])
        chain = prompt | self.llm | StrOutputParser()
        state["curated_query"] = chain.invoke({"query": state['messages'][-1].content})

        return state


    def retrieve_context(self, state: State) -> State:
        if state['action_plan'] != RetrievalAction.SEARCH:
            state['context'] = "No context retrieved"
            return state

        query = state['messages'][-1].content

        try:
            if os.path.exists(self.vstore_location):
                self.vectorstore = FAISS.load_local(
                    self.vstore_location,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )

            else:
                self.logger.info(f"Extracting text from {self.pdf_path}")
                reader = pypdf.PdfReader(self.pdf_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"

                self.logger.info("Processing text into chunks")
                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=650,
                    chunk_overlap=150
                ).split_text(text)

                self.vectorstore = FAISS.from_documents(chunks, embedding=self.embeddings)
                self.vectorstore.save_local(self.vstore_location)

            context_parts = []
            docs = self.vectorstore.similarity_search(query, k=4)
            context_parts.extend([doc.page_content for doc in docs])
            state["context"] = " ".join(context_parts)

        except Exception as e:
            logging.error(f"Context retrieval failed: {e}")
            state['context'] = f"Error retrieving context: {str(e)}"

        return state


    def filter_context(self, state: State) -> State:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Check which of the following context is relevant to the query.
             Query: {query}
             Context: {context}
             Return parts of the context that are relevant to the query, without
             any explanations.""")
        ])
        context = state["context"]
        chain = prompt | self.llm | StrOutputParser()
        state["context"] = chain.invoke(
            {"context": context, "query": state["curated_query"]}
        )

        return state


    def generate_response(self, state: State) -> State:
        if state['action_plan'] != RetrievalAction.NONE:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Generate a helpful response using the search results.
                 And incorporate it naturally. Include relevant sources.
                If you can't find relevant information just say so.

                Avaliable context: {context}
                Previous Summary: {summary}"""),
                MessagesPlaceholder(variable_name="messages")
            ])

            try:
                chain = prompt | self.llm | StrOutputParser()
                messages = chain.invoke({
                    "context": state['context'],
                    "summary": state.get("summary", "No summary"),
                    "messages": state['messages']
                })

            except Exception as e:
                logging.error(f"Response generation failed: {e}")
                state['messages'].append(
                    AIMessage(content="I apologize, but I encountered an error generating a response.")
                )
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant whose job is to respond to the user
                 the best way possible. If you don't know about the topic just say so. Be concise"""),
                MessagesPlaceholder(variable_name="messages"),
            ])
            chain = prompt | self.llm | StrOutputParser()
            messages = chain.invoke({"messages": state['messages']})
        
        state["messages"].append(AIMessage(content=messages))

        return state


    def should_continue(self, state: State) -> bool:
        """Determines whether memory summarization is needed."""
        if len(state["messages"]) > self.config.memory_summarizer_threshold:
            return "summarize"
        else:
            return END


    def summarize_conversation(self, state: State):
        summary = state.get("summary", "")

        if summary:
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = self.llm.invoke(messages)
        
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}


    def build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(State)

        workflow.add_node("plan", self.determine_action)
        workflow.add_node("curate_query", self.curate_query)
        workflow.add_node("retrieve", self.retrieve_context)
        workflow.add_node("curate_context", self.filter_context)
        workflow.add_node("generate", self.generate_response)
        workflow.add_node("summarize", self.summarize_conversation)

        workflow.set_entry_point("plan")
        workflow.add_conditional_edges(
            "plan",
            lambda x: x['action_plan'],
            {
                RetrievalAction.SEARCH: "curate_query",
                RetrievalAction.NONE: "generate",
                RetrievalAction.ERROR: END
            }
        )
        workflow.add_edge("curate_query", "retrieve")
        workflow.add_edge("retrieve", "curate_context")
        workflow.add_edge("curate_context", "generate")
        workflow.add_conditional_edges("generate", self.should_continue)
        workflow.add_edge("summarize", END)

        return workflow.compile(checkpointer=self.memory)


    def process_query(self, query: str) -> str:
        try:
            result_state = self.graph.invoke({"messages": query}, self.state_config)
            response = result_state['messages'][-1].content
            return response
        except Exception as e:
            logging.error(f"Query processing failed: {e}")
            return "I apologize, but I encountered an error processing your query."
