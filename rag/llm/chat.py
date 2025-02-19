import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from typing import List, Dict, Any
import logging

class PDFRAGSystem:
    def __init__(self, openai_api_key: str):
        """
        Initialize the PDF RAG system with necessary components.
        
        Args:
            openai_api_key (str): OpenAI API key for embeddings and chat
        """
        self.openai_api_key = openai_api_key
        os.environ['OPENAI_API_KEY'] = openai_api_key
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        
        self.vector_store = None
        self.chat_model = ChatOpenAI(temperature=0.7, model_name="gpt-4")
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            self.logger.info(f"Extracting text from {pdf_path}")
            reader = pypdf.PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def process_text_into_chunks(self, text: str) -> List[str]:
        """
        Split the text into manageable chunks for processing.
        
        Args:
            text (str): Input text to be split
            
        Returns:
            List[str]: List of text chunks
        """
        try:
            self.logger.info("Processing text into chunks")
            chunks = self.text_splitter.split_text(text)
            return chunks
        except Exception as e:
            self.logger.error(f"Error splitting text into chunks: {str(e)}")
            raise

    def create_vector_store(self, chunks: List[str]) -> None:
        """
        Create a vector store from text chunks using FAISS.
        
        Args:
            chunks (List[str]): List of text chunks to be embedded
        """
        try:
            self.logger.info("Creating vector store")
            self.vector_store = FAISS.from_texts(
                chunks,
                self.embeddings
            )
        except Exception as e:
            self.logger.error(f"Error creating vector store: {str(e)}")
            raise

    def setup_retrieval_chain(self) -> None:
        """
        Set up the conversational retrieval chain for question answering.
        """
        try:
            self.logger.info("Setting up retrieval chain")
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.chat_model,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
        except Exception as e:
            self.logger.error(f"Error setting up retrieval chain: {str(e)}")
            raise

    def process_pdf(self, pdf_path: str) -> None:
        """
        Process a PDF file and prepare it for querying.
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            # Process into chunks
            chunks = self.process_text_into_chunks(text)
            
            # Create vector store
            self.create_vector_store(chunks)
            
            # Setup retrieval chain
            self.setup_retrieval_chain()
            
            self.logger.info("PDF processing completed successfully")
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise

    def query_document(self, query: str) -> Dict[str, Any]:
        """
        Query the processed document with a question.
        
        Args:
            query (str): Question to ask about the document
            
        Returns:
            Dict[str, Any]: Response containing answer and source documents
        """
        try:
            self.logger.info(f"Processing query: {query}")
            if not self.vector_store:
                raise ValueError("No document has been processed yet")
                
            response = self.qa_chain({"question": query})
            
            return {
                "answer": response["answer"],
                "source_documents": response["source_documents"]
            }
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise

    def save_vector_store(self, path: str) -> None:
        """
        Save the vector store to disk for later use.
        
        Args:
            path (str): Path to save the vector store
        """
        try:
            self.logger.info(f"Saving vector store to {path}")
            self.vector_store.save_local(path)
        except Exception as e:
            self.logger.error(f"Error saving vector store: {str(e)}")
            raise

    def load_vector_store(self, path: str) -> None:
        """
        Load a previously saved vector store.
        
        Args:
            path (str): Path to the saved vector store
        """
        try:
            self.logger.info(f"Loading vector store from {path}")
            self.vector_store = FAISS.load_local(path, self.embeddings)
            self.setup_retrieval_chain()
        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            raise