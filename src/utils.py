from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI
from assistant.config import power_chat


def load_llm_chat(model, temperature=0.1, max_new_tokens=1024):
    if model == power_chat:
        chat = ChatOpenAI(model=model, temperature=temperature, max_new_tokens=max_new_tokens)
    else:
        llm = HuggingFaceEndpoint(
            repo_id=model,
            task="text-generation",
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
        )
        chat = ChatHuggingFace(llm=llm)

    return chat