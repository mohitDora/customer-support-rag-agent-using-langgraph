from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from src.config import ConfigurationManager

config = ConfigurationManager()
llm_config = config.get_llm_config()

EMBEDDING_MODEL = llm_config["EMBEDDING_MODEL"]
GENERATION_MODEL = llm_config["GENERATION_MODEL"]
LLM_TEMPERATURE = llm_config["LLM_TEMPERATURE"]
GOOGLE_API_KEY = llm_config["GEMINI_API_KEY"]

def get_gemini_llm():
    """Initializes and returns the Google Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model=GENERATION_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY
    )

def get_gemini_embeddings():
    """Initializes and returns the Google Gemini Embeddings model."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )

LLM = get_gemini_llm()
EMBEDDINGS = get_gemini_embeddings()