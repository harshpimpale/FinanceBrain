from llama_index.llms.groq import Groq
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from src.config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelManager:
    """Manages LLM and embedding models"""
    
    llm_instance = None
    embed_instance = None
    
    @classmethod
    def get_llm(self):
        """Get or create LLM instance"""
        if self.llm_instance is None:
            self.llm_instance = Groq(
                model=settings.LLM_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=0.1
            )
        return self.llm_instance
    
    @classmethod
    def get_embed_model(self):
        """Get or create embedding model instance"""
        if self.embed_instance is None:
            self.embed_instance = GoogleGenAIEmbedding(
                model_name=settings.EMBEDDING_MODEL,
                embed_batch_size=100,
                api_key=settings.GOOGLE_API_KEY,
            )
        return self.embed_instance

# Convenient exports
def get_llm():
    return ModelManager.get_llm()

def get_embed_model():
    return ModelManager.get_embed_model()
