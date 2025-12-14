import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application configuration settings"""
    
    def __init__(self):
        # API Keys - allow None for testing
        self.GROQ_API_KEY = os.getenv("API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        # Only validate if not in test environment
        if not os.getenv("PYTEST_CURRENT_TEST"):  # pytest sets this automatically
            if not self.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY (API_KEY) not found in environment")
            if not self.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not found in environment")
    
    # Model Configuration
    LLM_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "30"))
    
    # Paths
    VECTOR_DB_PATH = "./VectorDB/chroma_db"
    MEMORY_DB_PATH = "./VectorDB/MemoryBase"
    DOCUMENTS_PATH = "./data/documents"
    
    # Memory Configuration
    MEMORY_TOKEN_LIMIT = 30000
    MAX_FACTS = 50
    
    # Retrieval Configuration
    SIMILARITY_TOP_K = 5
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
