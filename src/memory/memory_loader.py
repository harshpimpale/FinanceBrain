import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.config.settings import settings

class MemoryLoader:
    """Loads and manages the persistent memory vector store"""
    
    def __init__(self, 
                 persist_path: str = None, 
                 collection_name: str = "memory"):
        self.persist_path = persist_path or settings.MEMORY_DB_PATH
        self.collection_name = collection_name
        self.vector_store = None
    
    def get_chroma_collection(self):
        """Get or create Chroma collection for memory"""
        os.makedirs(self.persist_path, exist_ok=True)
        
        try:
            chroma_client = chromadb.PersistentClient(path=self.persist_path)
            chroma_collection = chroma_client.get_or_create_collection(
                self.collection_name
            )
            return chroma_collection
        except Exception as e:
            raise
    
    def load_memory(self) -> ChromaVectorStore:
        """Load the vector store for memory"""
        chroma_collection = self.get_chroma_collection()
        self.vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection
        )
        return self.vector_store
