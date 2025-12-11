import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext
)
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from src.llm.models import get_embed_model
from src.config.settings import settings

class DocumentLoader:
    """Loads documents and creates/manages vector store index"""
    
    def __init__(self, documents_path: str = settings.DOCUMENTS_PATH):
        self.documents_path = documents_path
        self.vector_store_path = settings.VECTOR_DB_PATH
        self.index = None
    
    def get_chroma_collection(self):
        """Get or create Chroma collection for documents"""
        os.makedirs(self.vector_store_path, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=self.vector_store_path)
        chroma_collection = chroma_client.get_or_create_collection("documents")
        return chroma_collection
    
    def load_documents(self) -> VectorStoreIndex:
        """Load documents and create/load vector index"""
        chroma_collection = self.get_chroma_collection()
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Check if index already exists
        if chroma_collection.count() > 0:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=get_embed_model()
            )
        else:
            
            # Check if documents directory exists
            if not os.path.exists(self.documents_path):
                raise FileNotFoundError(
                    f"Documents directory not found: {self.documents_path}"
                )
            
            # Load documents
            documents = SimpleDirectoryReader(
                input_dir=self.documents_path,
                file_extractor={".pdf": PyMuPDFReader()}
            ).load_data()
            
            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            
            # Create index
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=get_embed_model(),
                show_progress=True
            )
        
        return self.index
    
    def get_index(self) -> VectorStoreIndex:
        """Get the current index"""
        if self.index is None:
            self.load_documents()
        return self.index
