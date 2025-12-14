from llama_index.core.memory import (
    Memory,
    FactExtractionMemoryBlock,
    VectorMemoryBlock,
)
from src.llm.models import get_llm, get_embed_model
from src.memory.memory_loader import MemoryLoader
from src.config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MemoryManager:
    """
    Manages both short-term (session) and long-term (persistent) memory.
    Uses FactExtractionMemoryBlock and VectorMemoryBlock for long-term storage.
    """
    
    def __init__(self, session_id: str = "default_session"):
        self.session_id = session_id
        # logger.info(f"Initializing MemoryManager for session: {session_id}")
        logger.info(f"Initializing MemoryManager")
        
        # Initialize long-term memory blocks
        long_term_blocks = self._create_long_term_blocks()
        
        # Create memory with both short and long-term components
        self.memory = Memory.from_defaults(
            session_id=session_id,
            token_limit=settings.MEMORY_TOKEN_LIMIT,
            chat_history_token_ratio=0.02,
            token_flush_size=500,
            memory_blocks=long_term_blocks,
            insert_method="user",
        )
        
        logger.info("MemoryManager initialized successfully")
    
    def _create_long_term_blocks(self):
        """Create long-term memory blocks with rate-limited LLM"""
        # Load persistent vector store
        memory_loader = MemoryLoader()
        vector_store = memory_loader.load_memory()
        
        if vector_store is None:
            raise RuntimeError("Vector store not initialized")
        
        llm = get_llm()
        
        blocks = [
            FactExtractionMemoryBlock(
                name="extracted_facts",
                llm=llm,
                max_facts=settings.MAX_FACTS,
                priority=1,
            ),
            VectorMemoryBlock(
                name="vector_memory",
                vector_store=vector_store,
                priority=2,
                embed_model=get_embed_model(),
            ),
        ]
        
        logger.info(f"Created {len(blocks)} long-term memory blocks")
        return blocks
    
    def get_memory(self):
        """Get the memory instance"""
        return self.memory
    
    def get_context(self) -> str:
        """Get current memory context as string"""
        try:
            context = self.memory.get()
            return str(context)
        except Exception as e:
            logger.error(f"Failed to get memory context: {e}")
            return ""
