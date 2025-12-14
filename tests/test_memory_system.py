"""
Tests for Memory Manager - short-term and long-term memory
Uses REAL components with actual vector store and LLM
"""
import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from src.config.settings import settings

class TestMemoryManager:
    """Test MemoryManager with real vector store and LLM"""
    
    def test_memory_manager_initialization(self, real_memory_manager):
        """Test memory manager initializes correctly with real components"""
        assert real_memory_manager.session_id == "test_session"
        assert real_memory_manager.memory is not None
        
        # Verify memory has token limit configured
        assert real_memory_manager.memory.token_limit == settings.MEMORY_TOKEN_LIMIT
        
        print("✅ Memory Manager initialized with real components")
        print(f"   Session ID: {real_memory_manager.session_id}")
        print(f"   Token limit: {settings.MEMORY_TOKEN_LIMIT}")
    
    def test_long_term_memory_blocks_created(self, real_memory_manager):
        """Test that long-term memory blocks are properly initialized"""
        memory = real_memory_manager.get_memory()
        
        # Check memory blocks exist
        assert hasattr(memory, 'memory_blocks')
        assert len(memory.memory_blocks) >= 2  # FactExtraction + VectorMemory
        
        # Verify block types
        block_names = [block.name for block in memory.memory_blocks]
        assert "extracted_facts" in block_names
        assert "vector_memory" in block_names
        
        print(f"✅ Created {len(memory.memory_blocks)} long-term memory blocks:")
        for block in memory.memory_blocks:
            print(f"   - {block.name} (priority: {block.priority})")
    
    def test_token_limit_configuration(self, real_memory_manager):
        """Test memory respects configured token limits"""
        memory = real_memory_manager.get_memory()
        
        # Verify token limit matches settings
        assert memory.token_limit == settings.MEMORY_TOKEN_LIMIT
        
        # Verify other memory configurations
        assert hasattr(memory, 'chat_history_token_ratio')
        assert hasattr(memory, 'token_flush_size')
        
        print(f"✅ Memory token limit configured: {memory.token_limit}")
        print(f"   Chat history ratio: {memory.chat_history_token_ratio}")
        print(f"   Token flush size: {memory.token_flush_size}")
    