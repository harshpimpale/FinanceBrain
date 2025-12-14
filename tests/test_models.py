"""
Tests for Model Manager (LLM and Embedding models)
"""
import pytest
from unittest.mock import Mock, patch
from src.llm.models import ModelManager, get_llm, get_embed_model

class TestModelManager:
    """Test model management and singleton pattern"""
    
    def test_llm_singleton_pattern(self):
        """Test that LLM uses singleton pattern"""
        # Reset singleton
        ModelManager.llm_instance = None
        
        with patch('src.llm.models.Groq') as mock_groq:
            mock_groq.return_value = Mock()
            
            llm1 = ModelManager.get_llm()
            llm2 = ModelManager.get_llm()
            
            # Should be same instance
            assert llm1 is llm2
            # Groq should only be called once
            assert mock_groq.call_count == 1
    
    def test_embed_model_singleton_pattern(self):
        """Test that embedding model uses singleton pattern"""
        # Reset singleton
        ModelManager.embed_instance = None
        
        with patch('src.llm.models.GoogleGenAIEmbedding') as mock_embed:
            mock_embed.return_value = Mock()
            
            embed1 = ModelManager.get_embed_model()
            embed2 = ModelManager.get_embed_model()
            
            # Should be same instance
            assert embed1 is embed2
            assert mock_embed.call_count == 1
    
    def test_get_llm_convenience_function(self):
        """Test convenience function for getting LLM"""
        ModelManager.llm_instance = None
        
        with patch('src.llm.models.Groq') as mock_groq:
            mock_groq.return_value = Mock()
            
            llm = get_llm()
            
            assert llm is not None
            assert mock_groq.called
    
    def test_get_embed_model_convenience_function(self):
        """Test convenience function for getting embed model"""
        ModelManager.embed_instance = None
        
        with patch('src.llm.models.GoogleGenAIEmbedding') as mock_embed:
            mock_embed.return_value = Mock()
            
            embed = get_embed_model()
            
            assert embed is not None
            assert mock_embed.called
