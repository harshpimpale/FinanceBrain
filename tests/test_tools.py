"""
Tests for all tools: Content Analyzer
Uses REAL LLM instances for actual API testing
"""
import pytest

class TestContentAnalyzer:
    """Test ContentAnalyzerTool with real LLM"""
    
    @pytest.fixture
    def analyzer(self, real_llm):
        """Create REAL content analyzer"""
        from src.tools.content_analyzer import ContentAnalyzerTool
        return ContentAnalyzerTool()
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis(self, analyzer, sample_document_text):
        """Test sentiment analysis with confidence scores"""
        sentiment = await analyzer.analyze_sentiment(sample_document_text)
        
        assert "sentiment" in sentiment
        assert "confidence" in sentiment
        assert "reasoning" in sentiment
        assert sentiment["sentiment"] in ["positive", "negative", "neutral", "mixed"]
        assert 0 <= sentiment["confidence"] <= 100
        
        print(f"✅ Sentiment analysis:")
        print(f"   Sentiment: {sentiment['sentiment']}")
        print(f"   Confidence: {sentiment['confidence']}%")
        print(f"   Reasoning: {sentiment['reasoning']}")
    
    @pytest.mark.asyncio
    async def test_entity_extraction(self, analyzer, sample_document_text):
        """Test named entity extraction"""
        entities = await analyzer.extract_entities(sample_document_text)
        
        assert "people" in entities
        assert "organizations" in entities
        assert "locations" in entities
        assert "dates" in entities
        assert "numbers" in entities
        assert all(isinstance(entities[key], list) for key in entities)
        
        total_entities = sum(len(v) for v in entities.values())
        print(f"✅ Extracted {total_entities} entities:")
        for category, items in entities.items():
            if items:
                print(f"   {category.capitalize()}: {', '.join(items[:3])}")
    
    @pytest.mark.asyncio
    async def test_structure_analysis(self, analyzer, sample_document_text):
        """Test document structure analysis"""
        structure = await analyzer.analyze_structure(sample_document_text)
        
        assert "word_count" in structure
        assert "sentence_count" in structure
        assert "paragraph_count" in structure
        assert "document_type" in structure
        assert structure["word_count"] > 0
        assert structure["sentence_count"] > 0
        
        print(f"✅ Document structure analysis:")
        print(f"   Words: {structure['word_count']}")
        print(f"   Sentences: {structure['sentence_count']}")
        print(f"   Paragraphs: {structure['paragraph_count']}")
        print(f"   Type: {structure['document_type']}")
        print(f"   Style: {structure.get('writing_style', 'N/A')}")
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self, analyzer, sample_document_text):
        """Test comprehensive analysis combining all methods"""
        report = await analyzer.comprehensive_analysis(sample_document_text)
        
        assert "themes" in report
        assert "sentiment" in report
        assert "entities" in report
        assert "structure" in report
        assert "summary" in report
        
        print(f"✅ Comprehensive analysis completed:")
        print(f"   Themes extracted: {len(report['themes'])}")
        print(f"   Sentiment: {report['sentiment']['sentiment']}")
        print(f"   Total entities: {report['summary']['entity_count']}")
        print(f"   Document type: {report['summary']['document_type']}")
