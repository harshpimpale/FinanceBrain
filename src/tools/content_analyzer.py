from typing import Dict, List, Any
from llama_index.core.schema import TextNode
from src.llm.models import get_llm
from src.llm.rate_limiter import rate_limiter
from src.utils.logger import setup_logger
import re

logger = setup_logger(__name__)


class ContentAnalyzerTool:
    """
    Comprehensive content analysis tool for documents:
    - Theme extraction
    - Sentiment analysis
    - Entity recognition (people, organizations, dates, numbers)
    - Document structure analysis
    - Key statistics extraction
    """
    
    def __init__(self):
        self.llm = get_llm()
        logger.info("ContentAnalyzerTool initialized")
    
    async def extract_themes(self, text: str, num_themes: int = 5) -> List[Dict[str, str]]:
        """
        Extract main themes from text with descriptions.
        
        Args:
            text: Input text
            num_themes: Number of themes to extract
            
        Returns:
            List of themes with descriptions
        """
        logger.info(f"Extracting {num_themes} themes from text")
        
        prompt = f"""
Analyze the following text and identify the {num_themes} most important themes.
For each theme, provide:
1. Theme name (2-4 words)
2. Brief description (one sentence)

Format your response as:
Theme 1: [Name]
Description: [Description]

Text:
{text[:3000]}

Themes:
"""
        
        response = await rate_limiter.call_with_limit(
            self.llm.acomplete,
            prompt
        )
        
        # Parse themes
        themes = []
        current_theme = {}
        
        for line in str(response).split('\n'):
            line = line.strip()
            if line.startswith('Theme'):
                if current_theme:
                    themes.append(current_theme)
                # Extract theme name
                theme_name = line.split(':', 1)[-1].strip()
                current_theme = {"theme": theme_name, "description": ""}
            elif line.startswith('Description:'):
                desc = line.split(':', 1)[-1].strip()
                current_theme["description"] = desc
        
        if current_theme:
            themes.append(current_theme)
        
        logger.info(f"Extracted {len(themes)} themes")
        return themes[:num_themes]
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text with confidence scores.
        
        Returns:
            Dictionary with sentiment, confidence, and reasoning
        """
        logger.info("Analyzing sentiment")
        
        prompt = f"""
Analyze the sentiment of the following text.
Provide:
1. Overall sentiment (positive/negative/neutral/mixed)
2. Confidence score (0-100)
3. Brief reasoning (one sentence)

Format:
Sentiment: [positive/negative/neutral/mixed]
Confidence: [0-100]
Reasoning: [explanation]

Text:
{text[:2000]}

Analysis:
"""
        
        response = await rate_limiter.call_with_limit(
            self.llm.acomplete,
            prompt
        )
        
        # Parse response
        result_text = str(response)
        sentiment_data = {
            "sentiment": "neutral",
            "confidence": 50,
            "reasoning": ""
        }
        
        for line in result_text.split('\n'):
            line = line.strip()
            if line.startswith('Sentiment:'):
                sentiment_data["sentiment"] = line.split(':', 1)[-1].strip().lower()
            elif line.startswith('Confidence:'):
                try:
                    conf_str = re.search(r'\d+', line)
                    if conf_str:
                        sentiment_data["confidence"] = int(conf_str.group())
                except:
                    pass
            elif line.startswith('Reasoning:'):
                sentiment_data["reasoning"] = line.split(':', 1)[-1].strip()
        
        logger.info(f"Sentiment: {sentiment_data['sentiment']} ({sentiment_data['confidence']}%)")
        return sentiment_data
    
    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Returns:
            Dictionary with entity types and their values
        """
        logger.info("Extracting entities")
        
        prompt = f"""
Extract all important entities from the text below.
Categorize them as:
- People: Names of individuals
- Organizations: Companies, institutions
- Locations: Cities, countries, places
- Dates: Specific dates or time periods
- Numbers: Important financial figures, statistics

Format:
People: name1, name2, name3
Organizations: org1, org2
Locations: loc1, loc2
Dates: date1, date2
Numbers: num1 (context), num2 (context)

Text:
{text[:3000]}

Entities:
"""
        
        response = await rate_limiter.call_with_limit(
            self.llm.acomplete,
            prompt
        )
        
        # Parse entities
        entities = {
            "people": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "numbers": []
        }
        
        for line in str(response).split('\n'):
            line = line.strip()
            if ':' in line:
                key, values = line.split(':', 1)
                key = key.strip().lower()
                
                if key in entities:
                    # Split by comma and clean
                    items = [item.strip() for item in values.split(',') if item.strip()]
                    entities[key].extend(items)
        
        total_entities = sum(len(v) for v in entities.values())
        logger.info(f"Extracted {total_entities} entities across {len(entities)} categories")
        return entities
    
    async def analyze_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze document structure and type.
        
        Returns:
            Dictionary with structure analysis
        """
        logger.info("Analyzing document structure")
        
        # Basic statistics
        words = text.split()
        sentences = text.split('.')
        paragraphs = text.split('\n\n')
        
        prompt = f"""
            Analyze the structure and type of this document.
            Provide:
            1. Document type (e.g., report, article, essay, financial document)
            2. Writing style (formal/informal/technical/narrative)
            3. Main sections identified (list 3-5 major sections)

            Text sample:
            {text[:1500]}

            Analysis:
            """
        
        response = await rate_limiter.call_with_limit(
            self.llm.acomplete,
            prompt
        )
        
        structure = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len([p for p in paragraphs if p.strip()]),
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "document_type": "",
            "writing_style": "",
            "sections": [],
            "analysis": str(response).strip()
        }
        
        # Parse LLM response
        for line in str(response).split('\n'):
            line = line.strip()
            if 'type:' in line.lower():
                structure["document_type"] = line.split(':', 1)[-1].strip()
            elif 'style:' in line.lower():
                structure["writing_style"] = line.split(':', 1)[-1].strip()
            elif line.startswith(('-', '•', '*', str)):
                section = line.lstrip('-•* 0123456789.').strip()
                if section:
                    structure["sections"].append(section)
        
        logger.info(f"Structure analyzed: {structure['document_type']}")
        return structure
    
    async def comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive content analysis combining all methods.
        
        Returns:
            Complete analysis report
        """
        logger.info("Starting comprehensive content analysis")
        
        # Run all analyses
        themes = await self.extract_themes(text, num_themes=5)
        sentiment = await self.analyze_sentiment(text)
        entities = await self.extract_entities(text)
        structure = await self.analyze_structure(text)
        
        report = {
            "themes": themes,
            "sentiment": sentiment,
            "entities": entities,
            "structure": structure,
            "summary": {
                "total_themes": len(themes),
                "primary_sentiment": sentiment["sentiment"],
                "entity_count": sum(len(v) for v in entities.values()),
                "document_type": structure.get("document_type", "unknown"),
                "word_count": structure.get("word_count", 0)
            }
        }
        
        logger.info("Comprehensive analysis completed")
        return report


# Convenience functions
async def analyze_content(text: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """
    Analyze content using specified analysis type.
    
    Args:
        text: Input text
        analysis_type: "comprehensive", "themes", "sentiment", "entities", "structure"
    """
    analyzer = ContentAnalyzerTool()
    
    if analysis_type == "comprehensive":
        return await analyzer.comprehensive_analysis(text)
    elif analysis_type == "themes":
        return {"themes": await analyzer.extract_themes(text)}
    elif analysis_type == "sentiment":
        return await analyzer.analyze_sentiment(text)
    elif analysis_type == "entities":
        return await analyzer.extract_entities(text)
    elif analysis_type == "structure":
        return await analyzer.analyze_structure(text)
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
