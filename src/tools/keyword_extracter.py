from llama_index.core.extractors import KeywordExtractor
from llama_index.core.schema import TextNode
from src.llm.models import get_llm
KEYWORD_PROMPT = """
You are a keyword extraction assistant.

Extract the most important keyword phrases from the text below.
Return ONLY a comma-separated list of keywords (5-10 keywords).
Do NOT include explanations, markdown, bullets, or extra text.

Text:
{context_str}

Keywords:
"""

class KeywordExtractorTool:
    """Tool for extracting keywords from text"""
    
    def __init__(self, max_keywords: int = 8):
        self.max_keywords = max_keywords
        self.llm = get_llm()
    
    async def extract(self, text: str) -> list[str]:
        """Extract keywords from text with rate limiting"""
        
        node = TextNode(text=text)
        
        extractor = KeywordExtractor(
            llm=self.llm,
            prompt_template=KEYWORD_PROMPT,
            keywords=self.max_keywords,
        )
        
        metadata_list = extractor.extract([node])
        
        meta = metadata_list[0]
        kw_str = meta.get("excerpt_keywords") or ""
        keywords = [k.strip() for k in kw_str.split(",") if k.strip()]
        
        return keywords

# Convenience function
async def extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    """Extract keywords from text"""
    extractor = KeywordExtractorTool(max_keywords=max_keywords)
    return await extractor.extract(text)
