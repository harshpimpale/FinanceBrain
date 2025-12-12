from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.node_parser import SentenceSplitter
from src.llm.models import get_llm
from src.llm.rate_limiter import rate_limiter
from src.utils.logger import setup_logger
from typing import List

logger = setup_logger(__name__)


class SummarizerTool:
    """
    Advanced summarization tool with multiple strategies:
    - TreeSummarize: Hierarchical bottom-up summarization for long texts
    - Extractive: Key sentence extraction using semantic similarity
    - Abstractive: Concise LLM-generated summaries
    """
    
    def __init__(self):
        self.llm = get_llm()
        
        # TreeSummarize for hierarchical summarization
        self.tree_summarizer = TreeSummarize(
            llm=self.llm,
            verbose=False,
            use_async=True
        )
        
        # Sentence splitter for extractive summarization
        self.sentence_splitter = SentenceSplitter(
            chunk_size=256,
            chunk_overlap=20
        )
        
        logger.info("SummarizerTool initialized with TreeSummarize")
    
    async def tree_summarize(self, text: str, query: str = "Summarize the key points") -> str:
        """
        Hierarchical tree-based summarization for long documents.
        Best for: Long documents (>5000 tokens), comprehensive summaries
        
        Args:
            text: Input text to summarize
            query: Guiding query for summarization focus
            
        Returns:
            Hierarchically generated summary
        """
        logger.info(f"Tree summarizing text (length: {len(text)})")
        
        # Create nodes from text
        nodes = [TextNode(text=text)]
        nodes_with_scores = [NodeWithScore(node=node, score=1.0) for node in nodes]
        
        # Summarize with rate limiting
        response = await rate_limiter.call_with_limit(
            self.tree_summarizer.asynthesize,
            query,
            nodes=nodes_with_scores
        )
        
        summary = str(response)
        logger.info(f"Tree summary generated (length: {len(summary)})")
        return summary
    
    async def extractive_summary(self, text: str, num_sentences: int = 5) -> List[str]:
        """
        Extract the most important sentences from text.
        Best for: Quick highlights, bullet points, key facts
        
        Args:
            text: Input text
            num_sentences: Number of key sentences to extract
            
        Returns:
            List of key sentences
        """
        logger.info(f"Extractive summary: extracting {num_sentences} key sentences")
        
        # Split into sentences
        sentences = self.sentence_splitter.split_text(text)
        
        if len(sentences) <= num_sentences:
            return sentences[:num_sentences]
        
        # Use LLM to rank sentences by importance
        prompt = f"""
Extract the {num_sentences} most important sentences from the following text.
Return ONLY the sentences, numbered 1-{num_sentences}, exactly as they appear in the text.

Text:
{text}

Important sentences:
"""
        
        response = await rate_limiter.call_with_limit(
            self.llm.acomplete,
            prompt
        )
        
        # Parse numbered sentences
        result_text = str(response)
        key_sentences = []
        for line in result_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                sentence = line.split('.', 1)[-1].strip()
                if sentence:
                    key_sentences.append(sentence)
        
        logger.info(f"Extracted {len(key_sentences)} key sentences")
        return key_sentences[:num_sentences]
    
    async def abstractive_summary(
        self, 
        text: str, 
        max_words: int = 150,
        focus: str = "general"
    ) -> str:
        """
        Generate a concise, LLM-based abstractive summary.
        Best for: Short summaries, specific focus areas, custom length
        
        Args:
            text: Input text
            max_words: Maximum words in summary
            focus: Focus area (e.g., "general", "financial", "technical")
            
        Returns:
            Abstractive summary
        """
        logger.info(f"Abstractive summary: max {max_words} words, focus: {focus}")
        
        prompt = f"""
Create a concise summary of the following text in under {max_words} words.
Focus on: {focus} information
Preserve important numbers, dates, names, and factual details.

Text:
{text}

Summary:
"""
        
        response = await rate_limiter.call_with_limit(
            self.llm.acomplete,
            prompt
        )
        
        summary = str(response).strip()
        logger.info(f"Abstractive summary generated: {len(summary.split())} words")
        return summary
    
    async def summarize_with_bullets(self, text: str, num_points: int = 5) -> List[str]:
        """
        Create a bullet-point summary.
        Best for: Quick overviews, executive summaries
        
        Args:
            text: Input text
            num_points: Number of bullet points
            
        Returns:
            List of bullet points
        """
        logger.info(f"Creating {num_points} bullet point summary")
        
        prompt = f"""
Summarize the following text in exactly {num_points} concise bullet points.
Each bullet should be a complete sentence capturing a key insight.

Text:
{text}

Bullet points:
"""
        
        response = await rate_limiter.call_with_limit(
            self.llm.acomplete,
            prompt
        )
        
        result_text = str(response)
        bullets = []
        for line in result_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                bullet = line.lstrip('-•* ').strip()
                if bullet:
                    bullets.append(bullet)
        
        logger.info(f"Generated {len(bullets)} bullet points")
        return bullets[:num_points]
    
    async def auto_summarize(self, text: str, target_length: str = "medium") -> dict:
        """
        Automatically choose the best summarization strategy based on text length.
        
        Args:
            text: Input text
            target_length: "short" (50 words), "medium" (150 words), "long" (300 words)
            
        Returns:
            Dictionary with summary and metadata
        """
        text_length = len(text.split())
        logger.info(f"Auto-summarizing: {text_length} words, target: {target_length}")
        
        word_limits = {
            "short": 50,
            "medium": 150,
            "long": 300
        }
        max_words = word_limits.get(target_length, 150)
        
        # Choose strategy based on text length
        if text_length < 500:
            # Short text: extractive summary
            sentences = await self.extractive_summary(text, num_sentences=3)
            summary = " ".join(sentences)
            strategy = "extractive"
        elif text_length < 3000:
            # Medium text: abstractive summary
            summary = await self.abstractive_summary(text, max_words=max_words)
            strategy = "abstractive"
        else:
            # Long text: tree summarize
            summary = await self.tree_summarize(text)
            strategy = "tree_summarize"
        
        return {
            "summary": summary,
            "strategy_used": strategy,
            "original_length": text_length,
            "summary_length": len(summary.split()),
            "compression_ratio": round(len(summary.split()) / text_length, 2)
        }


# Convenience functions
async def summarize_text(
    text: str, 
    method: str = "auto",
    **kwargs
) -> str:
    """
    Summarize text using specified method.
    
    Args:
        text: Input text
        method: "auto", "tree", "extractive", "abstractive", "bullets"
        **kwargs: Method-specific parameters
    """
    tool = SummarizerTool()
    
    if method == "auto":
        result = await tool.auto_summarize(text, **kwargs)
        return result["summary"]
    elif method == "tree":
        return await tool.tree_summarize(text, **kwargs)
    elif method == "extractive":
        sentences = await tool.extractive_summary(text, **kwargs)
        return " ".join(sentences)
    elif method == "abstractive":
        return await tool.abstractive_summary(text, **kwargs)
    elif method == "bullets":
        bullets = await tool.summarize_with_bullets(text, **kwargs)
        return "\n• " + "\n• ".join(bullets)
    else:
        raise ValueError(f"Unknown method: {method}")
