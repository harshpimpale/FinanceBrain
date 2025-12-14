# src/workflow/research_workflow.py

from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core import VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from src.tools.summarizer import SummarizerTool
from src.tools.content_analyzer import ContentAnalyzerTool
from src.tools.keyword_extracter import KeywordExtractorTool
from src.retrieval.retriever import RetrieverTool
from src.retrieval.subquery import SubqueriesOperations
from src.llm.models import get_llm
from src.memory.memory_manager import MemoryManager
from src.workflow.events import (
    QueryEvent,
    SubQueriesEvent,
    RetrievalEvent,
    AnalysisEvent,
    SynthesisEvent
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ResearchWorkflow(Workflow):
    """
    Clean research workflow without Context, with practical memory usage.
    Memory is automatically used by LlamaIndex - we just store conversations.
    """
    
    def __init__(self, 
                 index: VectorStoreIndex,
                 memory_manager: MemoryManager,
                 timeout: int = 180,
                 enable_deep_analysis: bool = True):
        super().__init__(timeout=timeout)
        self.index = index
        self.memory_manager = memory_manager
        self.llm = get_llm()
        self.enable_deep_analysis = enable_deep_analysis
        
        # Initialize tools
        self.analyzer = ContentAnalyzerTool()
        self.summarizer = SummarizerTool()
        self.retriever_tool = RetrieverTool(index)
        self.keyword_tool = KeywordExtractorTool()
        self.subquery_operations = SubqueriesOperations(index=index)
        
        logger.info(f"ResearchWorkflow initialized (deep_analysis={enable_deep_analysis})")
    
    # ============================================
    # STEP 1: Analyze Query
    # ============================================
    @step
    async def analyze_query(self, ev: StartEvent) -> QueryEvent:
        """
        Extract keywords and analyze sentiment for response adjustment.
        
        Memory note: Memory is AUTOMATICALLY used by the LLM.
        We only retrieve context here for LOGGING purposes.
        """
        query = ev.query
        logger.info(f"[Step 1] Analyzing query: {query}")
        
        # Extract keywords
        keywords = await self.keyword_tool.extract(query)
        logger.info(f"Keywords extracted: {keywords}")
        
        # Analyze query sentiment
        sentiment = await self.analyzer.analyze_sentiment(query)
        logger.info(f"Query sentiment: {sentiment['sentiment']} ({sentiment['confidence']}%)")
        
        # Get memory context ONLY for logging/debugging
        memory_context = self.memory_manager.get_context()
        if memory_context:
            logger.info(f"Memory context available: {len(memory_context)} chars")
            # Note: We DON'T pass this to the LLM manually
            # LlamaIndex's Memory system handles it automatically
        
        return QueryEvent(
            query=query,
            keywords=keywords,
            sentiment=sentiment
        )
    
    # ============================================
    # STEP 2: Decompose Query
    # ============================================
    @step
    async def decompose_query(self, ev: QueryEvent) -> SubQueriesEvent:
        """
        Break complex queries into sub-questions.
        """
        logger.info(f"[Step 2] Decomposing query: {ev.query}")
        
        sub_queries = await self.subquery_operations.create_sub_queries(ev.query)
        
        logger.info(f"Generated {len(sub_queries)} sub-queries:")
        for i, sq in enumerate(sub_queries, 1):
            logger.info(f"  {i}. {sq}")
        
        return SubQueriesEvent(
            sub_queries=sub_queries,
            original_query=ev.query,
            keywords=ev.keywords,
            sentiment=ev.sentiment
        )
    
    # ============================================
    # STEP 3: Retrieve Contexts
    # ============================================
    @step
    async def retrieve_contexts(self, ev: SubQueriesEvent) -> RetrievalEvent:
        """
        Retrieve relevant contexts for each sub-query.
        """
        logger.info(f"[Step 3] Retrieving contexts for {len(ev.sub_queries)} sub-queries")
        
        sub_queries_and_contexts = await self.subquery_operations.retrieve_for_sub_queries(
            ev.sub_queries
        )
        
        logger.info(f"Retrieved {len(sub_queries_and_contexts)} contexts")
        
        return RetrievalEvent(
            original_query=ev.original_query,
            sub_queries_and_contexts=sub_queries_and_contexts,
            keywords=ev.keywords,
            sentiment=ev.sentiment
        )
    
    # ============================================
    # STEP 4: Analyze Retrieved Content
    # ============================================
    @step
    async def analyze_content(self, ev: RetrievalEvent) -> AnalysisEvent:
        """
        Perform deep content analysis:
        - Extract entities (people, orgs, dates, numbers)
        - Identify themes
        - Analyze content sentiment
        """
        logger.info("[Step 4] Analyzing retrieved content")
        
        # Combine contexts for analysis
        combined_context = "\n\n".join([ctx for _, ctx in ev.sub_queries_and_contexts])
        
        content_analysis = {}
        
        if self.enable_deep_analysis and len(combined_context) > 100:
            # Extract entities
            entities = await self.analyzer.extract_entities(combined_context)
            content_analysis["entities"] = entities
            logger.info(f"Extracted {sum(len(v) for v in entities.values())} entities")
            
            # Extract themes
            themes = await self.analyzer.extract_themes(combined_context, num_themes=3)
            content_analysis["themes"] = themes
            logger.info(f"Identified {len(themes)} themes")
            
            # Analyze content sentiment
            content_sentiment = await self.analyzer.analyze_sentiment(combined_context)
            content_analysis["content_sentiment"] = content_sentiment
            logger.info(f"Content sentiment: {content_sentiment['sentiment']}")
        else:
            content_analysis = {
                "entities": {},
                "themes": [],
                "content_sentiment": {"sentiment": "neutral", "confidence": 0}
            }
        
        return AnalysisEvent(
            original_query=ev.original_query,
            sub_queries_and_contexts=ev.sub_queries_and_contexts,
            keywords=ev.keywords,
            query_sentiment=ev.sentiment,
            content_analysis=content_analysis
        )
    
    # ============================================
    # STEP 5: Summarize Contexts
    # ============================================
    @step
    async def summarize_contexts(self, ev: AnalysisEvent) -> AnalysisEvent:
        """
        Compress contexts to reduce token usage.
        """
        logger.info("[Step 5] Summarizing contexts")
        
        enriched = []
        for sq, ctx in ev.sub_queries_and_contexts:
            summary_result = await self.summarizer.auto_summarize(ctx, target_length="medium")
            
            logger.info(
                f"Compressed: {summary_result['original_length']} → "
                f"{summary_result['summary_length']} words "
                f"(strategy: {summary_result['strategy_used']})"
            )
            
            enriched.append((sq, summary_result["summary"]))
        
        ev.sub_queries_and_contexts = enriched
        return AnalysisEvent(
            original_query=ev.original_query,
            sub_queries_and_contexts=enriched,  # Use enriched data
            keywords=ev.keywords,
            query_sentiment=ev.query_sentiment,
            content_analysis=ev.content_analysis
        )
    
    # ============================================
    # STEP 6: Synthesize Answer
    # ============================================
    @step
    async def synthesize_answer(self, ev: AnalysisEvent) -> SynthesisEvent:
        """
        Generate final answer with content analysis enrichment.
        """
        logger.info("[Step 6] Synthesizing final answer")
        
        # Build enrichment context from analysis
        enrichment = self._build_enrichment_context(ev)
        
        # Synthesize answer
        final_answer = await self.subquery_operations.synthesize_final_answer(
            original_query=ev.original_query,
            sub_queries_and_contexts=ev.sub_queries_and_contexts,
            enrichment_context=enrichment
        )
        
        logger.info(f"Answer generated: {len(final_answer)} chars")
        
        return SynthesisEvent(
            original_query=ev.original_query,
            answer=final_answer,
            sub_queries=[sq for sq, _ in ev.sub_queries_and_contexts],
            keywords=ev.keywords,
            content_analysis=ev.content_analysis
        )
    
    # ============================================
    # STEP 7: Store and Return
    # ============================================
    @step
    async def store_and_return(self, ev: SynthesisEvent) -> StopEvent:
        """
        Store conversation in memory and return results.
        
        THIS is where memory is actually useful:
        - Stores Q&A for future sessions
        - Enables fact extraction (long-term memory)
        - Powers contextual responses in next queries
        """
        logger.info("[Step 7] Storing conversation in memory")
        
        # Store conversation - this is the MAIN use of memory
        memory = self.memory_manager.get_memory()
        # memory.put_messages([
        #     ChatMessage(role=MessageRole.USER, content=ev.original_query),
        #     ChatMessage(role=MessageRole.ASSISTANT, content=ev.answer)
        # ])
        
        user_msg = ChatMessage(
            role=MessageRole.USER,
            content=ev.original_query
        )
        
        assistant_msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=ev.answer
        )
        
        # Put messages in memory
        memory.put_messages([user_msg, assistant_msg])
        
        logger.info("✅ Conversation stored in short-term and long-term memory")
        
        # Build result
        result = {
            "answer": ev.answer,
            "sub_queries": ev.sub_queries,
            "original_query": ev.original_query,
            "keywords": ev.keywords,
            "content_analysis": ev.content_analysis,
        }
        
        logger.info("Workflow completed successfully")
        
        return StopEvent(result=result)
    
    # ============================================
    # Helper Methods
    # ============================================
    def _build_enrichment_context(self, ev: AnalysisEvent) -> str:
        """
        Build enrichment instructions for LLM based on analysis.
        
        USE CASES:
        1. Adjust tone based on query sentiment
        2. Ensure key entities are mentioned
        3. Structure answer around main themes
        """
        parts = []
        
        # Tone adjustment based on query sentiment
        query_sentiment = ev.query_sentiment.get("sentiment", "neutral")
        if query_sentiment == "negative" or "concern" in query_sentiment:
            parts.append("Note: Address the user's concerns directly and provide reassurance where appropriate.")
        elif query_sentiment == "positive" or "excited" in query_sentiment:
            parts.append("Note: Match the user's enthusiasm with an engaging response.")
        
        # Key entities to include
        entities = ev.content_analysis.get("entities", {})
        important_entities = []
        for entity_type, items in entities.items():
            if items and len(items) > 0:
                important_entities.extend(items[:3])  # Top 3 per type
        
        if important_entities:
            parts.append(f"Important entities to reference: {', '.join(important_entities[:5])}")
        
        # Themes to structure around
        themes = ev.content_analysis.get("themes", [])
        if themes:
            theme_names = [t.get("theme", "") for t in themes[:3]]
            parts.append(f"Structure your answer around these themes: {', '.join(theme_names)}")
        
        return "\n".join(parts) if parts else ""
