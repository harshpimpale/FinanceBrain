# src/workflow/research_workflow.py

from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Context
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
    SynthesisEvent
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ResearchWorkflow(Workflow):
    """
    Optimized research workflow following RAG best practices.
    """
    
    def __init__(self, 
                 index: VectorStoreIndex,
                 memory_manager: MemoryManager,
                 timeout: int = 120):
        super().__init__(timeout=timeout)
        self.index = index
        self.memory_manager = memory_manager
        self.llm = get_llm()
        
        # Initialize tools
        self.analyzer = ContentAnalyzerTool()
        self.summarizer = SummarizerTool()
        self.retriever_tool = RetrieverTool(index)
        self.keyword_tool = KeywordExtractorTool()
        self.subquery_operations = SubqueriesOperations(index=index)
        
        logger.info("ResearchWorkflow initialized")
    
    @step
    async def analyze_query(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        """
        Step 1: Analyze incoming query
        """
        query = ev.query
        logger.info(f"[Step 1] Query Analysis - Query: {query}")
        
        # Extract keywords
        keywords = await self.keyword_tool.extract(query)
        logger.info(f"Keywords: {keywords}")
        
        # Analyze sentiment
        sentiment = await self.analyzer.analyze_sentiment(query)
        logger.info(f"Sentiment: {sentiment['sentiment']}")
        
        # Get memory context
        memory_context = self.memory_manager.get_context()
        
        # Store in event instead of context
        return QueryEvent(
            query=query,
            keywords=keywords,
            sentiment=sentiment
        )
    
    @step
    async def decompose_query(self, ev: QueryEvent) -> SubQueriesEvent:
        """
        Step 2: Decompose complex query into sub-queries
        """
        query = ev.query
        logger.info(f"[Step 2] Query Decomposition: {query}")
        
        sub_queries = await self.subquery_operations.create_sub_queries(query)
        
        logger.info(f"Generated {len(sub_queries)} sub-queries")
        for i, sq in enumerate(sub_queries, 1):
            logger.info(f"  {i}. {sq}")
        
        return SubQueriesEvent(
            sub_queries=sub_queries,
            original_query=query,
            keywords=ev.keywords  # Pass keywords forward
        )
    
    @step
    async def retrieve_contexts(self, ev: SubQueriesEvent) -> RetrievalEvent:
        """
        Step 3: Retrieve relevant contexts
        """
        logger.info(f"[Step 3] Retrieving contexts for {len(ev.sub_queries)} sub-queries")
        
        sub_queries_and_contexts = await self.subquery_operations.retrieve_for_sub_queries(
            ev.sub_queries
        )
        
        logger.info("All contexts retrieved")
        
        return RetrievalEvent(
            original_query=ev.original_query,
            sub_queries_and_contexts=sub_queries_and_contexts,
            keywords=ev.keywords
        )
    
    @step
    async def summarize_contexts(self, ev: RetrievalEvent) -> RetrievalEvent:
        """
        Step 4: Summarize retrieved contexts
        """
        logger.info("[Step 4] Summarizing contexts")
        
        enriched = []
        for sq, ctx in ev.sub_queries_and_contexts:
            summary_result = await self.summarizer.auto_summarize(
                ctx, 
                target_length="medium"
            )
            
            logger.info(
                f"Compressed {summary_result['original_length']} → "
                f"{summary_result['summary_length']} words"
            )
            
            enriched.append((sq, summary_result["summary"]))
        
        # Update contexts
        ev.sub_queries_and_contexts = enriched
        return ev
    
    @step
    async def synthesize_answer(self, ev: RetrievalEvent) -> SynthesisEvent:
        """
        Step 5: Synthesize final answer
        """
        logger.info("[Step 5] Synthesizing final answer")
        
        final_answer = await self.subquery_operations.synthesize_final_answer(
            original_query=ev.original_query,
            sub_queries_and_contexts=ev.sub_queries_and_contexts
        )
        
        logger.info("Final answer synthesized")
        
        return SynthesisEvent(
            original_query=ev.original_query,
            answer=final_answer,
            sub_queries=[sq for sq, _ in ev.sub_queries_and_contexts],
            keywords=ev.keywords
        )
    
    @step
    async def store_and_return(self, ev: SynthesisEvent) -> StopEvent:
        """
        Step 6: Store in memory and return
        """
        logger.info("[Step 6] Storing in memory and returning result")
        
        # Store in memory
        memory = self.memory_manager.get_memory()
        memory.put_messages([
            ChatMessage(role=MessageRole.USER, content=ev.original_query),
            ChatMessage(role=MessageRole.ASSISTANT, content=ev.answer)
        ])
        
        logger.info("Conversation stored in memory")
        
        result = {
            "answer": ev.answer,
            "sub_queries": ev.sub_queries,
            "original_query": ev.original_query,
            "keywords": ev.keywords,
        }
        
        logger.info("Workflow completed successfully")
        
        return StopEvent(result=result)
