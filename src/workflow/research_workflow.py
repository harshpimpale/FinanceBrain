from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core import VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from src.workflow.events import (
    QueryEvent,
    SubQueriesEvent,
    RetrievalEvent,
    SynthesisEvent
)
from src.llm.models import get_llm
from src.memory.memory_manager import MemoryManager
from src.retrieval.retriever import RetrieverTool
from src.tools.keyword_extracter import KeywordExtractorTool
from src.retrieval.subquery import SubqueriesOperations
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ResearchWorkflow(Workflow):
    """
    Multi-step research workflow with query decomposition, retrieval, and synthesis.
    Integrates memory and all tools.
    """
    
    def __init__(self, 
                 index: VectorStoreIndex,
                 memory_manager: MemoryManager,
                 timeout: int = 120):
        super().__init__(timeout=timeout)
        self.index = index
        self.memory_manager = memory_manager
        self.llm = get_llm()
        self.retriever_tool = RetrieverTool(index)
        self.keyword_tool = KeywordExtractorTool()
        self.subquery_operations = SubqueriesOperations(index=index)
        print("ResearchWorkflow initialized")
    
    @step
    async def query_decomposition(self, ev: StartEvent) -> QueryEvent:
        """Step 1: Extract keywords and check memory"""
        query = ev.query
        logger.info(f"[Step 1] Query Decomposition - Query: {query}")
        
        # Extract keywords
        keywords = await self.keyword_tool.extract(query)
        logger.info(f"Extracted keywords: {keywords}")
        
        # Get memory context
        memory_context = self.memory_manager.get_context()
        if memory_context:
            logger.info("Retrieved memory context")
        
        return QueryEvent(query=query)
    
    @step
    async def create_sub_queries(self, ev: QueryEvent) -> SubQueriesEvent:
        """Step 2: Decompose complex query into sub-queries"""
        query = ev.query
        logger.info(f"[Step 2] Creating sub-queries for: {query}")
        
        sub_queries = await self.subquery_operations.create_sub_queries(query)
        
        logger.info(f"Generated {len(sub_queries)} sub-queries: {sub_queries}")
        
        return SubQueriesEvent(
            sub_queries=sub_queries,
            original_query=query
        )
    
    @step
    async def retrieve_for_sub_queries(self, ev: SubQueriesEvent) -> RetrievalEvent:
        """Step 3: Retrieve context for each sub-query"""
        logger.info(f"[Step 3] Retrieving context for {len(ev.sub_queries)} sub-queries")
        
        sub_queries_and_contexts = await self.subquery_operations.retrieve_for_sub_queries(ev.sub_queries)
        
        logger.info("All contexts retrieved")
        
        return RetrievalEvent(
            original_query=ev.original_query,
            sub_queries_and_contexts=sub_queries_and_contexts
        )
    
    @step
    async def synthesize_answer(self, ev: RetrievalEvent) -> StopEvent:
        """Step 4: Synthesize final answer from all contexts"""
        logger.info("[Step 4] Synthesizing final answer")
        
        final_answer = await self.subquery_operations.synthesize_final_answer(
            original_query=ev.original_query,
            sub_queries_and_contexts=ev.sub_queries_and_contexts
        )

        logger.info("Final answer synthesized")
        
        # Store in memory using proper ChatMessage objects
        memory = self.memory_manager.get_memory()
        memory.put_messages([
            ChatMessage(role=MessageRole.USER, content=ev.original_query),
            ChatMessage(role=MessageRole.ASSISTANT, content=final_answer)
        ])
        
        logger.info("Conversation stored in memory")
        
        return StopEvent(result={
            "answer": final_answer,
            "sub_queries": [sq for sq, _ in ev.sub_queries_and_contexts],
            "original_query": ev.original_query
        })
