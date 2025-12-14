


from src.llm.models import get_llm
from src.llm.rate_limiter import rate_limiter
from src.retrieval.retriever import RetrieverTool
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


DECOMPOSE_PROMPT = """
Given the following complex question, break it down into 2-4 simpler sub-questions that need to be answered sequentially.

Question: {query}

Return ONLY the sub-questions as a numbered list, one per line.
Example format:
1. First sub-question?
2. Second sub-question?
3. Third sub-question?
"""



SYNTHESIS_PROMPT = """
Based on the following sub-questions and their answers, provide a comprehensive answer to the original question.

Original Question: {original_query}

Sub-questions and Answers:
{sub_qa_pairs}

Provide a well-structured, comprehensive answer to the original question.
"""


class SubqueriesOperations:
    """
    This class handles subquery operations for retrieval tasks.
    SubQuery Planning:
    Subquery Retrieval:
    Subquery Synthesis:
    """

    def __init__(self, index):
        if index is None:
            raise ValueError("index is required for RetrieverTool")
        self.llm = get_llm()
        self.retriever_tool = RetrieverTool(index=index)

    async def create_sub_queries(self, query: str) -> list[str]:
        """Plan subquery based on the main query"""
        prompt = DECOMPOSE_PROMPT.format(query=query)
        
        # Decompose query
        response = await rate_limiter.call_with_limit(
            self.llm.complete,
            prompt
        )
        
        # Parse sub-queries
        response_text = str(response)
        lines = response_text.strip().split('\n')
        sub_queries = []
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                query_text = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                if query_text:
                    sub_queries.append(query_text)

        return sub_queries
    
    async def retrieve_for_sub_queries(self, sub_queries: list[str]) -> list[tuple[str, str]]:
        """Retrieve answers for each sub-query"""
        sub_queries_and_contexts = []
        
        for sub_query in sub_queries:
            logger.info(f"Retrieving for sub-query: {sub_query}")

            # Retrieve relevant documents
            nodes = await self.retriever_tool.retrieve(sub_query)
            context = self.retriever_tool.get_text_from_nodes(nodes)
            
            sub_queries_and_contexts.append((sub_query, context))
        
        return sub_queries_and_contexts
    

    async def synthesize_final_answer(
        self, 
        original_query: str, 
        sub_queries_and_contexts: list[tuple[str, str]],
        enrichment_context: str = ""  # enrichment from content analysis
    ) -> str:
        """
        Synthesize a comprehensive final answer from sub-query contexts.
        
        Args:
            original_query: The original user question
            sub_queries_and_contexts: List of (sub_query, context) tuples
            enrichment_context: Optional enrichment from content analysis
                               (entities, themes, sentiment guidance)
        
        Returns:
            Final synthesized answer
        """
        logger.info("Synthesizing final answer from sub-query contexts")
        
        # Format sub-Q&A pairs for the prompt
        sub_qa_text = ""
        for i, (sub_q, context) in enumerate(sub_queries_and_contexts, 1):
            sub_qa_text += f"\n{i}. Sub-question: {sub_q}\n"
            # Limit context length to avoid token overflow
            context_preview = context[:800] + "..." if len(context) > 800 else context
            sub_qa_text += f"   Answer: {context_preview}\n"
        
        # Build synthesis prompt with optional enrichment
        enrichment_section = ""
        if enrichment_context:
            enrichment_section = f"\nAdditional Context:\n{enrichment_context}\n"
        
        prompt = SYNTHESIS_PROMPT.format(
            original_query=original_query,
            sub_qa_pairs=sub_qa_text,
            enrichment_context=enrichment_section
        )
        
        # Generate final answer with rate limiting
        response = await rate_limiter.call_with_limit(
            self.llm.complete,
            prompt
        )
        
        final_answer = str(response).strip()
        logger.info(f"Final answer synthesized: {len(final_answer)} chars")
        
        return final_answer


# Convenience function for standalone usage
async def handle_subqueries(
    index, 
    query: str,
    enrichment_context: str = ""
) -> str:
    """
    Handle complete subquery pipeline end-to-end.
    
    Args:
        index: VectorStoreIndex for retrieval
        query: User's original question
        enrichment_context: Optional enrichment from analysis
        
    Returns:
        Final synthesized answer
    """
    logger.info(f"Handling subqueries for: {query}")
    
    subquery_ops = SubqueriesOperations(index=index)
    
    # Step 1: Decompose query
    sub_queries = await subquery_ops.create_sub_queries(query)
    
    # Step 2: Retrieve contexts
    sub_queries_and_contexts = await subquery_ops.retrieve_for_sub_queries(sub_queries)
    
    # Step 3: Synthesize final answer
    final_answer = await subquery_ops.synthesize_final_answer(
        original_query=query,
        sub_queries_and_contexts=sub_queries_and_contexts,
        enrichment_context=enrichment_context
    )
    
    return final_answer