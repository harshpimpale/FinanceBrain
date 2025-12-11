from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from src.config.settings import settings


class RetrieverTool:
    """Tool for retrieving relevant documents"""
    
    def __init__(self, index: VectorStoreIndex, top_k: int = settings.SIMILARITY_TOP_K):
        self.index = index
        self.top_k = top_k
        self.retriever = index.as_retriever(similarity_top_k=self.top_k)
        print(f"RetrieverTool initialized with top_k={self.top_k}")
    
    async def retrieve(self, query: str) -> list[NodeWithScore]:
        """Retrieve relevant nodes for a query"""
        print(f"Retrieving documents for query: {query[:100]}...")
        
        nodes = self.retriever.retrieve(query)
        
        print(f"Retrieved {len(nodes)} nodes")
        return nodes
    
    def get_text_from_nodes(self, nodes: list[NodeWithScore]) -> str:
        """Extract text from retrieved nodes"""
        texts = [node.node.get_content() for node in nodes]
        return "\n\n".join(texts)

# Convenience function
async def retrieve_documents(index: VectorStoreIndex, query: str, top_k: int = 5) -> list[NodeWithScore]:
    """Retrieve relevant documents"""
    retriever = RetrieverTool(index, top_k=top_k)
    return await retriever.retrieve(query)
