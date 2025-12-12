from llama_index.core.workflow import Event

class QueryEvent(Event):
    """Event carrying the user query"""
    query: str

class SubQueriesEvent(Event):
    """Event carrying decomposed sub-queries"""
    sub_queries: list[str]
    original_query: str

class RetrievalEvent(Event):
    """Event carrying retrieved context"""
    original_query: str
    sub_queries_and_contexts: list[tuple[str, str]]

class SynthesisEvent(Event):
    """Event carrying all contexts for final synthesis"""
    original_query: str
    sub_queries_and_contexts: list[tuple[str, str]]
