# src/workflow/events.py

from llama_index.core.workflow import Event
from typing import List, Dict, Any


class QueryEvent(Event):
    """Event carrying analyzed query with metadata"""
    query: str
    keywords: List[str] = []
    sentiment: Dict[str, Any] = {}


class SubQueriesEvent(Event):
    """Event carrying decomposed sub-queries"""
    sub_queries: List[str]
    original_query: str
    keywords: List[str] = []  # Pass keywords forward


class RetrievalEvent(Event):
    """Event carrying retrieved contexts"""
    original_query: str
    sub_queries_and_contexts: List[tuple[str, str]]
    keywords: List[str] = []  # Pass keywords forward


class SynthesisEvent(Event):
    """Event carrying synthesized answer"""
    original_query: str
    answer: str
    sub_queries: List[str]
    keywords: List[str] = []  # Pass keywords forward
