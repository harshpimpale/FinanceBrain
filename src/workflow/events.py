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
    keywords: List[str] = []
    sentiment: Dict[str, Any] = {}  # Add sentiment to pass forward


class RetrievalEvent(Event):
    """Event carrying retrieved contexts"""
    original_query: str
    sub_queries_and_contexts: List[tuple[str, str]]
    keywords: List[str] = []
    sentiment: Dict[str, Any] = {}  # Add sentiment to pass forward


class AnalysisEvent(Event):
    """Event carrying content analysis results"""
    original_query: str
    sub_queries_and_contexts: List[tuple[str, str]]
    keywords: List[str] = []
    query_sentiment: Dict[str, Any] = {}  # Sentiment of the user's query
    content_analysis: Dict[str, Any] = {}  # Analysis of retrieved content (entities, themes, sentiment)


class SynthesisEvent(Event):
    """Event carrying synthesized answer with analysis"""
    original_query: str
    answer: str
    sub_queries: List[str]
    keywords: List[str] = []
    content_analysis: Dict[str, Any] = {}  # Include analysis in final result
