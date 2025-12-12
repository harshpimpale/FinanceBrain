## Finance Brain

This project implements a memory-augmented, query-planning research assistant on top of LlamaIndex workflows and a Streamlit UI. It is designed to show clear separation of concerns (memory, retrieval, tools, workflow, UI) and to match the assignment requirements around short/long-term memory, query decomposition, tools, and rate limiting.

## Project Overview

The assistant is a document-focused RAG system (e.g., for `adobe-annual-report.pdf`) that can:
- Remember facts across turns (short- and long-term memory).
- Decompose complex questions into sub-questions.
- Retrieve and summarize relevant context from a vector store.
- Analyze content (themes, sentiment, entities, structure).
- Respect LLM API rate limits via a central router.
- Provide an interactive Streamlit chat UI.

The codebase is structured so each concern (LLM, embeddings, memory, retrieval, tools, workflow, UI) is modular and replaceable.

***

## Directory Structure

A typical layout:

```bash
FinanceBrain/
├── app.py                   # Streamlit UI (chat interface)
├── src/
│   ├── config/
│   │   └── settings.py      # Config: API keys, models, paths, limits
│   ├── llm/
│   │   ├── models.py        # Groq LLM + embedding model factory
│   │   └── rate_limiter.py  # Central async rate limiter
│   ├── memory/
│   │   ├── memory_loader.py # Chroma-backed vector store for memory
│   │   └── memory_manager.py# Short + long term memory manager
│   ├── loader/
│   │   └── document_loader.py # Loads PDFs, builds VectorStoreIndex
│   ├── tools/
│   │   ├── keyword_extracter.py
│   │   ├── summarizer.py
│   │   ├── content_analyzer.py
│   │   └── retriever.py
│   ├── retrieval/
│   │   ├── retriever.py     # Query-time retrieval from doc index
│   │   └── subquery.py      # Query decomposition & synthesis helpers
│   ├── workflow/
│   │   ├── events.py
│   │   └── research_workflow.py
│   └── utils/
│       └── logger.py
├── data/
│   └── documents/
│       └── adobe-annual-report.pdf
├── VectorDB/
│   ├── chroma_db/
│   └── MemoryBase/
```

You can adjust names, but this structure makes it easy to understand where things live.

***

## Configuration (`src/config/settings.py`)

`Settings` is a simple config holder to keep magic constants and secrets out of the code:

- **API keys**: `GROQ_API_KEY`, `GOOGLE_API_KEY` (or whatever you use).
- **Models**:
  - `LLM_MODEL` for the chat/completion model (e.g., `llama3-70b-8192`).
  - `EMBEDDING_MODEL` for vector embeddings (e.g., `BAAI/bge-small-en-v1.5` or `models/text-embedding-004`).
- **Paths**:
  - `VECTOR_DB_PATH` – persistent Chroma DB for documents.
  - `MEMORY_DB_PATH` – persistent Chroma DB for long-term memory.
  - `DOCUMENTS_PATH` – folder with PDFs (e.g., `adobe-annual-report.pdf`).
- **Performance & memory**:
  - `MEMORY_TOKEN_LIMIT` – how many tokens are kept in short-term chat memory.
  - `MAX_FACTS` – how many extracted facts to keep in long-term memory.
  - `SIMILARITY_TOP_K` – how many nodes to retrieve from vector store.
  - `MAX_REQUESTS_PER_MINUTE` – rate limiting budget per minute.
  - `WORKFLOW_TIMEOUT` – max seconds for a workflow run.

Everything is read from `.env` so you can easily change behavior between dev / demo / production without code changes.

***

## LLM & Embeddings Layer (`src/llm/`)

### `models.py`

This module centralizes LLM and embedding model initialization.

- `get_llm()` returns a singleton LLM instance (e.g., Groq’s `llama3-70b-8192`).
- `get_embed_model()` returns a singleton embedding model (e.g., HuggingFace or Google embedding).

Why centralize?
- Ensures only one instance of each is created and reused.
- Keeps API keys and model names in one place (`settings`).
- Makes it easy to mock models in tests.

### `rate_limiter.py`

Implements an async rate limiter:

- Tracks timestamps of recent LLM calls.
- If more than `MAX_REQUESTS_PER_MINUTE` are attempted in the last 60 seconds, it sleeps before allowing new calls.
- Exposes `call_with_limit(func, *args, **kwargs)`:
  - Wraps any LLM call (`llm.acomplete`, `llm.complete`, synthesis calls, etc.).
  - All tools and workflow steps use this wrapper for API calls.

This ensures:
- You don’t exceed API quotas.
- You can tune throughput with a single config.

***

## Memory System (`src/memory/`)

### `memory_loader.py`

- Wraps Chroma’s `PersistentClient` to manage a dedicated collection for **memory**.
- Creates or loads a collection at `MEMORY_DB_PATH`.
- Returns a `ChromaVectorStore` suitable for LlamaIndex `VectorMemoryBlock`.

This isolates vector-store setup from the rest of the memory logic.

### `memory_manager.py`

Implements **both** short-term and long-term memory:

- Uses `Memory.from_defaults` from LlamaIndex with:
  - A **token-limited sliding window** for short-term chat history.
  - Custom memory blocks for long-term memory:
    - `FactExtractionMemoryBlock`:
      - Uses the LLM to automatically pull out important facts from the conversation and store them as structured entries.
    - `VectorMemoryBlock`:
      - Stores memories in a vector store (Chroma) so they can be semantically retrieved later.

Responsibilities:
- Initialize the memory with proper blocks.
- Provide `get_memory()` to access the underlying `Memory` object.
- Provide `get_context()` to fetch the current memory context (for debugging, or to include as additional context in prompts).
- In the workflow, final user/assistant messages are written to memory as `ChatMessage` objects, which triggers fact extraction and long-term storage.

This satisfies the assignment’s requirement for:
- Short-term memory (per-session context).
- Long-term memory (learned facts reused in future sessions).

***

## Document Loading & Indexing (`src/loader/document_loader.py`)

`DocumentLoader` is responsible for converting raw files into an index:

- Uses `SimpleDirectoryReader` + `PyMuPDFReader` to load `.pdf` files from `DOCUMENTS_PATH`.
- Uses Chroma as the vector store for **document** embeddings.
- On first run:
  - Reads all PDFs into `Document` objects.
  - Builds a `VectorStoreIndex` with the chosen `embed_model`.
- On subsequent runs:
  - Reuses the existing Chroma collection and constructs a `VectorStoreIndex` from the vector store, avoiding recomputing embeddings.

The resulting `VectorStoreIndex` is used for retrieval during the workflow.

***

## Tools Layer (`src/tools/`)

These are reusable, composable utilities the workflow calls. Each tool is independent and can be swapped or extended.

### Keyword Extractor (`keyword_extracter.py`)

- Wraps LlamaIndex’s `KeywordExtractor` with:
  - Custom prompt: “Return ONLY a comma-separated list of keywords…”
  - Configurable `max_keywords`.
- Input: a string (`text`).
- Output: `List[str]` of cleaned keywords.
- Used at the beginning of the workflow to:
  - Enhance retrieval.
  - Optionally feed into memory or analytics.

### Summarizer (`summarizer.py`)

Provides **multiple summarization strategies**:

- `tree_summarize(text, query=...)`
  - Uses `TreeSummarize` response synthesizer for hierarchical summarization.
  - Good for very long contexts or documents.
- `extractive_summary(text, num_sentences=N)`
  - Identifies key sentences, often by prompting the LLM to pick the N most important ones.
- `abstractive_summary(text, max_words=...)`
  - Short, LLM-generated summaries with a word budget.
- `summarize_with_bullets(text, num_points=N)`
  - Returns a list of bullet-point summaries.
- `auto_summarize(text, target_length="short|medium|long")`
  - Chooses strategy based on the length of the input text:
    - Short text → extractive.
    - Medium → abstractive.
    - Long → tree summarization.

The workflow uses `auto_summarize` to **compress retrieved contexts** before final synthesis, reducing token usage while keeping important details.

### Content Analyzer (`content_analyzer.py`)

A higher-level tool for document/query analysis:

- `extract_themes(text, num_themes=5)`
  - Finds main themes + short descriptions.
- `analyze_sentiment(text)`
  - Returns overall sentiment, confidence, reasoning.
- `extract_entities(text)`
  - Extracts people, organizations, locations, dates, and key numbers.
- `analyze_structure(text)`
  - Estimates document type, style, key sections, and basic stats (word count, sentences, etc.).
- `comprehensive_analysis(text)`
  - Runs all of the above and combines the results into a structured report.

In the workflow, you can:
- Use sentiment on the query to adapt tone.
- Use theme/entity analysis to enrich answers or for debugging.

### Retriever (`tools/retriever.py`)

- Wraps `VectorStoreIndex.as_retriever` into a simple class.
- `retrieve(query)`: returns a list of `NodeWithScore` for the top-k similar nodes.
- `get_text_from_nodes(nodes)`: concatenates node text for further processing (summarization / synthesis).

This is separated from `retrieval/subquery.py` so the same tool can be reused in other workflows or components.

***

## Retrieval & Query Planning (`src/retrieval/`)

### `retriever.py` (document retrieval)

Thin convenience wrapper to keep retrieval concerns local.

### `subquery.py` (query planning & synthesis)

Responsible for query decomposition and multi-step retrieval logic:

- `create_sub_queries(query)`:
  - Calls the LLM with a “decompose into sub-questions” prompt.
  - Parses numbered output (e.g., “1. … 2. …”) into a list of sub-queries.
- `retrieve_for_sub_queries(sub_queries)`:
  - For each sub-query, uses the document retriever to fetch relevant nodes.
  - Returns a list of `(sub_query, context_text)` tuples.
- `synthesize_final_answer(original_query, sub_queries_and_contexts)`:
  - Builds a prompt that includes:
    - Original question.
    - Each sub-question with its summarized context.
  - Calls the LLM once to generate a coherent final answer.

This module isolates the logic for:
- Query planning & decomposition.
- Orchestrating multi-step retrieval.
- Final answer synthesis.

***

## Workflow Orchestration (`src/workflow/`)

### Events (`events.py`)

Defines strongly-typed events that travel between workflow steps:

- `QueryEvent` – carries the original query plus metadata (keywords, sentiment).
- `SubQueriesEvent` – holds the list of sub-queries and the original query.
- `RetrievalEvent` – holds sub-queries paired with their (possibly summarized) contexts.
- `SynthesisEvent` – carries the final answer and sub-queries list.

This makes the workflow’s data flow explicit and easy to follow.

### Research Workflow (`research_workflow.py`)

This is the **heart of the system**: a LlamaIndex `Workflow` that wires together memory, tools, and retrieval into a coherent pipeline.

Recommended flow:

1. **`analyze_query` (StartEvent → QueryEvent)**  
   - Input: raw query (from StartEvent).
   - Actions:
     - Extract keywords (keyword tool).
     - Analyze sentiment (content analyzer).
     - Optionally consult memory context (memory manager).
   - Output: `QueryEvent(query, keywords, sentiment)`.

2. **`decompose_query` (QueryEvent → SubQueriesEvent)**  
   - Uses `SubqueriesOperations.create_sub_queries`.
   - Breaks a complex question into smaller ones.
   - Output: `SubQueriesEvent(sub_queries, original_query, keywords)`.

3. **`retrieve_contexts` (SubQueriesEvent → RetrievalEvent)**  
   - Uses `SubqueriesOperations.retrieve_for_sub_queries`.
   - For each sub-query, retrieves relevant document text.
   - Output: `RetrievalEvent(original_query, sub_queries_and_contexts, keywords)`.

4. **`summarize_contexts` (RetrievalEvent → RetrievalEvent)**  
   - For each `(sub_query, raw_context)`:
     - Calls summarizer’s `auto_summarize` to compress the context.
   - Replaces long contexts with short summaries.
   - Output: updated `RetrievalEvent` with summarized contexts.

5. **`synthesize_answer` (RetrievalEvent → SynthesisEvent)**  
   - Calls `SubqueriesOperations.synthesize_final_answer`.
   - Produces a final, coherent answer that references all sub-questions.
   - Output: `SynthesisEvent(original_query, answer, sub_queries, keywords)`.

6. **`store_and_return` (SynthesisEvent → StopEvent)**  
   - Writes user question and answer into memory as `ChatMessage`s.
     - Short-term: for immediate context.
     - Long-term: via fact extraction & vector memory.
   - Returns a `StopEvent` with a result dict:
     - `answer`
     - `sub_queries`
     - `original_query`
     - `keywords`
     - (optionally) sentiment or extra metadata.

The workflow thus satisfies:
- **Memory system**: short + long-term.
- **Query planning**: explicit decomposition step.
- **Tool usage**: keyword, summarization, analysis integrated at appropriate stages.
- **Rate limiting**: all LLM calls go through the rate-limiting wrapper.

***

## Streamlit UI (`app.py`)

The UI turns everything into an interactive chat experience.

Key elements:

- **Session state**:
  - `messages`: list of `{role, content, ...}` for chat history.
  - `index`: cached `VectorStoreIndex` (documents).
  - `memory_manager`: per-session `MemoryManager`.
- **Chat interface**:
  - Uses `st.chat_message("user"/"assistant")` to render chat bubbles.
  - `st.chat_input(...)` for user input.
- **Workflow execution**:
  - When a user submits a query:
    - The app appends it to `messages`.
    - Shows a spinner while running `workflow.run(query=...)`.
    - Displays the final answer.
    - Optionally shows:
      - Extracted sub-queries in an expander.
      - Keywords or other metadata.
- **Caching**:
  - Uses `st.cache_resource` (or similar) to:
    - Avoid re-building the document index on each request.
    - Optionally reuse a single workflow per process.

This provides a user-friendly front-end for your agent and is easy to extend (add analytics page, settings page, etc.).