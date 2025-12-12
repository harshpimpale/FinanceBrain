# app.py

import streamlit as st
import asyncio
import nest_asyncio

# Apply nest_asyncio to fix event loop issues
nest_asyncio.apply()

from src.loader.document_loader import DocumentLoader
from src.workflow.research_workflow import ResearchWorkflow
from src.memory.memory_manager import MemoryManager


# --------- Cached Resources ---------
@st.cache_resource
def initialize_system():
    """Initialize and cache the document loader and index"""
    loader = DocumentLoader()
    index = loader.load_documents()
    return index


# --------- Helpers ---------
async def run_workflow(query: str, index, memory_manager):
    """
    Executes your ResearchWorkflow asynchronously.
    """
    wf = ResearchWorkflow(
        index=index,
        memory_manager=memory_manager,
        timeout=180
    )
    
    result = await wf.run(query=query)
    return result


# --------- Streamlit UI ---------
st.set_page_config(page_title="FinanceBrain Chat", page_icon="💬")

st.title("💬 FinanceBrain — Research Chatbot")

# Initialize system
index = initialize_system()

# Memory manager - persistent per session
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = MemoryManager(session_id="main_session")

# Store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Show sub-queries if available
        if msg["role"] == "assistant" and "sub_queries" in msg:
            with st.expander("🔍 Query Decomposition"):
                for i, sq in enumerate(msg["sub_queries"], 1):
                    st.markdown(f"{i}. {sq}")

# Chat input
if query := st.chat_input("Ask something about finance, investing, economics..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    # Process query
    with st.chat_message("assistant"):
        with st.spinner("⏳ Running research workflow..."):
            try:
                # Run workflow with nest_asyncio support
                final_result = asyncio.run(
                    run_workflow(
                        query, 
                        index, 
                        st.session_state.memory_manager
                    )
                )
                
                # Extract answer
                answer = final_result.get("answer", "⚠️ No answer returned")
                sub_queries = final_result.get("sub_queries", [])
                keywords = final_result.get("keywords", [])
                
                # Display answer
                st.write(answer)
                
                # Show sub-queries
                if sub_queries:
                    with st.expander("🔍 Query Decomposition"):
                        for i, sq in enumerate(sub_queries, 1):
                            st.markdown(f"{i}. {sq}")
                
                # Show keywords
                if keywords:
                    with st.expander("🔑 Extracted Keywords"):
                        st.write(", ".join(keywords))
                
                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sub_queries": sub_queries,
                    "keywords": keywords
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
