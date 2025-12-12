import streamlit as st
import asyncio

from src.loader.document_loader import DocumentLoader
from src.workflow.research_workflow import ResearchWorkflow
from src.memory.memory_manager import MemoryManager


# --------- Helpers ---------
async def run_workflow(query: str):
    """
    Executes your ResearchWorkflow asynchronously and returns the final result.
    """
    loader = DocumentLoader()

    # Load index only once per session
    if "index" not in st.session_state:
        st.session_state.index = loader.load_documents()

    # Memory manager - persistent per session
    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = MemoryManager(session_id="main_session")

    # Create workflow instance
    wf = ResearchWorkflow(
        index=st.session_state.index,
        memory_manager=st.session_state.memory_manager,
        timeout=180
    )

    # Run the workflow async
    result = await wf.run(query=query)
    return result


# --------- Streamlit UI ---------
st.set_page_config(page_title="FinanceBrain Chat", page_icon="💬")

st.title("💬 FinanceBrain — Research Chatbot")

# Store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages already stored
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# Chat input
query = st.chat_input("Ask something about finance, investing, economics...")

if query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.write("⏳ Running research workflow...")

    # Run workflow asynchronously inside Streamlit
    final_result = asyncio.run(run_workflow(query))

    # Extract answer
    answer = final_result["answer"] or "⚠️ No answer returned by workflow"

    # Update assistant message
    with st.chat_message("assistant"):
        st.write(answer)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
