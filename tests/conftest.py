import pytest
import asyncio
import os
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# Set environment variables - tests will skip if these aren't set
os.environ.setdefault("API_KEY", os.getenv("API_KEY", ""))
os.environ.setdefault("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
os.environ.setdefault("MAX_REQUESTS_PER_MINUTE", "60")

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")  # Changed to session scope to match real_vector_index
def sample_document_text():
    """Sample Adobe annual report text for testing"""
    return """
    Adobe Inc. is a multinational software company specializing in multimedia 
    and creativity software products. In fiscal year 2022, Adobe reported 
    total revenue of $17.61 billion, representing a 15% year-over-year growth.
    
    Key products include Adobe Creative Cloud with $12.84 billion in revenue,
    Adobe Document Cloud with $2.18 billion, and Adobe Experience Cloud with 
    $4.42 billion in revenue.
    
    The company's subscription business model has proven successful, with 
    recurring revenue accounting for 95% of total revenue. Adobe serves 
    creative professionals, enterprises, and individual consumers worldwide.
    
    Financial highlights include a gross profit margin of 88%, operating 
    income of $6.1 billion, and diluted earnings per share of $13.71.
    """

@pytest.fixture(scope="session")  # Changed to session scope
def sample_documents(sample_document_text):
    """Create sample documents for indexing"""
    return [
        Document(text=sample_document_text, id_="doc1"),
        Document(
            text="Adobe's digital media segment continues to drive growth through "
                 "creative cloud subscriptions and strategic acquisitions. The company "
                 "focuses on innovation in AI and machine learning.",
            id_="doc2"
        ),
        Document(
            text="The company invested $2.5 billion in R&D in 2022, focusing on "
                 "AI and machine learning capabilities. Adobe Sensei powers intelligent "
                 "features across all product lines.",
            id_="doc3"
        ),
        Document(
            text="Adobe Experience Cloud helps enterprises deliver personalized customer "
                 "experiences. Major clients include Fortune 500 companies worldwide.",
            id_="doc4"
        )
    ]

@pytest.fixture(scope="session")
def real_llm():
    """Get REAL LLM instance - reuse across all tests"""
    if not os.getenv("API_KEY"):
        pytest.skip("API_KEY not set - required for LLM tests")
    
    from src.llm.models import get_llm
    # Force reload to ensure we get a fresh instance if needed, or rely on singleton
    llm = get_llm()
    print(f"\n✅ Created real LLM instance: {llm.model}")
    return llm

@pytest.fixture(scope="session")
def real_embed_model():
    """Get REAL embedding model instance - reuse across all tests"""
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set - required for embedding tests")
    
    from src.llm.models import get_embed_model
    embed_model = get_embed_model()
    print(f"\n✅ Created real embedding model")
    return embed_model

@pytest.fixture(scope="session")
def real_vector_index(sample_documents, real_embed_model):
    """Create REAL vector index - cached for session"""
    print("\n📚 Creating vector index with 4 sample documents...")
    index = VectorStoreIndex.from_documents(
        sample_documents,
        embed_model=real_embed_model,
        show_progress=False
    )
    print("✅ Vector index created successfully")
    return index

@pytest.fixture
def real_memory_manager(temp_memory_path):
    """Create REAL memory manager with real vector store"""
    if not os.getenv("API_KEY") or not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("API keys not set - required for memory manager")
    
    import src.config.settings as settings_module
    
    # Use temp path for testing
    original_path = settings_module.settings.MEMORY_DB_PATH
    settings_module.settings.MEMORY_DB_PATH = temp_memory_path
    
    from src.memory.memory_manager import MemoryManager
    manager = MemoryManager(session_id="test_session")
    
    yield manager
    
    # Cleanup
    settings_module.settings.MEMORY_DB_PATH = original_path

@pytest.fixture
def temp_memory_path(tmp_path):
    """Temporary path for memory storage"""
    memory_path = tmp_path / "test_memory"
    memory_path.mkdir()
    return str(memory_path)

@pytest.fixture
def temp_vector_db_path(tmp_path):
    """Temporary path for vector database"""
    vector_path = tmp_path / "test_vector_db"
    vector_path.mkdir()
    return str(vector_path)

@pytest.fixture
def clean_rate_limiter():
    """Reset rate limiter between tests"""
    from src.llm.rate_limiter import rate_limiter
    rate_limiter.request_times.clear()
    rate_limiter.total_calls = 0
    yield rate_limiter
    rate_limiter.request_times.clear()
    rate_limiter.total_calls = 0

@pytest.fixture
async def mock_retriever_nodes():
    """Mock retriever nodes for tests that don't need real retrieval"""
    nodes = [
        NodeWithScore(
            node=TextNode(
                text="Adobe reported revenue of $17.61 billion in 2022",
                id_="node1"
            ),
            score=0.95
        ),
        NodeWithScore(
            node=TextNode(
                text="Creative Cloud revenue was $12.84 billion",
                id_="node2"
            ),
            score=0.89
        ),
        NodeWithScore(
            node=TextNode(
                text="Adobe's gross profit margin is 88%",
                id_="node3"
            ),
            score=0.82
        )
    ]
    return nodes