import asyncio
import os
import sys
from llama_index.core import Settings

# Add the project root to the python path to ensure imports work correctly
# This assumes the script is located in src/main.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.loader.document_loader import DocumentLoader
from src.retrieval.retriever import RetrieverTool
from src.llm.models import get_llm

Settings.llm = get_llm()

async def test_retriever_tool():
    """
    Test run for RetrieverTool.
    Loads the index and performs a sample retrieval.
    """
    print("Starting RetrieverTool test run...")
    
    try:
        # Initialize DocumentLoader to get the index
        print("Initializing DocumentLoader...")
        loader = DocumentLoader()
        
        # Load the index (this will load from disk or create new from documents)
        print("Loading index...")
        # Note: This might raise FileNotFoundError if ./data/documents doesn't exist and index is empty
        try:
            index = loader.load_documents()
            print("Index loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading documents: {e}")
            print("Please ensure 'data/documents' directory exists and contains documents, or a vector store is already created.")
            return
        
        if not index:
            print("Failed to load index.")
            return

        # Initialize RetrieverTool
        print("Initializing RetrieverTool...")
        retriever_tool = RetrieverTool(index=index)
        
        # Define a test query
        test_query = "What is the financial outlook?"
        print(f"Running test query: '{test_query}'")
        
        # Perform retrieval
        query_engine = index.as_query_engine()
        result = await query_engine.aquery(test_query)
        
        # Display results
        print("\n--- Retrieval Results ---")
        # print(f"Number of nodes retrieved: {len(result)}")
        print(result)
        # if nodes:
        #     for i, node in enumerate(nodes, 1):
        #         print(f"\nResult {i}:")
        #         print(f"Score: {node.score}")
        #         # Print first 200 chars of content
        #         content_preview = node.node.get_content()[:200].replace('\n', ' ')
        #         print(f"Content: {content_preview}...")
                
        #     # Test get_text_from_nodes
        #     print("\n--- Combined Text ---")
        #     combined_text = retriever_tool.get_text_from_nodes(nodes)
        #     print(f"{combined_text[:300]}...")
        # else:
        #     print("No documents retrieved.")
            
        print("\nTest run completed successfully.")
        
    except Exception as e:
        print(f"An error occurred during the test run: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the async test function
    asyncio.run(test_retriever_tool())
