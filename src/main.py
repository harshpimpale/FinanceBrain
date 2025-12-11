import asyncio
from src.config.settings import Settings
from src.tools.keyword_extracter import extract_keywords
from src.retrieval.subquery import handle_subqueries
from src.loader.document_loader import DocumentLoader

async def testing():
    "key word to test the retrieval and LLM integration"

    loader = DocumentLoader()
    index = loader.load_documents()

    query = "Explain the concept of financial leverage and its impact on a company's return on equity (ROE)."
    # Load documents and create/load vector index
    keywords = await extract_keywords(query)
    print(f"Extracted Keywords: {keywords}")

    final_answer = await handle_subqueries(index, query)
    print(f"Final Answer: {final_answer}")




if __name__ == "__main__":
    # Run the async test function
    asyncio.run(testing())
