import asyncio
from src.config.settings import Settings
from src.tools.keyword_extracter import extract_keywords

async def testing():
    "key word to test the retrieval and LLM integration"
    query = "Explain the concept of financial leverage and its impact on a company's return on equity (ROE)."
    # Load documents and create/load vector index
    keywords = await extract_keywords(query)
    print(f"Extracted Keywords: {keywords}")



if __name__ == "__main__":
    # Run the async test function
    asyncio.run(testing())
