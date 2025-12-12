import asyncio
from src.loader.document_loader import DocumentLoader
from src.workflow.research_workflow import ResearchWorkflow
from src.memory.memory_manager import MemoryManager

async def testing():
    "key word to test the retrieval and LLM integration"

    loader = DocumentLoader()
    index = loader.load_documents()

    query = "Explain the concept of financial leverage and its impact on a company's return on equity (ROE)."
    
    # Initialize memory manager
    memory_manager = MemoryManager(session_id="main_session")

    # create workflow
    wf = ResearchWorkflow(index=index, memory_manager=memory_manager, timeout=180)

    # create a start event with the user query
    # start = StartEvent(query="Compare the efficiency and cost of solar vs wind energy")

    # run workflow (API may differ by llama-index version; often .run exists)
    stop_event = await wf.run(query="Compare the efficiency and cost of solar vs wind energy")
    print(stop_event)

    # stop_event.result should contain your final output
    # print("Workflow result:", stop_event.result)



if __name__ == "__main__":
    # Run the async test function
    asyncio.run(testing())
