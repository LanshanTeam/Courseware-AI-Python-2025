import asyncio

from agno.agent import Agent
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.vectordb.lancedb import LanceDb, SearchType

# Load Agno documentation into Knowledge
knowledge = Knowledge(
    vector_db=LanceDb(
        uri="tmp/financial",
        table_name="financial",
        search_type=SearchType.hybrid,
        # Use OpenAI for embeddings
        embedder=OpenAIEmbedder(id="text-embedding-3-small", dimensions=1536),
    ),
)

asyncio.run(
    knowledge.add_content_async(
        path="data/公司日常报销流程及步骤.docx"
    )
)

agent = Agent(
    name="Agno Assist",
    model=OpenAIChat(id="gpt-4o"),
    instructions=[
        "Use tables to display data.",
        "Include sources in your response.",
        "Search your knowledge before answering the question.",
        "Only include the output in your response. No other text.",
    ],
    knowledge=knowledge,
    tools=[ReasoningTools(add_instructions=True)],
    add_datetime_to_context=True,
    markdown=True,
)

if __name__ == "__main__":
    agent.print_response(
        "公司日常报销流程及步骤?",
        stream=True,
        show_full_reasoning=True,
        stream_events=True,
    )
