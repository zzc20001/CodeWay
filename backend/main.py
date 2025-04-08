import os
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.models.openai.like import OpenAILike
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.vectordb.lancedb import LanceDb, SearchType
from dotenv import load_dotenv
from agno.embedder.huggingface import HuggingfaceCustomEmbedder
from agno.reranker.cohere import CohereReranker
from agno.knowledge.website import WebsiteKnowledgeBase
load_dotenv()

# Create a knowledge base of PDFs from URLs
knowledge_base = WebsiteKnowledgeBase(
    urls=["https://gh.catmak.name/https://raw.githubusercontent.com/jax-ml/jax/refs/heads/main/docs/export/shape_poly.md"],
    max_links=1,
    # Use LanceDB as the vector database and store embeddings in the `recipes` table
    vector_db=LanceDb(
        table_name="recipes",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=OpenAIEmbedder(),
        reranker=CohereReranker(model="rerank-multilingual-v3.0"),
    ),
)
# Load the knowledge base: Comment after first run as the knowledge base is already loaded
knowledge_base.load()

agent = Agent(
    model=OpenAIChat(base_url=os.environ.get("OPENAI_API_BASE")),
    knowledge=knowledge_base,
    # Add a tool to search the knowledge base which enables agentic RAG.
    # This is enabled by default when `knowledge` is provided to the Agent.
    search_knowledge=True,
    # show_tool_calls=True,
    # markdown=True,
)
agent.print_response(
    "Introduce dimension variable with constraints.", stream=True
)