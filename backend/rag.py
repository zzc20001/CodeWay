import dotenv
import os
import nest_asyncio

# Load environment variables
dotenv.load_dotenv()

# Apply nest_asyncio early if needed by underlying libraries
nest_asyncio.apply()

from llama_index.core import Settings, VectorStoreIndex
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.callbacks import CallbackManager # Import CallbackManager

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# Import Langfuse handlers
from langfuse.callback import CallbackHandler as LangfuseLangchainCallbackHandler # Rename for clarity
from langfuse.llama_index import LlamaIndexCallbackHandler as LangfuseLlamaIndexCallbackHandler # Specific handler for LlamaIndex

# --- 1. Langfuse Configuration ---
langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY") # Use env var or default
langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY") # Use env var or default
langfuse_host = os.environ.get("LANGFUSE_HOST") # Use env var or default

# Create Langfuse handlers
# Handler for Langchain
langfuse_langchain_handler = LangfuseLangchainCallbackHandler(
    public_key=langfuse_public_key,
    secret_key=langfuse_secret_key,
    host=langfuse_host
    # You can add user_id, session_id, etc. here if needed
    # user_id="your_user_id",
    # session_id="your_session_id"
)

# Handler for LlamaIndex
langfuse_llama_index_handler = LangfuseLlamaIndexCallbackHandler(
    public_key=langfuse_public_key,
    secret_key=langfuse_secret_key,
    host=langfuse_host
)

# --- 2. Environment and API Keys ---
github_token = os.environ.get("GITHUB_TOKEN")
openai_api_base = os.environ.get("OPENAI_API_BASE") # Optional: if using a proxy/custom base
openai_api_key = os.environ.get("OPENAI_API_KEY")
local_api_base = os.environ.get("LOCAL_API_BASE")
if not github_token:
    raise ValueError("GITHUB_TOKEN environment variable not set.")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
if not langfuse_public_key or not langfuse_secret_key:
    print("Warning: Langfuse public or secret key not found. Tracing might not work.")
    # Decide if you want to raise an error or just continue without tracing
    # raise ValueError("Langfuse public or secret key not set.")


# --- 3. LlamaIndex Setup ---
owner = "stepbystepcode"
repo = "CodeWay"
branch = "main"
docs_folder_path = "docs"

# Configure LlamaIndex Settings with LLM and Callback Manager (including Langfuse)
mode="local"

Settings.callback_manager = CallbackManager([langfuse_llama_index_handler])

# Initialize Github Client and Reader
github_client = GithubClient(github_token=github_token, verbose=True)
reader = GithubRepositoryReader(
    github_client=github_client,
    owner=owner,
    repo=repo,
    use_parser=True,
    verbose=False,
    filter_directories=(
        [docs_folder_path],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
    filter_file_extensions=(
        [".md"],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
)

# Load documents
print("Loading documents from GitHub...")
try:
    documents = reader.load_data(branch=branch)
    if not documents:
        print(f"Warning: No documents found in {owner}/{repo}/{docs_folder_path} on branch {branch}.")
        # Decide how to proceed: exit, or continue with an empty index?
        # For this example, we'll let it potentially fail at index creation if empty.
    print(f"Loaded {len(documents)} documents.")
    # Optional: Print document details for verification
    # for doc in documents:
    #    print(f"- {doc.id_} ({doc.metadata.get('file_path', 'N/A')})")

except Exception as e:
    print(f"Error loading documents from GitHub: {e}")
    # Handle error appropriately, maybe exit
    exit(1)


# Create index and query engine (LlamaIndex operations will now be traced via Settings)
print("Creating vector store index...")
index = VectorStoreIndex.from_documents(documents) # Tracing happens here too
print("Index created.")
query_engine = index.as_query_engine() # Tracing for queries happens via this engine
print("Query engine ready.")

# --- 4. Langchain Tool Definition ---
def query_github_docs(query: str) -> str:
    """
    Uses LlamaIndex to query documentation within the stepbystepcode/CodeWay GitHub repository's docs folder.
    This function is traced by the LangfuseLlamaIndexCallbackHandler via LlamaIndex Settings.
    """
    print(f"\n LlamaIndex Tool executing query: {query}\n")
    response = query_engine.query(query)
    print(f"\n LlamaIndex Tool response: {response}\n")
    return str(response)

github_docs_tool = Tool(
    name="github_document_query",
    func=query_github_docs,
    description=f"Answers questions about the content within the '{docs_folder_path}' folder of the GitHub repository '{owner}/{repo}'. Use this for specific questions about the documentation found there.",
    # Ensure the description is clear for the agent
)

# --- 5. Langchain Agent Setup ---
# Use ChatOpenAI for Langchain agent
if mode == "local":
    llm_langchain = ChatOpenAI(
        model_name="LLM-Research/gemma-3-27b-it",
        openai_api_base=local_api_base, # Pass api_base if set
        temperature=0.2 # Agents often work better with low temperature
    )
else:
    llm_langchain = ChatOpenAI(
        model_name="gpt-4o",
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base, # Pass api_base if set
        temperature=0.2 # Agents often work better with low temperature
    )

# Initialize the agent
# Using AgentExecutor is the more modern approach compared to initialize_agent + run
# ZERO_SHOT_REACT_DESCRIPTION is suitable for using tool descriptions
agent_executor = initialize_agent( # initialize_agent returns an AgentExecutor
    tools=[github_docs_tool],
    llm=llm_langchain,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True, # See the agent's thought process
    # Pass the Langfuse handler in the callback manager configuration for the agent
    # Note: Langchain expects callbacks within agent_executor.run or .invoke
    # However, initialize_agent might accept callback_manager in some versions/setups.
    # It's more reliable to pass it during execution (.run or .invoke).
)


# --- 6. Execute the Agent with Langfuse Tracing ---
# query = "Use the 'github_document_query' tool to determine what language and format this documentation is primarily written in. Answer with exactly two words."
query = "使用 'github_document_query' 工具来确定这是一个使用_____编写的_____文档？请用2个词回答填空部分。" # Original query in Chinese

print(f"\nExecuting agent with query: \"{query}\"")

# Execute the agent using .invoke (preferred) or .run
# Pass the Langfuse handler via the 'callbacks' argument in the config dictionary
try:
    # Using invoke (more modern)
    response = agent_executor.invoke(
        {"input": query},
        config={"callbacks": [langfuse_langchain_handler]} # Pass handler here
    )
    agent_response = response.get("output", "No output field found.")

    # Or using run (older style, less structured output)
    # response = agent_executor.run(
    #    query,
    #    callbacks=[langfuse_langchain_handler] # Pass handler here
    # )
    # agent_response = response

    print(f"\nAgent's Final Answer: {agent_response}")

except Exception as e:
    print(f"Error during agent execution: {e}")
    # Log the error to Langfuse if possible/needed
    langfuse_langchain_handler.flush() # Ensure buffered data is sent before exiting on error


# --- 7. Shutdown Langfuse (Optional but Recommended) ---
# Ensures all buffered traces are sent before the script exits.
print("\nFlushing Langfuse handlers...")
langfuse_langchain_handler.flush()
# The LlamaIndex handler flushes automatically in many cases, but explicit flush is safe.
# Note: LlamaIndex's handler might not have an explicit flush method exposed in older versions.
# If using Settings.callback_manager, it should handle shutdown gracefully.
# If you created the LlamaIndex handler manually and passed it differently, you might need .shutdown() or .flush() if available.
print("Langfuse flushing complete.")