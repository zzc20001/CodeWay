from typing import List, Optional
import os
import nest_asyncio
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.core import Settings, VectorStoreIndex
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.callbacks import CallbackManager
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.schema import Document
from Models.Factory import ChatModelFactory, EmbeddingModelFactory
from llama_index.core.node_parser import SentenceSplitter
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
# Load environment variables
load_dotenv()

nest_asyncio.apply()

class GitHubDocsQA:
    def __init__(
        self,
        owner: str = "stepbystepcode",
        repo: str = "CodeWay",
        branch: str = "main",
        docs_folder_path: str = "docs",
        mode: str = "local",
        local_api_base: Optional[str] = None,
        openai_api_base: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        github_token: Optional[str] = None
    ):
        """Initialize the GitHub documentation QA system."""
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.docs_folder_path = docs_folder_path
        self.mode = mode
        
        # Set API keys and bases
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.openai_api_base = openai_api_base or os.environ.get("OPENAI_API_BASE")
        self.local_api_base = local_api_base or os.environ.get("LOCAL_API_BASE")
        
        if not self.github_token:
            raise ValueError("GitHub token is required")
        if not self.openai_api_key and self.mode != "local":
            raise ValueError("OpenAI API key is required for non-local mode")
            
        # self._setup_llama_index()
        self._load_documents()
        self._create_index()
        
    # def _setup_llama_index(self):
        """Configure LlamaIndex settings."""
        # if self.mode == "local":
        #     Settings.llm = LlamaIndexOpenAI(
        #         model="LLM-Research/gemma-3-27b-it",
        #         temperature=0.2,
        #         api_base=self.local_api_base,
        #         api_key="no-key-required"  # Some local models don't need keys
        #     )
        # else:
        #     Settings.llm = LlamaIndexOpenAI(
        #         model="gpt-4o",
        #         temperature=0.2,
        #         api_base=self.openai_api_base,
        #         api_key=self.openai_api_key
        #     )
    
    def _load_documents(self):
        """Load documents from GitHub repository."""
        github_client = GithubClient(github_token=self.github_token, verbose=True)
        self.reader = GithubRepositoryReader(
            github_client=github_client,
            owner=self.owner,
            repo=self.repo,
            use_parser=True,
            verbose=False,
            filter_directories=(
                [self.docs_folder_path],
                GithubRepositoryReader.FilterType.INCLUDE,
            ),
            filter_file_extensions=(
                [".md"],
                GithubRepositoryReader.FilterType.INCLUDE,
            ),
        )
        
        print(f"Loading documents from {self.owner}/{self.repo}/{self.docs_folder_path}...")
        self.documents = self.reader.load_data(branch=self.branch)
        if not self.documents:
            print(f"Warning: No documents found in {self.owner}/{self.repo}/{self.docs_folder_path}")
    
    def _create_index(self):
        """Create vector store index from documents."""
        print("Creating vector store index...")
        splitter = SentenceSplitter(chunk_size=1024)
        embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")
        self.index = VectorStoreIndex.from_documents(self.documents,transformations=[splitter], embed_model=embed_model)
        self.query_engine = self.index.as_query_engine()
        print("Index created.")
    
    def query_docs(self, query: str) -> str:
        """
        Query the GitHub documentation and return the response.
        
        Args:
            query: The question to ask about the documentation
            
        Returns:
            str: The answer to the question
        """
        if not self.documents:
            return "抱歉，文档内容为空"
            
        response = self.query_engine.query(query)
        return str(response)
    
    def get_tool(self) -> Tool:
        """
        Get a LangChain Tool instance for this QA system.
        
        Returns:
            Tool: Configured LangChain tool
        """
        return Tool(
            name="github_document_query",
            func=self.query_docs,
            description=f"Answers questions about the content within the '{self.docs_folder_path}' folder of the GitHub repository '{self.owner}/{self.repo}'. Use this for specific questions about the documentation found there."
        )

def ask_github_docs(
    query: str,
    owner: str = "stepbystepcode",
    repo: str = "CodeWay",
    branch: str = "main",
    docs_folder_path: str = "docs",
    mode: str = "local",
    local_api_base: Optional[str] = None,
    openai_api_base: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    github_token: Optional[str] = None
) -> str:
    """
    Query documentation in a GitHub repository and return the answer.
    
    Args:
        query: The question to ask
        owner: GitHub repository owner
        repo: GitHub repository name
        branch: Branch to use
        docs_folder_path: Path to docs folder in repository
        mode: "local" or other for model selection
        local_api_base: Base URL for local model API
        openai_api_base: Base URL for OpenAI API
        openai_api_key: OpenAI API key
        github_token: GitHub access token
        
    Returns:
        str: The answer to the query
    """
    qa_system = GitHubDocsQA(
        owner=owner,
        repo=repo,
        branch=branch,
        docs_folder_path=docs_folder_path,
        mode=mode,
        local_api_base=local_api_base,
        openai_api_base=openai_api_base,
        openai_api_key=openai_api_key,
        github_token=github_token
    )
    
    return qa_system.query_docs(query)

if __name__ == "__main__":
    # Example usage
    query = "文档主要使用什么语言和格式编写的？请用两个词回答。"
    response = ask_github_docs(
        query=query,
        github_token=os.environ.get("GITHUB_TOKEN"),
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    print(response)
