import os
import asyncio
import concurrent.futures
from functools import partial
import pathlib
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.callbacks import CallbackManager
from langchain.tools import Tool
from llama_index.core.node_parser import SentenceSplitter
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings

# 确保加载正确的环境变量文件
# 计算当前文件的目录
current_dir = pathlib.Path(__file__).parent.absolute()
# 计算后端根目录的路径(假设结构是backend/Tools/当前文件)
backend_dir = current_dir.parent
# 加载根目录中的.env文件
dotenv_path = backend_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# 打印环境变量状态(仅调试用)
print(f"[GithubDocumentQueryTool] 环境变量状态:")
print(f"  GITHUB_TOKEN: {'已设置' if os.environ.get('GITHUB_TOKEN') else '未设置'}")

# 线程池执行器，用于运行可能阻塞的操作
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

class GitHubDocsQA:
    def __init__(
        self,
        owner: str = "stepbystepcode",
        repo: str = "CodeWay",
        branch: str = "main",
        docs_folder_path: str = "docs",
        mode: str = "local"
    ):
        """Initialize the GitHub documentation QA system."""
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.docs_folder_path = docs_folder_path
        self.mode = mode
        
        # 默认初始化为None，将在加载方法中设置
        self.documents = None
        self.reader = None
        self.query_engine = None
        self.index = None
        
        # 使用线程异步加载文档
        future = _executor.submit(self._load_documents_sync)
        self.documents = future.result()  # 立即获取结果
        
        # 如果文档加载成功，创建索引
        if self.documents:
            try:
                future = _executor.submit(self._create_index_sync, self.documents)
                self.index = future.result()
                if self.index:
                    self.query_engine = self.index.as_query_engine()
                else:
                    self.query_engine = None
            except Exception as e:
                print(f"创建索引时出错: {e}")
                self.query_engine = None
    
    def _load_documents_sync(self):
        """Synchronously load documents from GitHub repository."""
        try:
            # 获取GitHub token并验证
            github_token = os.environ.get("GITHUB_TOKEN")
            if not github_token:
                print("错误: 无法获取GITHUB_TOKEN环境变量。请确保.env文件已正确配置。")
                return []
                
            github_client = GithubClient(github_token=github_token, verbose=True)
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
            documents = self.reader.load_data(branch=self.branch)
            if not documents:
                print(f"Warning: No documents found in {self.owner}/{self.repo}/{self.docs_folder_path}")
            return documents
        except Exception as e:
            print(f"Error loading documents: {e}")
            return []
    
    def _create_index_sync(self, documents):
        """Synchronously create an index from documents."""
        try:
            if not documents:
                print("Warning: No documents provided to create index")
                return None
            
            print("Creating vector store index...")
            
            # 配置LlamaIndex的Langfuse集成
            from Utils.LangfuseMonitor import LangfuseMonitor
            monitor = LangfuseMonitor()
            llama_index_handler = monitor.get_llama_index_handler()
            langchain_handler = monitor.get_langchain_handler()
            
            # 创建回调管理器
            callbacks = []
            if llama_index_handler:
                callbacks.append(llama_index_handler)
                callback_manager = CallbackManager(callbacks)
            else:
                callback_manager = None
                
            # 设置文本分割器 - 追踪分块过程
            splitter = SentenceSplitter(chunk_size=1024)
            
            # 设置嵌入模型 - 添加langfuse监控
            embedding_model = None
            if self.mode == "local":
                # 使用模型工厂创建嵌入模型
                embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")
                # 如果存在langchain回调，设置跟踪
                if langchain_handler:
                    # 如果模型有callbacks参数，设置langfuse监控
                    if hasattr(embedding_model, 'callbacks') and embedding_model.callbacks is not None:
                        embedding_model.callbacks.append(langchain_handler)
                    elif hasattr(embedding_model, 'client') and hasattr(embedding_model.client, 'callbacks'):
                        if embedding_model.client.callbacks is None:
                            embedding_model.client.callbacks = [langchain_handler]
                        else:
                            embedding_model.client.callbacks.append(langchain_handler)
            
                # 配置Settings
                if embedding_model:
                    Settings.embed_model = embedding_model
            
            # 使用回调管理器创建索引
            if callback_manager:
                index = VectorStoreIndex.from_documents(
                    documents, 
                    transformations=[splitter],
                    callback_manager=callback_manager
                )
            else:
                # 不使用Langfuse时的创建方式
                index = VectorStoreIndex.from_documents(
                    documents, 
                    transformations=[splitter]
                )
                
            print("Index created.")
            return index
        except Exception as e:
            print(f"Error creating index: {e}")
            return None
    
    def query_docs(self, query: str) -> str:
        """
        Query the GitHub documentation and return the response.
        
        Args:
            query: The question to ask about the documentation
            
        Returns:
            str: The answer to the question
        """
        if not self.documents:
            return "抱歉，文档内容为空或无法加载"
           
        if self.query_engine is None:
            return "抱歉，查询引擎未初始化，可能是由于运行环境限制"
        
        try:
            # 在线程池中执行查询，避免阻塞事件循环
            future = _executor.submit(self._query_sync, query)
            response = future.result(timeout=30)  # 设置30秒超时
            return response
        except concurrent.futures.TimeoutError:
            return "抱歉，查询超时，请稍后再试"
        except Exception as e:
            print(f"查询错误: {e}")
            return f"查询错误: {str(e)}"
            
    def _query_sync(self, query: str) -> str:
        """执行同步查询，在线程池中调用"""
        try:
            response = self.query_engine.query(query)
            return str(response)
        except Exception as e:
            return f"查询执行错误: {str(e)}"
    
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

async def ask_github_docs_async(
    query: str,
    owner: str = "stepbystepcode",
    repo: str = "CodeWay",
    branch: str = "main",
    docs_folder_path: str = "docs",
    mode: str = "local"
) -> str:
    """Asynchronous version of ask_github_docs"""
    # 将函数包装在executor中运行，避免阻塞事件循环
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        partial(
            ask_github_docs,
            query=query,
            owner=owner,
            repo=repo,
            branch=branch,
            docs_folder_path=docs_folder_path,
            mode=mode,
        )
    )

def ask_github_docs(
    query: str,
    owner: str = "stepbystepcode",
    repo: str = "CodeWay",
    branch: str = "main",
    docs_folder_path: str = "docs",
    mode: str = "local",
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
        
    Returns:
        str: The answer to the query
    """
    # 直接从环境变量获取凭证
    github_token = os.environ.get("GITHUB_TOKEN")
    
    if not github_token:
        return "抱歉，GitHub令牌未设置。请确保环境变量GITHUB_TOKEN已正确设置。"
    
    try:
        # 创建QA系统并执行查询
        qa_system = GitHubDocsQA(
            owner=owner,
            repo=repo,
            branch=branch,
            docs_folder_path=docs_folder_path,
            mode=mode,
        )
        
        return qa_system.query_docs(query)
    except Exception as e:
        print(f"GitHub文档查询失败: {e}")
        return f"处理查询时出错: {str(e)}"