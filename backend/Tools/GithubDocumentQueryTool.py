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

# GitHub QA系统管理器，存储不同仓库的QA系统实例
class GitHubQAManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GitHubQAManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # 存储QA系统实例，key为 "owner/repo/docs_folder_path"
        self.qa_systems = {}
        self._initialized = True
    
    def get_or_create_qa_system(self, owner: str, repo: str, docs_folder_path: str, mode: str = "local", branch: str = "main") -> "GitHubDocsQA":
        """获取或创建指定仓库的QA系统实例"""
        key = f"{owner}/{repo}/{docs_folder_path}"
        
        if key not in self.qa_systems:
            print(f"[GitHub索引] 开始为 {key} 创建新的GitHub QA系统")
            print(f"[GitHub索引] 配置信息: 分支 = {branch}, 模式 = {mode}")
            self.qa_systems[key] = GitHubDocsQA(
                owner=owner,
                repo=repo,
                branch=branch,
                docs_folder_path=docs_folder_path,
                mode=mode
            )
            print(f"[GitHub索引] GitHub QA系统创建完成: {key}")
        else:
            print(f"[GitHub索引] 使用已存在的GitHub QA系统: {key}")
        
        return self.qa_systems[key]
    
    def list_repositories(self):
        """列出所有已加载的仓库"""
        return list(self.qa_systems.keys())

# 创建全局实例
github_qa_manager = GitHubQAManager()

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
                
            print(f"[GitHub索引] 正在初始化 GitHub 客户端...")
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
            
            print(f"[GitHub索引] 开始从 {self.owner}/{self.repo}/{self.docs_folder_path} 加载文档...")
            print(f"[GitHub索引] 正在克隆或更新仓库...")
            documents = self.reader.load_data(branch=self.branch)
            if not documents:
                print(f"[GitHub索引] 警告: 在 {self.owner}/{self.repo}/{self.docs_folder_path} 中未找到文档")
            else:
                print(f"[GitHub索引] 成功加载了 {len(documents)} 个文档")
                # 显示文档类型的统计信息
                extensions = {}
                for doc in documents:
                    # 从文档元数据中提取文件扩展名
                    if hasattr(doc, 'metadata') and 'file_path' in doc.metadata:
                        ext = os.path.splitext(doc.metadata['file_path'])[1]
                        extensions[ext] = extensions.get(ext, 0) + 1
                    
                # 打印文档类型统计
                if extensions:
                    ext_info = ", ".join([f"{ext}: {count}" for ext, count in extensions.items()])
                    print(f"[GitHub索引] 文档类型统计: {ext_info}")
            return documents
        except Exception as e:
            print(f"Error loading documents: {e}")
            return []
    
    def _create_index_sync(self, documents):
        """Synchronously create an index from documents."""
        try:
            if not documents:
                print("[GitHub索引] 警告: 没有提供文档来创建索引")
                return None
            
            # 创建自定义进度回调类
            class IndexProgressCallback:
                def __init__(self, total_docs):
                    self.total_docs = total_docs
                    self.processed_docs = 0
                    self.last_percent = -1
                    self.phase = "准备中"
                
                def set_phase(self, phase):
                    self.phase = phase
                    print(f"[GitHub索引] 阶段: {phase}")
                
                def update(self, increment=1):
                    self.processed_docs += increment
                    percent = int((self.processed_docs / self.total_docs) * 100)
                    # 只在百分比变化时打印，避免过多输出
                    if percent > self.last_percent:
                        print(f"[GitHub索引] {self.phase} - 进度: {percent}% ({self.processed_docs}/{self.total_docs})")
                        self.last_percent = percent
            
            # 初始化进度跟踪器
            progress = IndexProgressCallback(len(documents))
            progress.set_phase("初始化索引")
            
            print(f"[GitHub索引] 开始创建向量索引...共 {len(documents)} 个文档")
            
            # 配置LlamaIndex的Langfuse集成
            progress.set_phase("配置监控与回调")
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
            progress.set_phase("配置文本分割器")
            splitter = SentenceSplitter(chunk_size=1024)
            
            # 设置嵌入模型 - 添加langfuse监控
            progress.set_phase("加载嵌入模型")
            embedding_model = None
            if self.mode == "local":
                print(f"[GitHub索引] 使用本地嵌入模型: BAAI/bge-m3")
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
            progress.set_phase("构建文档索引")
            print(f"[GitHub索引] 开始处理文档并创建嵌入，这可能需要一些时间...")
            
            # 创建或处理每个文档时的回调
            class DocumentProcessingCallback:
                def __init__(self, progress_tracker):
                    self.progress = progress_tracker
                    self.doc_count = 0
                
                def on_document_processed(self, **kwargs):
                    self.doc_count += 1
                    self.progress.update()
                    
            doc_callback = DocumentProcessingCallback(progress)
            
            # 包装VectorStoreIndex.from_documents方法以捕获处理进度
            original_from_documents = VectorStoreIndex.from_documents
            
            try:
                # 创建包装函数，在处理每个文档后调用进度更新
                def wrapped_from_documents(documents, *args, **kwargs):
                    # 这里可以添加文档处理前的逻辑
                    index = original_from_documents(documents, *args, **kwargs)
                    # 处理完成后更新进度
                    return index
                
                # 暂时替换原始方法
                VectorStoreIndex.from_documents = wrapped_from_documents
                
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
            finally:
                # 恢复原始方法
                VectorStoreIndex.from_documents = original_from_documents
                
            progress.set_phase("完成")
            print(f"[GitHub索引] 索引创建完成。已处理 {len(documents)} 个文档。")
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
        # 使用管理器获取或创建QA系统
        qa_system = github_qa_manager.get_or_create_qa_system(
            owner=owner,
            repo=repo,
            branch=branch,
            docs_folder_path=docs_folder_path,
            mode=mode
        )
        
        # 如果QA系统没有成功初始化，返回错误
        if qa_system.index is None:
            return f"抱歉，无法为仓库 {owner}/{repo} 中的 {docs_folder_path} 创建文档索引。请确保仓库路径正确且有权限访问。"
        
        return qa_system.query_docs(query)
    except Exception as e:
        print(f"GitHub文档查询失败: {e}")
        return f"处理查询时出错: {str(e)}"