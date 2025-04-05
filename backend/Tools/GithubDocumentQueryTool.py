import os
import asyncio
import concurrent.futures
import hashlib
import json
from functools import partial
import pathlib
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
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

# 定义缓存目录
CACHE_DIR = backend_dir / "cache" / "github_indexes"
# 确保缓存目录存在
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 导入时间模块来跟踪性能
import time

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
    
    def get_or_create_qa_system(self, owner: str, repo: str, docs_folder_path: str, mode: str = "local", branch: str = "main", use_cache: bool = True) -> "GitHubDocsQA":
        """获取或创建指定仓库的QA系统实例"""
        key = f"{owner}/{repo}/{docs_folder_path}"
        
        if key not in self.qa_systems:
            print(f"[GitHub索引] 开始为 {key} 创建新的GitHub QA系统")
            print(f"[GitHub索引] 配置信息: 分支 = {branch}, 模式 = {mode}, 使用缓存 = {use_cache}")
            self.qa_systems[key] = GitHubDocsQA(
                owner=owner,
                repo=repo,
                branch=branch,
                docs_folder_path=docs_folder_path,
                mode=mode,
                use_cache=use_cache
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
        mode: str = "local",
        use_cache: bool = True
    ):
        """Initialize the GitHub documentation QA system."""
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.docs_folder_path = docs_folder_path
        self.mode = mode
        self.use_cache = use_cache
        
        # 默认初始化为None，将在加载方法中设置
        self.documents = None
        self.reader = None
        self.query_engine = None
        self.index = None
        self.cache_key = self._generate_cache_key()
        self.cache_dir = CACHE_DIR / self.cache_key
        
        # 如果启用缓存，尝试从缓存加载索引
        if use_cache and self._load_from_cache():
            print(f"[GitHub索引] 已从本地缓存加载索引: {self.cache_key}")
        else:
            if not use_cache:
                print(f"[GitHub索引] 缓存已禁用，将重新创建索引")
            # 使用线程异步加载文档
            future = _executor.submit(self._load_documents_sync)
            self.documents = future.result()  # 立即获取结果
            
            # 如果文档加载成功，创建索引
            if self.documents:
                try:
                    future = _executor.submit(self._create_index_sync, self.documents)
                    self.index = future.result()
                    if self.index:
                        # 保存索引到缓存（如枟启用缓存）
                        if use_cache:
                            self._save_to_cache()
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
    
    def _generate_cache_key(self):
        """生成基于仓库信息的缓存键"""
        # 组合所有相关字段
        key_string = f"{self.owner}_{self.repo}_{self.branch}_{self.docs_folder_path}"
        # 使用哈希函数创建一个唯一标识符
        hash_obj = hashlib.md5(key_string.encode())
        # 返回一个对人类友好的标识符
        return f"{self.owner}_{self.repo}_{hash_obj.hexdigest()[:8]}"

    def _save_to_cache(self):
        """将索引保存到本地缓存"""
        try:
            if self.index:
                # 确保缓存目录存在
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存索引
                self.index.storage_context.persist(persist_dir=str(self.cache_dir))
                
                # 获取当前嵌入模型信息
                embedding_model_info = self._get_embedding_model_info()
                
                # 保存元数据
                metadata = {
                    "owner": self.owner,
                    "repo": self.repo,
                    "branch": self.branch,
                    "docs_folder_path": self.docs_folder_path,
                    "mode": self.mode,
                    "timestamp": self._get_current_timestamp(),
                    "embedding_model": embedding_model_info
                }
                
                with open(self.cache_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
                print(f"[GitHub索引] 已成功缓存索引: {self.cache_key}")
                print(f"[GitHub索引] 使用嵌入模型: {embedding_model_info.get('name', '未知')}, 维度: {embedding_model_info.get('dimensions', '未知')}")
                return True
            return False
        except Exception as e:
            print(f"[GitHub索引] 缓存索引时出错: {e}")
            return False
            
    def _get_embedding_model_info(self):
        """获取当前使用的嵌入模型信息"""
        try:
            # 默认模型信息
            model_info = {
                "name": "BAAI/bge-m3",
                "dimensions": 1024  # 默认维度
            }
            
            # 如果可以从索引中获取嵌入模型信息
            if self.index:
                # 检查是否可以访问嵌入模型
                if hasattr(self.index, "services") and "vector_store" in self.index.services:
                    vector_store = self.index.services["vector_store"]
                    if hasattr(vector_store, "_embedding_dimensionality"):
                        model_info["dimensions"] = vector_store._embedding_dimensionality
                    elif hasattr(vector_store, "client") and hasattr(vector_store.client, "embedding_dimension"):
                        model_info["dimensions"] = vector_store.client.embedding_dimension
            
            return model_info
        except Exception as e:
            print(f"[GitHub索引] 获取嵌入模型信息失败: {e}")
            return {"name": "unknown", "dimensions": 0}
        
    def _get_current_timestamp(self):
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _load_from_cache(self):
        """从本地缓存加载索引"""
        try:
            # 检查缓存目录和元数据文件是否存在
            metadata_path = self.cache_dir / "metadata.json"
            if not self.cache_dir.exists() or not metadata_path.exists():
                print(f"[GitHub索引] 没有找到缓存: {self.cache_key}")
                return False
                
            # 加载元数据
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                
            # 显示缓存信息
            print(f"[GitHub索引] 找到缓存的索引: {self.cache_key}")
            print(f"[GitHub索引] 索引时间: {metadata.get('timestamp', '未知')}")
            
            # 检查嵌入模型兼容性
            cached_model = metadata.get('embedding_model', {})
            current_model = self._get_embedding_model_info()
            
            if cached_model and current_model:
                cached_dim = cached_model.get('dimensions', 0)
                current_dim = current_model.get('dimensions', 0)
                
                # 输出调试信息
                print(f"[GitHub索引] 缓存使用的嵌入模型: {cached_model.get('name')}, 维度: {cached_dim}")
                print(f"[GitHub索引] 当前使用的嵌入模型: {current_model.get('name')}, 维度: {current_dim}")
                
                # 如果维度信息可用且不匹配，不使用缓存
                if cached_dim != 0 and current_dim != 0 and cached_dim != current_dim:
                    print(f"[GitHub索引] 缓存的嵌入模型维度({cached_dim})与当前模型维度({current_dim})不匹配")
                    print(f"[GitHub索引] 删除不兼容的缓存并重新创建索引")
                    
                    # 删除不兼容的缓存目录
                    import shutil
                    if self.cache_dir.exists():
                        shutil.rmtree(self.cache_dir)
                    return False
            
            # 使用存储上下文加载索引
            try:
                storage_context = StorageContext.from_defaults(persist_dir=str(self.cache_dir))
                self.index = load_index_from_storage(storage_context)
                
                # 创建查询引擎
                if self.index:
                    # 设置全局嵌入模型，确保使用与创建索引时相同的模型
                    if cached_model and 'name' in cached_model:
                        try:
                            # 强制使用与缓存相同的嵌入模型类型和名称
                            model_name = cached_model.get('name', "BAAI/bge-m3")
                            if self.mode == "local":
                                # 创建与缓存相同的嵌入模型实例
                                embedding_model = HuggingFaceBgeEmbeddings(model_name=model_name)
                                
                                # 将模型设置到全局设置和索引中
                                Settings.embed_model = embedding_model
                                
                                # 尝试直接设置索引的嵌入模型
                                if hasattr(self.index, 'set_embed_model'):
                                    self.index.set_embed_model(embedding_model)
                                    
                                print(f"[GitHub索引] 已设置全局嵌入模型和索引嵌入模型为: {model_name}")
                        except Exception as e:
                            print(f"[GitHub索引] 设置嵌入模型失败: {e}")
                    
                    self.query_engine = self.index.as_query_engine()
                    # 设置documents为非空列表，因为从缓存加载时我们没有原始文档
                    self.documents = ["cached"]
                    print(f"[GitHub索引] 成功从缓存加载索引")
                    return True
                return False
            except Exception as e:
                print(f"[GitHub索引] 加载索引失败，可能是模型不兼容: {e}")
                return False
        except Exception as e:
            print(f"[GitHub索引] 从缓存加载索引时出错: {e}")
            return False

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
                    self.start_time = time.time()
                
                def set_phase(self, phase):
                    self.phase = phase
                    print(f"[GitHub索引] 阶段: {phase}")
                
                def update(self, increment=1):
                    self.processed_docs += increment
                    percent = int((self.processed_docs / self.total_docs) * 100)
                    # 只在百分比变化时打印，避免过多输出
                    if percent > self.last_percent:
                        elapsed = time.time() - self.start_time
                        print(f"[GitHub索引] {self.phase} - 进度: {percent}% ({self.processed_docs}/{self.total_docs}), 用时: {elapsed:.2f}秒")
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
                
            # 设置文本分割器 - 使用更大的块大小减少总块数(性能优化)
            progress.set_phase("配置文本分割器")
            # 使用更大的分块大小，减少分块总数
            splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=100)
            
            # 设置嵌入模型 - 使用更高效的模型(性能优化)
            progress.set_phase("加载嵌入模型")
            embedding_model = None
            
            if self.mode == "local":
                # 性能优化: 使用更快速的嵌入模型
                # 您可以在此处选择使用原型模型或轻量级模型
                # 性能优先: "all-MiniLM-L6-v2" (零品质罪牌)
                # 质量优先: "BAAI/bge-m3" (原始模型)
                model_name = "all-MiniLM-L6-v2"  # 轻量级、更快速的模型
                print(f"[GitHub索引] 使用高效嵌入模型: {model_name}")
                embedding_model = HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},  # 显式指定设备
                    encode_kwargs={"batch_size": 32}  # 使用批处理加速嵌入
                )
                
                # 如果存在langchain回调，设置跟踪
                if langchain_handler:
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
            
            # 优化文档批处理
            progress.set_phase("优化文档处理")
            # 使用更大的批处理大小处理文档，减少嵌入调用次数
            batch_size = 10
            # 将文档分批处理
            document_batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
            print(f"[GitHub索引] 将{len(documents)}个文档分成{len(document_batches)}批处理。每批{batch_size}个文档")
            
            # 执行多进程嵌入处理
            progress.set_phase("构建文档索引")
            print(f"[GitHub索引] 开始批量处理文档并创建嵌入...")
            
            # 记录开始时间优化
            start_time = time.time()
            
            # 使用回调管理器创建索引
            # 性能优化：这里使用默认分块器，不再单独指定
            try:
                # 使用回调管理器创建索引
                if callback_manager:
                    index = VectorStoreIndex.from_documents(
                        documents, 
                        transformations=[splitter],  # 使用优化的分割器
                        callback_manager=callback_manager,
                        show_progress=True  # 显示进度条
                    )
                else:
                    # 不使用Langfuse，使用默认构建索引
                    index = VectorStoreIndex.from_documents(
                        documents, 
                        transformations=[splitter],
                        show_progress=True  # 显示进度条
                    )
                
                # 记录完成时间和性能指标
                end_time = time.time()
                total_time = end_time - start_time
                docs_per_second = len(documents) / total_time if total_time > 0 else 0
                
                print(f"[GitHub索引] 性能指标: 总用时 {total_time:.2f}秒, 平均 {docs_per_second:.2f} 文档/秒")
            
            except Exception as e:
                print(f"[GitHub索引] 索引创建错误: {e}")
                raise e
                
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
            # 在查询前确保所有层面都使用正确的嵌入模型
            embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")
            Settings.embed_model = embed_model
            print(f"[GitHub索引] 查询前强制设置统一嵌入模型: BAAI/bge-m3")
            
            # 尝试访问查询引擎内部并设置嵌入模型
            if hasattr(self.query_engine, '_service_context'):
                if hasattr(self.query_engine._service_context, 'embed_model'):
                    self.query_engine._service_context.embed_model = embed_model
                    print(f"[GitHub索引] 直接设置查询引擎的嵌入模型")
            
            # 如果自定义嵌入模型不起作用，尝试使用简单的内容查询
            try:
                # 执行原始查询器
                response = self.query_engine.query(query)
                return str(response)
            except Exception as e:
                # 如果因维度问题失败，我们将使用更直接的方式
                if "shapes" in str(e) and "not aligned" in str(e):
                    print(f"[GitHub索引] 使用原始查询失败，尝试直接搜索索引节点...")
                    # 如果可能，直接搜索索引节点
                    if hasattr(self.index, 'as_retriever'):
                        try:
                            retriever = self.index.as_retriever()
                            nodes = retriever.retrieve(query)
                            result = "\n\n".join([node.node.text for node in nodes])
                            return f"搜索结果:\n{result}"
                        except Exception as inner_e:
                            print(f"[GitHub索引] 直接搜索失败: {inner_e}")
                    
                    # 返回错误信息
                    return f"维度不匹配错误: {str(e)}\n\n建议重启服务或删除缓存后重试"
                raise e
        except Exception as e:
            print(f"[GitHub索引] 查询失败: {str(e)}")
            # 记录更多调试信息
            if "shapes" in str(e) and "not aligned" in str(e):
                print(f"[GitHub索引] 维度不匹配错误，请确保查询与索引使用相同的嵌入模型")
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
    mode: str = "local",
    use_cache: bool = True
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
            use_cache=use_cache,
        )
    )

def ask_github_docs(
    query: str,
    owner: str = "stepbystepcode",
    repo: str = "CodeWay",
    branch: str = "main",
    docs_folder_path: str = "docs",
    mode: str = "local",
    use_cache: bool = True,
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
            mode=mode,
            use_cache=use_cache
        )
        
        # 如果QA系统没有成功初始化，返回错误
        if qa_system.index is None:
            return f"抱歉，无法为仓库 {owner}/{repo} 中的 {docs_folder_path} 创建文档索引。请确保仓库路径正确且有权限访问。"
        
        return qa_system.query_docs(query)
    except Exception as e:
        print(f"GitHub文档查询失败: {e}")
        return f"处理查询时出错: {str(e)}"