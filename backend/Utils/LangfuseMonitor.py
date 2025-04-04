import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any, Union

# Import Langfuse handlers
from langfuse.callback import CallbackHandler as LangfuseLangchainHandler
from langfuse.llama_index import LlamaIndexCallbackHandler as LangfuseLlamaIndexHandler

# Load environment variables if needed
load_dotenv()

class LangfuseMonitor:
    """
    管理 Langfuse 监控的类，为 LangChain 和 LlamaIndex 提供回调处理程序
    """
    
    _instance = None
    
    def __new__(cls):
        """
        使用单例模式确保只有一个 LangfuseMonitor 实例
        """
        if cls._instance is None:
            cls._instance = super(LangfuseMonitor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        初始化 Langfuse 监控器
        """
        if self._initialized:
            return
            
        # 从环境变量获取 Langfuse 凭据
        self.langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        self.langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        self.langfuse_host = os.environ.get("LANGFUSE_HOST")
        
        # 验证是否设置了必要的凭据
        if not self.langfuse_public_key or not self.langfuse_secret_key:
            print("警告: 未找到 Langfuse 公钥或密钥。跟踪功能可能无法正常工作。")
            self.enabled = False
        else:
            self.enabled = True
            
        # 初始化回调处理程序
        self._init_handlers()
        
        self._initialized = True
    
    def _init_handlers(self):
        """
        初始化 LangChain 和 LlamaIndex 的回调处理程序
        """
        if not self.enabled:
            self.langchain_handler = None
            self.llama_index_handler = None
            return
            
        # 为 LangChain 创建 Langfuse 处理程序
        self.langchain_handler = LangfuseLangchainHandler(
            public_key=self.langfuse_public_key,
            secret_key=self.langfuse_secret_key,
            host=self.langfuse_host
        )
        
        # 为 LlamaIndex 创建 Langfuse 处理程序
        self.llama_index_handler = LangfuseLlamaIndexHandler(
            public_key=self.langfuse_public_key,
            secret_key=self.langfuse_secret_key,
            host=self.langfuse_host
        )
    
    def get_langchain_handler(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Optional[LangfuseLangchainHandler]:
        """
        获取配置好的 LangChain 处理程序
        
        Args:
            user_id: 可选的用户 ID，用于在 Langfuse 中标识用户
            session_id: 可选的会话 ID，用于在 Langfuse 中标识会话
            
        Returns:
            配置好的 LangChain 回调处理程序，如果禁用则返回 None
        """
        if not self.enabled or not self.langchain_handler:
            return None
            
        # 如果提供了 user_id 或 session_id，创建一个新实例
        if user_id or session_id:
            return LangfuseLangchainHandler(
                public_key=self.langfuse_public_key,
                secret_key=self.langfuse_secret_key,
                host=self.langfuse_host,
                user_id=user_id,
                session_id=session_id
            )
            
        return self.langchain_handler
    
    def get_llama_index_handler(self) -> Optional[LangfuseLlamaIndexHandler]:
        """
        获取配置好的 LlamaIndex 处理程序
        
        Returns:
            配置好的 LlamaIndex 回调处理程序，如果禁用则返回 None
        """
        if not self.enabled:
            return None
            
        return self.llama_index_handler
    
    def get_langchain_callbacks(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[Any]:
        """
        获取用于 LangChain 操作的回调列表
        
        Args:
            user_id: 可选的用户 ID
            session_id: 可选的会话 ID
            
        Returns:
            包含回调处理程序的列表，如果禁用则为空列表
        """
        handler = self.get_langchain_handler(user_id, session_id)
        return [handler] if handler else []
    
    def flush(self):
        """
        确保所有缓冲的跟踪数据在程序退出前发送
        """
        if self.enabled and self.langchain_handler:
            self.langchain_handler.flush()
