import asyncio
import shutil
import os
import pathlib
import time

# 先用延迟导入，这样可以在其前设置模型
# 这样能确保在其他模块导入前先设置好全局模型
from Tools.GithubDocumentQueryTool import ask_github_docs_async

async def test_jax_documentation():
    """测试JAX文档查询功能"""
    # 首先确保使用正确的嵌入模型
    # 这一步很重要，必须在其他导入之前设置模型
    from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
    from llama_index.core import Settings
    
    # 强制指定相同的嵌入模型，确保测试和缓存使用相同的模型
    print("[测试脚本] 全局设置BAAI/bge-m3嵌入模型")
    Settings.embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")
    
    # 检查缓存目录状态
    repo_owner = "stepbystepcode"
    repo_name = "CodeWay"
    
    # 显示缓存信息
    backend_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = backend_dir / "cache" / "github_indexes"
    
    # 检查是否存在缓存
    cache_exists = False
    if cache_dir.exists():
        for item in cache_dir.iterdir():
            if item.is_dir() and item.name.startswith(f"{repo_owner}_{repo_name}_"):
                print(f"发现现有缓存: {item}")
                cache_exists = True
    
    # 一些有关JAX的问题
    questions = [
        "JAX 是什么？它有哪些主要特性？",
        "JAX 中的 jit 装饰器有什么用途？",
        "如何在JAX中使用随机数？",
        "JAX 与 TensorFlow 和 PyTorch 相比有什么优势？",
        "JAX 中的 vmap 函数的作用是什么？"
        # "a的值是多少？b的值是多少？"
    ]
    
    print("=" * 80)
    print("JAX文档查询测试")
    print("=" * 80)
    
    for i, question in enumerate(questions):
        print(f"\n问题 {i+1}: {question}")
        print("-" * 80)
        
        # 调用GitHub文档查询工具，指定CodeWay仓库
        # 如果发现缓存，则使用缓存（use_cache=True）
        response = await ask_github_docs_async(
            query=question,
            owner="jax-ml",
            repo="jax",
            docs_folder_path="docs",
            branch="main",
            use_cache=True
        )
        
        print(f"回答:\n{response}")
        print("=" * 80)
        
        # 短暂休息，避免过快发送请求
        await asyncio.sleep(1)

if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_jax_documentation())
