"""创建Agent实例"""
import Agently
import os
from github_reader import get_markdown_urls_from_github
agent_factory = (
    Agently.AgentFactory()
        # 给Agent工厂实例提供设置项：
        ## 将默认模型请求客户端设置为OAIClient（我们为OpenAI兼容格式定制的请求客户端）
        .set_settings("current_model", "OAIClient")
        ## 提供你的模型API-KEY
        # .set_settings("model.OAIClient.auth", { "api_key": "" })
        ## 指定你的模型Base-URL，如DeepSeek
        .set_settings("model.OAIClient.url", os.environ.get("OPENAI_API_BASE"))
        ## 指定你想要调用的具体模型
        .set_settings("model.OAIClient.options", { "model": "LLM-Research/gemma-3-27b-it" })
)
# 从Agent工厂实例中创建Agent实例
## 你可以创建很多个Agent实例，它们都会继承Agent工厂的设置项
## 也就是说，你不需要再为每一个Agent实例指定模型、输入授权
agent = agent_factory.create_agent()
"""定义并注册自定义工具"""
# 自定义工具函数及依赖

# 自定义工具信息字典
tool_info = {
    "tool_name": "get_markdown_urls_from_github",
    "desc": "get markdown urls from github",
    "args": {
        "github_url": (
            "str",
            "[*Required] GitHub URL in format https://github.com/{owner}/{repo}/tree/{branch}/{path}"
        )
    },
    "func": get_markdown_urls_from_github
}
# 向Agent实例注册自定义工具
agent.register_tool(
    tool_name = tool_info["tool_name"],
    desc = tool_info["desc"],
    args = tool_info["args"],
    func = tool_info["func"],
)
"""发起请求"""
# 正确的方式是给agent一个指令，让它使用工具，而不是直接写工具调用代码
print(agent.input("请获取GitHub仓库 https://github.com/jax-ml/jax/tree/main/docs 中的所有markdown文件的URL列表").start().result)