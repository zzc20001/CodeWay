import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_huggingface import  HuggingFaceEndpoint
openai_api_base = os.environ.get("OPENAI_API_BASE") # Optional: if using a proxy/custom base
openai_api_key = os.environ.get("OPENAI_API_KEY")
local_api_base = os.environ.get("LOCAL_API_BASE")

class ChatModelFactory:
    model_params = {
        "temperature": 0,
        "seed": 42,
    }

    @classmethod
    def get_model(cls, model_name):
        
        # if "gpt" in model_name:
        #     if not use_azure:
        #         return ChatOpenAI(model=model_name, **cls.model_params)
        #     else:
        #         return AzureChatOpenAI(
        #             azure_deployment=model_name,
        #             api_version="2024-05-01-preview",
        #             **cls.model_params
        #         )
        # elif model_name == "deepseek":
        #     # 换成开源模型试试
        #     # https://siliconflow.cn/
        #     # 一个 Model-as-a-Service 平台
        #     # 可以通过与 OpenAI API 兼容的方式调用各种开源语言模型。
        #     return ChatOpenAI(
        #         model="deepseek-ai/DeepSeek-V2-Chat",  # 模型名称
        #         openai_api_key=os.getenv("SILICONFLOW_API_KEY"),  # 在平台注册账号后获取
        #         openai_api_base="https://api.siliconflow.cn/v1",  # 平台 API 地址
        #         **cls.model_params,
        #     )‘
        if model_name == "gemma3":
            llm_langchain = ChatOpenAI(
            model_name="LLM-Research/gemma-3-27b-it",
            openai_api_base=local_api_base, # Pass api_base if set
            temperature=0.2 # Agents often work better with low temperature
        )
        elif model_name == "openai":
            llm_langchain = ChatOpenAI(
            model_name="gpt-4o",
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base, # Pass api_base if set
            temperature=0.2 # Agents often work better with low temperature
        )
        else :
            HuggingFaceEndpoint(
            repo_id="model_name",
            temperature=0.3,  # 降低随机性
            max_new_tokens=512,
            top_p=0.9,
            repetition_penalty=1.1
        )

        return llm_langchain

    @classmethod
    def get_default_model(cls):
        return cls.get_model(model="gemma3")


class EmbeddingModelFactory:

    @classmethod
    def get_model(cls, model_name: str, use_azure: bool = False):
        if model_name.startswith("text-embedding"):
            if not use_azure:
                return OpenAIEmbeddings(model=model_name)
            else:
                return AzureOpenAIEmbeddings(
                    azure_deployment=model_name,
                    openai_api_version="2024-05-01-preview",
                )
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")

    @classmethod
    def get_default_model(cls):
        return cls.get_model("text-embedding-ada-002")
