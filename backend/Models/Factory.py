import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
        # Default parameters for the model
        params = {
            "temperature": 0.2  # Agents often work better with low temperature
        }
        
        # Handle different model types
        if model_name == "gemma3":
            return ChatOpenAI(
                model_name="LLM-Research/gemma-3-27b-it",
                openai_api_base=local_api_base,  # Pass api_base if set
                **params
            )
        elif model_name == "openai":
            return ChatOpenAI(
                model_name="gpt-4o",
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,  # Pass api_base if set
                **params
            )
        elif "gpt" in model_name.lower():  # Handle any GPT model including gpt-4o
            return ChatOpenAI(
                model_name=model_name,
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,  # Pass api_base if set
                **params
            )
        else:  # Default to HuggingFace for any other model
            return HuggingFaceEndpoint(
                repo_id=model_name,
                temperature=0.3,  # 降低随机性
                max_new_tokens=512,
                top_p=0.9,
                repetition_penalty=1.1
            )

    @classmethod
    def get_default_model(cls):
        return cls.get_model(model_name="openai")


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
