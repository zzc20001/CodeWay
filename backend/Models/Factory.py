import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
openai_api_base = os.environ.get("OPENAI_API_BASE") # Optional: if using a proxy/custom base
openai_api_key = os.environ.get("OPENAI_API_KEY")

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
                openai_api_base=openai_api_base,  # Pass api_base if set
                **params
            )
        elif model_name == "openai":
            return ChatOpenAI(
                model_name="gpt-4o",
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,  # Pass api_base if set
                **params
            )

    @classmethod
    def get_default_model(cls):
        return cls.get_model(model_name="gemma3")


class EmbeddingModelFactory:

    @classmethod
    def get_model(cls, model_name: str):
        if model_name.startswith("text-embedding"):
            return OpenAIEmbeddings(model=model_name)
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")