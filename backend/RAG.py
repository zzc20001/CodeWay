
import bs4
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_huggingface import  HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv(dotenv_path=".env")  # 自动从 .env 文件读取环境变量
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03
)

# llm = init_chat_model("gpt-4o-mini", model_provider="openai")
# 加载函数，接收前端传入的参数
def load_documents(web_paths: List[str] = [], file_paths: List[str] = [], embeddings=None):
    """
    加载网页或本地文件，并进行拆分，返回拆分后的文档集合。
    
    :param web_paths: 网页的 URL 列表
    :param file_paths: 本地文件路径的列表
    :param embeddings: 用于文档向量化的嵌入模型
    :return: 拆分后的文档列表
    """
    all_docs = []

    # 处理网页加载
    if web_paths:
        for url in web_paths:
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
                ),
            )
            all_docs.extend(loader.load())

    # 处理本地文件加载
    if file_paths:
        for file_path in file_paths:
            file_extension = file_path.split('.')[-1].lower()
            if file_extension == 'txt':
                loader = TextLoader(file_path)
            elif file_extension == 'pdf':
                loader = PyPDFLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            all_docs.extend(loader.load())

    # 文本拆分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(all_docs)    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # 将文档添加到向量库
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(all_splits)

    return vector_store


# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# 将文档添加到向量库
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)
# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response} 

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])
