import dotenv

dotenv.load_dotenv()
import nest_asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
nest_asyncio.apply()
from llama_index.core import VectorStoreIndex
from llama_index.readers.github import GithubRepositoryReader, GithubClient
import os
github_token = os.environ.get("GITHUB_TOKEN")
openai_api_base = os.environ.get("OPENAI_API_BASE")
openai_api_key = os.environ.get("OPENAI_API_KEY")
owner = "stepbystepcode"
repo = "CodeWay"
branch = "main"

github_client = GithubClient(github_token=github_token, verbose=True)
Settings.llm = OpenAI(model="gpt-4o", temperature=0.2, api_base=openai_api_base, api_key=openai_api_key)
documents = GithubRepositoryReader(
    github_client=github_client,
    owner=owner,
    repo=repo,
    use_parser=True,  # Set to True to parse Markdown content into Document objects
    verbose=False,
    filter_directories=(
        ["docs"],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
    filter_file_extensions=(
        [".md"],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
).load_data(branch=branch)
embed_model = OPEAEmbedding(
    model_name="...",
    api_base=openai_api_base,
    api_key=openai_api_key,
)
print(documents)
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = index.as_query_engine()
response = query_engine.query(
    "What is the difference between VectorStoreIndex and SummaryIndex?",
    verbose=True,
)
print(response)
# chat = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_base=openai_api_base)
# from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings

# vectorstore = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())
# # k is the number of chunks to retrieve
# retriever = vectorstore.as_retriever(k=4)

# docs = retriever.invoke("What is the difference between VectorStoreIndex and SummaryIndex?")
# print(docs)
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# SYSTEM_TEMPLATE = """
# Answer the user's questions based on the below context. 
# If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

# <context>
# {context}
# </context>
# """

# question_answering_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             SYSTEM_TEMPLATE,
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )

# document_chain = create_stuff_documents_chain(chat, question_answering_prompt)
# from langchain_core.messages import HumanMessage

# document_chain.invoke(
#     {
#         "context": docs,
#         "messages": [
#             HumanMessage(content="Can LangSmith help test my LLM applications?")
#         ],
#     }
# )
# document_chain.invoke(
#     {
#         "context": [],
#         "messages": [
#             HumanMessage(content="Can LangSmith help test my LLM applications?")
#         ],
#     }
# )
# from typing import Dict

# from langchain_core.runnables import RunnablePassthrough


# def parse_retriever_input(params: Dict):
#     return params["messages"][-1].content


# retrieval_chain = RunnablePassthrough.assign(
#     context=parse_retriever_input | retriever,
# ).assign(
#     answer=document_chain,
# )
# retrieval_chain.invoke(
#     {
#         "messages": [
#             HumanMessage(content="Can LangSmith help test my LLM applications?")
#         ],
#     }
# )