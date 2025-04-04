import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from supabase import create_client, Client, ClientOptions
from dotenv import load_dotenv
import httpx
from typing import Optional, List

# LangChain imports
from Agent.ReAct import ReActAgent
from Models.Factory import ChatModelFactory
from Tools import *
from Tools.PythonTool import ExcelAnalyser
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory

# Load environment variables
load_dotenv()

app = FastAPI()
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_ANON_KEY")  # Changed from SUPABASE_KEY to SUPABASE_ANON_KEY

def get_supabase() -> Client:
    if not supabase_url or not supabase_key:
        raise HTTPException(status_code=500, detail="Supabase credentials not configured")
    return create_client(supabase_url, supabase_key,
        options=ClientOptions(
            postgrest_client_timeout=10,
            storage_client_timeout=10,
        )
    )
REQUEST_TIMEOUT: float = 30.0
custom_sync_transport = httpx.HTTPTransport(retries=0) # 可选：如果你不希望 httpx 自动重试
sync_timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=30.0) # 总超时10秒，连接超时5秒
custom_sync_client = httpx.Client(timeout=sync_timeout, transport=custom_sync_transport)
# Authentication models
class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserRegister(BaseModel):
    email: EmailStr
    password: str

class VerifyRequest(BaseModel):
    email: EmailStr
    token: str
    username: str

class AuthResponse(BaseModel):
    token: str
    user_id: str


class GptQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = Field(default=None, description="Session ID for maintaining conversation context")


class GptQueryResponse(BaseModel):
    response: str
    session_id: str


# Create a dictionary to store chat histories by session_id
chat_histories = {}

# Initialize LangChain agent
def get_langchain_agent():
    # Language model
    llm = ChatModelFactory.get_model("gpt-4o")
    
    # Custom tools
    tools = [
        document_qa_tool,
        document_generation_tool,
        email_tool,
        excel_inspection_tool,
        directory_inspection_tool,
        finish_placeholder,
        ExcelAnalyser(
            llm=llm,
            prompt_file="./prompts/tools/excel_analyser.txt",
            verbose=True
        ).as_tool()
    ]
    
    # Define agent
    agent = ReActAgent(
        llm=llm,
        tools=tools,
        work_dir="./data",
        main_prompt_file="./prompts/main/main.txt",
        max_thought_steps=20,
    )
    
    return agent


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/api/login", response_model=AuthResponse)
async def login(user_credentials: UserLogin, supabase: Client = Depends(get_supabase)):
    try:
        response = supabase.auth.sign_in_with_password({
            'email': user_credentials.email,
            'password': user_credentials.password,
        })
        
        return {
            "token": response.session.access_token,
            "user_id": response.user.id
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}") # timed out


@app.post("/api/register", response_model=AuthResponse)
async def register(user_data: UserRegister, supabase: Client = Depends(get_supabase)):
    try:
        response = supabase.auth.sign_up({
            'email': user_data.email,
            'password': user_data.password,
        })
        
        if response.session is None:
            # Email confirmation required
            return {
                "token": "",  # No token until email confirmed
                "user_id": response.user.id
            }
        
        return {
            "token": response.session.access_token,
            "user_id": response.user.id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")


@app.post("/api/verify", response_model=AuthResponse)
async def verify(verify_data: VerifyRequest, supabase: Client = Depends(get_supabase)):
    try:
        response = supabase.auth.verify_otp(
            {
                "email": verify_data.email,
                "token": verify_data.token,
                "type": "email",  # Hardcoded to email as per Supabase requirements
            }
        )
        
        if not response.session:
            raise HTTPException(status_code=400, detail="Verification failed: Invalid token")
            
        update_user = (
            supabase.table("profiles")
            .update({"username": verify_data.username, "email": verify_data.email})
            .eq("id", response.user.id)
            .execute()
        )
        print("updated_user", update_user)
        return {
            "token": response.session.access_token,
            "user_id": response.user.id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Verification failed: {str(e)}")

# GPT Agent endpoint
@app.post("/api/gpt", response_model=GptQueryResponse)
async def query_gpt(request: GptQueryRequest):
    try:
        # Generate a random session ID if not provided
        session_id = request.session_id or f"session_{len(chat_histories) + 1}"
        
        # Get or create chat history for this session
        if session_id not in chat_histories:
            chat_histories[session_id] = ChatMessageHistory()
        
        # Get the agent
        agent = get_langchain_agent()
        
        # Run the agent with the query
        response = agent.run(
            task=request.query,
            chat_history=chat_histories[session_id],
            verbose=False
        )
        
        return {
            "response": response,
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT query failed: {str(e)}")


# For development server (uvicorn/granian)
# Command: granian --interface asgi main:app