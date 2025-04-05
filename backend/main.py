import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from supabase import create_client, Client, ClientOptions
from dotenv import load_dotenv
import httpx
from typing import Optional

# LangChain imports
from Agent.ReAct import ReActAgent
from Models.Factory import ChatModelFactory
from Tools import *
from Tools.PythonTool import ExcelAnalyser
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_openai import ChatOpenAI

# Langfuse 监控支持
from Utils.LangfuseMonitor import LangfuseMonitor

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

# 初始化 Langfuse 监控
langfuse_monitor = LangfuseMonitor()

# Initialize LangChain agent
def get_langchain_agent():
    # Ensure environment variables are properly loaded
    from dotenv import load_dotenv
    load_dotenv()
    
    # Print debug info about available environment variables (without showing the actual values)
    print("Available environment variables:")
    required_vars = ["OPENAI_API_KEY", "OPENAI_API_BASE", "LOCAL_API_BASE", "GITHUB_TOKEN"]
    for var in required_vars:
        print(f"  {var}: {'✅ Set' if os.environ.get(var) else '❌ Not set'}")
    
    # Create model with more explicit error handling
    try:
        # Use a safer default model configuration
        llm = ChatOpenAI(
            model_name="gpt-4o",  # Use a more reliable model
            temperature=0.2,
            request_timeout=60,  # Longer timeout
        )
        print("LLM initialized successfully")
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        # Fallback to a very simple configuration
        llm = ChatOpenAI(temperature=0)
    
    # Use a minimal set of tools to reduce potential errors
    tools = [
        finish_placeholder,  # Always include the finish tool
    ]
    
    # Add optional tools if they're available and properly initialized
    try:
        tools.append(document_qa_tool)
        tools.append(document_generation_tool)
        tools.append(github_document_query_tool)
        print("Document tools loaded successfully")
    except Exception as e:
        print(f"Error loading document tools: {str(e)}")
    
    # Define agent with more explicit error handling
    try:
        agent = ReActAgent(
            llm=llm,
            tools=tools,
            work_dir="./data",
            main_prompt_file="./prompts/main/main.txt",
            max_thought_steps=5,  # Reduce max steps for initial testing
        )
        print("Agent initialized successfully")
        return agent
    except Exception as e:
        print(f"Error initializing agent: {str(e)}")
        raise


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
        
        # 初始化Langfuse监控
        langfuse_monitor = LangfuseMonitor()
        langfuse_handler = langfuse_monitor.get_langchain_handler(session_id=session_id)
        
        # 创建回调列表，如果Langfuse启用，则包含Langfuse处理程序
        callbacks = []
        if langfuse_handler:
            callbacks.append(langfuse_handler)
        
        # Get or create chat history for this session
        if session_id not in chat_histories:
            chat_histories[session_id] = ChatMessageHistory()
        
        try:
            # Get the agent
            agent = get_langchain_agent()
            
            # Log the query
            print(f"Processing query: {request.query[:50]}...")
            
            # Run the agent with the query，添加Langfuse回调
            response = agent.run(
                task=request.query,
                chat_history=chat_histories[session_id],
                verbose=True,  # Enable verbose output for debugging
                session_id=session_id,  # 传递会话ID
                user_id=None,  # 可选：如果有用户ID也可以传递
                callbacks=callbacks  # 传递Langfuse回调
            )
            
            return {
                "response": response,
                "session_id": session_id
            }
            
        except Exception as agent_error:
            # If agent execution fails, provide a fallback response
            error_message = str(agent_error)
            print(f"Agent execution error: {error_message}")
            
            # Provide a more user-friendly message for the streaming error
            if "No generation chunks were returned" in error_message:
                response_message = "抱歉，处理请求时发生错误：模型未返回任何内容。可能是网络连接问题或模型服务暂时不可用。请稍后再试。"
            else:
                response_message = f"抱歉，处理请求时发生问题: {error_message}"
                
            return {
                "response": response_message,
                "session_id": session_id
            }
            
    except Exception as e:
        import traceback
        print(f"Error in query_gpt: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"GPT query failed: {str(e)}")


# For development server (uvicorn/granian)
# Command: granian --interface asgi main:app