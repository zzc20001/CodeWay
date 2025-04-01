import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from supabase import create_client, Client, ClientOptions
from dotenv import load_dotenv

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
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


@app.post("/api/register", response_model=AuthResponse)
async def register(user_data: UserRegister, supabase: Client = Depends(get_supabase)):
    # try:
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
    # except Exception as e:
    #     raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")


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

# For development server (uvicorn/granian)
# Command: granian --interface asgi main:app