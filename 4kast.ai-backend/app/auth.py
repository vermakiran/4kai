# app/auth.py
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel
from .security import create_access_token
from .db import get_hana_connection
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
import os
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

router = APIRouter(prefix="/api/auth", tags=["auth"])

load_dotenv()

# OAuth config
config_data = {
    "GOOGLE_CLIENT_ID": os.getenv("GOOGLE_CLIENT_ID"),
    "GOOGLE_CLIENT_SECRET": os.getenv("GOOGLE_CLIENT_SECRET"),
    "MICROSOFT_CLIENT_ID": os.getenv("MICROSOFT_CLIENT_ID"),
    "MICROSOFT_CLIENT_SECRET": os.getenv("MICROSOFT_CLIENT_SECRET"),
    "REDIRECT_URI": os.getenv("REDIRECT_URI"),
}
config = Config(environ=config_data)
oauth = OAuth(config)

oauth.register(
    name="google",
    client_id=config_data["GOOGLE_CLIENT_ID"],
    client_secret=config_data["GOOGLE_CLIENT_SECRET"],
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

oauth.register(
    name="microsoft",
    client_id=config_data["MICROSOFT_CLIENT_ID"],
    client_secret=config_data["MICROSOFT_CLIENT_SECRET"],
    server_metadata_url="https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


class UserIn(BaseModel):
    username: str
    password: str

@router.get("/")
async def home(request: Request):
    user = request.session.get("user")
    # if user:
    #     return {"message": "Logged in", "user": user}
    return HTMLResponse("""
    <h3>Select Login Provider</h3>
    <a href="/api/auth/login/google">Login with Google</a><br>
    <a href="/api/auth/login/microsoft">Login with Microsoft</a>
    """)


# 2. Start login based on provider
@router.get("/login/{provider}")
async def login(request: Request, provider: str):
    if provider not in ("google", "microsoft"):
        raise HTTPException(status_code=400, detail="Unsupported provider")    
    request.session["provider"] = provider  # Save provider  
    redirect_uri = config_data["REDIRECT_URI"]
    print("Redirect URI:", config_data["REDIRECT_URI"])
    
    return await oauth.create_client(provider).authorize_redirect(request, redirect_uri)

@router.get("/auth/callback")
async def auth_callback(request: Request):
    print("testing....")
    try:
        print("testing....")
        provider = request.session.get("provider")
        print(provider)
        if not provider:
            raise HTTPException(status_code=400, detail="Missing provider context")
        token = await oauth.create_client(provider).authorize_access_token(request)
        user_info = token.get("userinfo")
        email = user_info.get("email")
        
        print(email)

        # with conn.cursor() as cur:
        #     cur.execute("SELECT username FROM DBADMIN.USERS")
        #     result = cur.fetchall()
        #     allowed_emails = [row[0] for row in result]
        #     print(allowed_emails)
        allowed_emails = {
            "vermakiran1998@gmail.com"
        }

        if email in allowed_emails:
            request.session["user"] = {"email": email, "provider": provider}
            return RedirectResponse(url="/api/auth/")
        else:
            return HTMLResponse(f"<h2>Access Denied</h2><p>{email} is not authorized to use this app.</p>")
    except Exception as e:
        return HTMLResponse(f"<h2>Authentication Error</h2><pre>{e}</pre>", status_code=400)

@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/api/auth/")

@router.post("/token")
def login(payload: UserIn, conn=Depends(get_hana_connection)):
    hashed = get_userpassword(payload.username, conn)
    if hashed != payload.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(payload.username)
    return {"access_token": token, "token_type": "bearer"}

def get_userpassword(username, conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT PASSWORDHASH FROM DBADMIN.USERS WHERE username = ?", (username,))
            result = cur.fetchone()
            return result[0] if result else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")
