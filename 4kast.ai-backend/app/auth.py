# app/auth.py
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from .security import verify_password, create_access_token, hash_password
from .db import get_hana_connection

router = APIRouter(prefix="/api/auth", tags=["auth"])

class UserIn(BaseModel):
    username: str
    password: str

@router.post("/token")
def login(payload: UserIn, conn=Depends(get_hana_connection)):
    hashed = get_userpassword(payload.username, conn)
    if hashed != payload.password:
    # if not hashed or not verify_password(payload.password, hashed):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(payload.username)
    return {"access_token": token, "token_type": "bearer"}

def get_userpassword(username, conn):
    try:
        with conn.cursor() as cur:
            query = "SELECT PASSWORDHASH FROM DBADMIN.USERS WHERE username = ?"
            cur.execute(query, (username,))
            result = cur.fetchone()
            print(result[0])
            return result[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")
