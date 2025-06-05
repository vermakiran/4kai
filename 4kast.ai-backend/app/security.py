# app/security.py
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import Header, HTTPException, status
from typing import Optional
from .config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(sub: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": sub, "exp": expire}
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

def validate_token(token: str):
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        
        # Check for expiration
        exp = payload.get("exp")
        if exp is None:
            raise HTTPException(status_code=401, detail="Token has no expiration")
        
        # Convert expiration to datetime and compare with current time
        expiration = datetime.fromtimestamp(exp, timezone.utc)
        if expiration < datetime.now(timezone.utc):
            raise HTTPException(status_code=401, detail="Token has expired")

        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Token missing username")
        
        return username

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(
    authorization: Optional[str] = Header(None, description="Bearer <JWT token>")
) -> str:
    """
    Expects header:
      Authorization: Bearer <token>
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header"
        )
    token = authorization.split(" ", 1)[1]
    return validate_token(token)