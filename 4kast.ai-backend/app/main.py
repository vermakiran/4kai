from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.auth import router as auth_router
from app.engine import router as engine_router
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # your frontend origin
    allow_methods=["*"],
    allow_headers=["*"]
)

app.add_middleware(
    SessionMiddleware,
    secret_key='hurgfweyreyrygxyugqweyeuy'
)

# allow_origins=["http://localhost:3000"]

app.include_router(auth_router)
app.include_router(engine_router)