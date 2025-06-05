from pydantic import BaseModel
from typing import List, Optional, Dict

class LoginRequest(BaseModel):
    username: str
    password: str


class ForecastInput(BaseModel):
    filename: str
    data: List[dict]


class UploadCleanedData(BaseModel):
    filename: str
    granularity: str
    timeBucket: str
    forecastHorizon: int
    data: List[Dict[str, str]]

class DeleteFileRequest(BaseModel):
    filename: str