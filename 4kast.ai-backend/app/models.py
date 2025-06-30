from pydantic import BaseModel, EmailStr, constr
from typing import List, Optional, Dict, Annotated

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

class UserCreate(BaseModel):
    full_name: Annotated[str, constr(min_length=1)]
    email: EmailStr
    role: Annotated[str, constr(pattern="^(Admin|Planner)$")]
    employee_id: Annotated[str, constr(max_length=10)]
    password: Annotated[str, constr(min_length=6)] = "Temp123!"

class UserOut(BaseModel):
    userid: int
    full_name: Optional[str] = None
    email: str
    role: str
    employee_id: Optional[str] = None
    org_id: int
    isactive: bool