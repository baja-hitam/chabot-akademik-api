from pydantic import BaseModel, EmailStr
from datetime import datetime
import uuid

# Shared properties
class UserBase(BaseModel):
    email: EmailStr
    username: str

# Properties to receive via API on creation
class UserCreate(UserBase):
    password: str

# Properties to receive via API on login
class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Properties to return via API
class UserResponse(UserBase):
    id: uuid.UUID
    role: str
    is_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class OTPVerify(BaseModel):
    email: EmailStr
    otp_code: str
