from typing import Optional
from pydantic import BaseModel, EmailStr, model_validator
from datetime import datetime
import uuid

# Shared properties
class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: str
    role: str
    kd_prodi: Optional[int] = None

# Properties to receive via API on creation
class UserCreate(UserBase):
    password: str

    @model_validator(mode='after')
    def validate_kd_prodi(self):
        if self.role == 'student' and not self.kd_prodi:
            raise ValueError('kd_prodi is required for student role')
        if self.role == 'prodi' and not self.kd_prodi:
            raise ValueError('kd_prodi is required for prodi role')
        return self

# Properties to receive via API on login
class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Properties to return via API
class UserResponse(UserBase):
    id: uuid.UUID
    role: str
    is_verified: bool
    nama_prodi: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True
        
class UserResponseApi(BaseModel):
    responseStatus: bool
    responseMessage: str
    responseBody: UserResponse

class UserUpdate(BaseModel):
    username: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[str] = None
    kd_prodi: Optional[int] = None

class UserListResponseApi(BaseModel):
    responseStatus: bool
    responseMessage: str
    responseBody: list[UserResponse]

class UserResponseUpdateApi(BaseModel):
    responseStatus: bool
    responseMessage: str
    responseBody: None

class UserResponseBodyLogin(BaseModel):
    user: UserResponse
    access_token: Optional[str] = None
    token_type: Optional[str] = None
    is_verified: bool

class UserResponseLoginApi(BaseModel):
    responseStatus: bool
    responseMessage: str
    responseBody: UserResponseBodyLogin

class OTPVerify(BaseModel):
    email: EmailStr
    otp_code: str
