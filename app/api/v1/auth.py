from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.infrastructure.database import get_db
from app.schemas.user import UserCreate, UserLogin, OTPVerify, UserResponse, UserResponseApi, UserResponseLoginApi
from app.schemas.token import Token
from app.services.auth_service import AuthService
from app.api.dependencies.auth import get_current_user
from app.domain.models.user import User

router = APIRouter(tags=["Auth"])

@router.post("/register", response_model=UserResponseApi, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    auth_service = AuthService(db)
    user = auth_service.register_user(user_in)
    return user

@router.post("/login", response_model=UserResponseLoginApi)
def login(user_in: UserLogin, db: Session = Depends(get_db)):
    auth_service = AuthService(db)
    return auth_service.login(user_in)

@router.post("/verify-otp", response_model=UserResponseLoginApi)
def verify_otp(payload: OTPVerify, db: Session = Depends(get_db)):
    auth_service = AuthService(db)
    return auth_service.verify_otp(payload)

@router.get("/me", response_model=UserResponseApi)
def get_me(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    auth_service = AuthService(db)
    user = auth_service.get_by_id(current_user.id)
    return user
