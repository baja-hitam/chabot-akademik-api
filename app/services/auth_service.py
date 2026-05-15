import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.domain.models.user import User
from app.schemas.user import UserCreate, UserLogin, OTPVerify
from app.core.security import get_password_hash, verify_password, generate_otp, create_access_token

logger = logging.getLogger(__name__)

class AuthService:
    def __init__(self, db: Session):
        self.db = db

    def register_user(self, user_in: UserCreate) -> User:
        if self.db.query(User).filter(User.email == user_in.email).first():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        if self.db.query(User).filter(User.username == user_in.username).first():
            raise HTTPException(status_code=400, detail="Username already taken")

        otp = generate_otp()
        otp_expiry = datetime.utcnow() + timedelta(minutes=5)
        
        user = User(
            email=user_in.email,
            username=user_in.username,
            password_hash=get_password_hash(user_in.password),
            otp_code=otp,
            otp_expires_at=otp_expiry
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        
        self._send_otp_email(user.email, otp)
        return user

    def login(self, user_in: UserLogin):
        user = self.db.query(User).filter(User.email == user_in.email).first()
        if not user or not verify_password(user_in.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Incorrect email or password")
            
        if not user.is_verified:
            # Resend OTP
            user.otp_code = generate_otp()
            user.otp_expires_at = datetime.utcnow() + timedelta(minutes=5)
            self.db.commit()
            self._send_otp_email(user.email, user.otp_code)
            
            return {"is_verified": False, "message": "Please verify your account. New OTP sent to email."}
            
        access_token = create_access_token(subject=user.id)
        return {"access_token": access_token, "token_type": "bearer", "is_verified": True}

    def verify_otp(self, payload: OTPVerify):
        user = self.db.query(User).filter(User.email == payload.email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
            
        if user.is_verified:
            raise HTTPException(status_code=400, detail="User already verified")
            
        if not user.otp_code or user.otp_code != payload.otp_code:
            raise HTTPException(status_code=400, detail="Invalid OTP code")
            
        if not user.otp_expires_at or user.otp_expires_at < datetime.utcnow():
            raise HTTPException(status_code=400, detail="OTP code expired")
            
        user.is_verified = True
        user.otp_code = None
        user.otp_expires_at = None
        self.db.commit()
        
        access_token = create_access_token(subject=user.id)
        return {"access_token": access_token, "token_type": "bearer", "is_verified": True}
        
    def _send_otp_email(self, email: str, otp: str):
        # TODO: Implement actual SMTP sending here.
        # For now, we mock it by printing to console.
        logger.info(f"========== MOCK EMAIL ==========")
        logger.info(f"To: {email}")
        logger.info(f"Subject: Academic Chatbot - Verify your account")
        logger.info(f"Your OTP code is: {otp}")
        logger.info(f"=================================")
