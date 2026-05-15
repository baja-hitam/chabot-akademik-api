from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from pydantic import ValidationError
from sqlalchemy.orm import Session
import uuid

from app.core.config import get_settings
from app.infrastructure.database import get_db
from app.domain.models.user import User
from app.schemas.token import TokenPayload

settings = get_settings()
security_scheme = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    db: Session = Depends(get_db)
) -> User:
    try:
        payload = jwt.decode(
            credentials.credentials, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        token_data = TokenPayload(**payload)
        
        if not token_data.sub or token_data.type != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token or token type",
            )
    except (JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    user = db.query(User).filter(User.id == uuid.UUID(token_data.sub)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Account is not verified. Please verify OTP."
        )
        
    return user
