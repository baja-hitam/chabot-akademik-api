from pydantic import BaseModel
import uuid

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenPayload(BaseModel):
    sub: str | None = None
    exp: int | None = None
    type: str | None = None
