from datetime import datetime
import uuid
from pydantic import BaseModel, Field
from app.domain.models.announcement import AnnouncementStatus

class AnnouncementResponse(BaseModel):
    id: uuid.UUID
    title: str
    content: str
    status: AnnouncementStatus
    created_at: datetime
    
    class Config:
        from_attributes = True

class AnnouncementCreate(BaseModel):
    title: str
    content: str
    status: AnnouncementStatus = AnnouncementStatus.INFO
