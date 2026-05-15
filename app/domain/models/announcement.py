import uuid
import enum
from sqlalchemy import Column, String, Text, DateTime, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.domain.models.base import Base

class AnnouncementStatus(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    SUCCESS = "success"

class Announcement(Base):
    __tablename__ = "announcements"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    status = Column(Enum(AnnouncementStatus), default=AnnouncementStatus.INFO, nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
