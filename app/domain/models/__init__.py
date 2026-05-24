from app.domain.models.base import Base
from app.domain.models.user import User
from app.domain.models.chat import ChatSession, ChatMessage, SenderRole
from app.domain.models.announcement import Announcement, AnnouncementStatus
from app.domain.models.prodi import Prodi

# This file imports all models so that when we import `Base` from here, 
# it has all the metadata registered for `Base.metadata.create_all()`.
