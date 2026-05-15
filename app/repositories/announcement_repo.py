from sqlalchemy.orm import Session
from app.domain.models.announcement import Announcement

class AnnouncementRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_latest_announcements(self, limit: int = 5):
        return self.db.query(Announcement)\
            .order_by(Announcement.created_at.desc())\
            .limit(limit)\
            .all()

    def create_announcement(self, title: str, content: str, status: str) -> Announcement:
        announcement = Announcement(title=title, content=content, status=status)
        self.db.add(announcement)
        self.db.commit()
        self.db.refresh(announcement)
        return announcement
