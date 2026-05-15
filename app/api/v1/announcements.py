from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from app.infrastructure.database import get_db
from app.repositories.announcement_repo import AnnouncementRepository
from app.schemas.announcement import AnnouncementResponse, AnnouncementCreate
from app.api.dependencies.auth import get_current_user
from app.domain.models.user import User

router = APIRouter(tags=["Announcements"])

@router.get("/", response_model=List[AnnouncementResponse], summary="Dapatkan daftar pengumuman")
def get_announcements(
    limit: int = 5, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)  # Requires auth to see announcements
):
    repo = AnnouncementRepository(db)
    return repo.get_latest_announcements(limit=limit)

@router.post("/", response_model=AnnouncementResponse, summary="Buat pengumuman baru (Admin)")
def create_announcement(
    payload: AnnouncementCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Optional: You can check if current_user.role == 'admin' here
    repo = AnnouncementRepository(db)
    return repo.create_announcement(
        title=payload.title,
        content=payload.content,
        status=payload.status
    )
