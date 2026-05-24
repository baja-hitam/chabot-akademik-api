from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import uuid
from typing import List

from app.infrastructure.database import get_db
from app.repositories.user_repo import UserRepository
from app.schemas.user import UserResponseApi, UserListResponseApi, UserResponseUpdateApi, UserUpdate, UserCreate
from app.api.dependencies.auth import get_current_user
from app.domain.models.user import User
from app.services.auth_service import AuthService

router = APIRouter(tags=["User Management"])

@router.get("/", response_model=UserListResponseApi, summary="Dapatkan semua user")
def get_all_users(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    repo = UserRepository(db)
    users = repo.get_all()
    # Explicitly set nama_prodi for serialization
    for user in users:
        user.nama_prodi = user.prodi.nama_prodi if user.prodi else None
    return {
        "responseStatus": True,
        "responseMessage": "User berhasil ditemukan",
        "responseBody": users
    }

@router.get("/{id}", response_model=UserResponseApi, summary="Dapatkan user berdasarkan ID")
def get_user(id: uuid.UUID, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    repo = UserRepository(db)
    user = repo.get_by_id(id)
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    user.nama_prodi = user.prodi.nama_prodi if user.prodi else None
    return {
        "responseStatus": True,
        "responseMessage": "User berhasil ditemukan",
        "responseBody": user
    }

@router.post("/", response_model=UserResponseUpdateApi, summary="Buat user baru")
def create_user(payload: UserCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    auth_service = AuthService(db)
    auth_service.register_user(payload)
    return {
        "responseStatus": True,
        "responseMessage": "User berhasil dibuat",
        "responseBody": None
    }

@router.put("/{id}", response_model=UserResponseUpdateApi, summary="Update user")
def update_user(id: uuid.UUID, payload: UserUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    repo = UserRepository(db)
    user = repo.get_by_id(id)
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    
    update_data = payload.model_dump(exclude_unset=True)
    repo.update(user, update_data)
    
    return {
        "responseStatus": True,
        "responseMessage": "User berhasil diupdate",
        "responseBody": None
    }

@router.delete("/{id}", response_model=UserResponseUpdateApi, summary="Hapus user")
def delete_user(id: uuid.UUID, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    repo = UserRepository(db)
    user = repo.get_by_id(id)
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    repo.delete(user)
    return {
        "responseStatus": True,
        "responseMessage": "User berhasil dihapus",
        "responseBody": None
    }
