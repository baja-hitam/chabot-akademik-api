from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.infrastructure.database import get_db
from app.repositories.prodi_repo import ProdiRepository
from app.schemas.prodi import ProdiResponse,ProdiCreate,ProdiResponseApi, ProdiDetailResponseApi, ProdiResponseCreateApi,ProdiResponseUtilsApi
from app.api.dependencies.auth import get_current_user
from app.domain.models.user import User

router = APIRouter(tags=["Master Prodi"])

@router.get("/", response_model=ProdiResponseApi, summary="Dapatkan daftar prodi")
def get_all_prodi(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    repo = ProdiRepository(db)
    return {
        "responseStatus": True,
        "responseMessage": "Prodi berhasil ditemukan",
        "responseBody": repo.get_all()
    }

@router.get("/utils", response_model=ProdiResponseUtilsApi, summary="Dapatkan daftar prodi untuk utils")
def get_all_prodi_utils(
    db: Session = Depends(get_db),
):
    repo = ProdiRepository(db)
    prodi = repo.get_all()
    if not prodi:
        raise HTTPException(status_code=404, detail="Prodi tidak ditemukan")
    prodi_utils = []
    for prodi_item in prodi:
        prodi_utils.append({
            "value": prodi_item.kd_prodi,
            "label": prodi_item.nama_prodi
        })
    return {
        "responseStatus": True,
        "responseMessage": "Prodi berhasil ditemukan",
        "responseBody": prodi_utils
    }

@router.get("/{kd_prodi}", response_model=ProdiDetailResponseApi, summary="Dapatkan prodi berdasarkan kode")
def get_prodi_by_kd(
    kd_prodi: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    repo = ProdiRepository(db)
    prodi = repo.get_by_kd(kd_prodi)
    if not prodi:
        raise HTTPException(status_code=404, detail="Prodi tidak ditemukan")
    return {
        "responseStatus": True,
        "responseMessage": "Prodi berhasil ditemukan",
        "responseBody": prodi
    }

@router.post("/", response_model=ProdiResponseCreateApi, summary="Buat prodi baru")
def create_prodi(
    payload: ProdiCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    repo = ProdiRepository(db)
    prodi = repo.create(
        nama_prodi=payload.nama_prodi
    )
    return {
        "responseStatus": True,
        "responseMessage": "Prodi berhasil dibuat",
        "responseBody": None
    }

@router.put("/{kd_prodi}", response_model=ProdiResponseCreateApi, summary="Update prodi")
def update_prodi(
    kd_prodi: int,
    payload: ProdiCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    repo = ProdiRepository(db)
    prodi = repo.get_by_kd(kd_prodi)
    if not prodi:
        raise HTTPException(status_code=404, detail="Prodi tidak ditemukan")
    
    prodi = repo.update(prodi, payload.nama_prodi)
    return {
        "responseStatus": True,
        "responseMessage": "Prodi berhasil diupdate",
        "responseBody": None
    }

@router.delete("/{kd_prodi}", response_model=ProdiResponseCreateApi, summary="Hapus prodi")
def delete_prodi(
    kd_prodi: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    repo = ProdiRepository(db)
    prodi = repo.get_by_kd(kd_prodi)
    if not prodi:
        raise HTTPException(status_code=404, detail="Prodi tidak ditemukan")
    
    repo.delete(prodi)
    return {
        "responseStatus": True,
        "responseMessage": "Prodi berhasil dihapus",
        "responseBody": None
    }
