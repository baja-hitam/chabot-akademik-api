from sqlalchemy.orm import Session
from app.domain.models.prodi import Prodi

class ProdiRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self):
        return self.db.query(Prodi).all()

    def get_by_kd(self, kd_prodi: int):
        return self.db.query(Prodi).filter(Prodi.kd_prodi == kd_prodi).first()

    def create(self, nama_prodi: str) -> Prodi:
        prodi = Prodi(nama_prodi=nama_prodi)
        self.db.add(prodi)
        self.db.commit()
        self.db.refresh(prodi)
        return prodi

    def update(self, prodi: Prodi, nama_prodi: str) -> Prodi:
        prodi.nama_prodi = nama_prodi
        self.db.commit()
        self.db.refresh(prodi)
        return prodi

    def delete(self, prodi: Prodi):
        self.db.delete(prodi)
        self.db.commit()
