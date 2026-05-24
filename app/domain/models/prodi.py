from sqlalchemy import Column, String, Integer
from app.domain.models.base import Base

class Prodi(Base):
    __tablename__ = "mprodi"

    kd_prodi = Column(Integer, primary_key=True, index=True, autoincrement=True)
    nama_prodi = Column(String(255), nullable=False)
