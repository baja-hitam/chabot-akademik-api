from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.POSTGRES_URL, 
    pool_pre_ping=True,
    # echo=True  # Uncomment to see SQL queries
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for FastAPI to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
