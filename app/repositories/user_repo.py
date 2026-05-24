import uuid
from sqlalchemy.orm import Session
from app.domain.models.user import User

class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self):
        return self.db.query(User).all()

    def get_by_id(self, id: uuid.UUID):
        return self.db.query(User).filter(User.id == id).first()

    def update(self, user: User, payload: dict):
        for key, value in payload.items():
            if value is not None:
                setattr(user, key, value)
        self.db.commit()
        self.db.refresh(user)
        return user

    def delete(self, user: User):
        self.db.delete(user)
        self.db.commit()
