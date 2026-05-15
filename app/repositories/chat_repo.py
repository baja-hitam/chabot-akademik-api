import uuid
from sqlalchemy.orm import Session
from app.domain.models.chat import ChatSession, ChatMessage, SenderRole

class ChatRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_session(self, user_id: uuid.UUID, title: str) -> ChatSession:
        session = ChatSession(user_id=user_id, title=title)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def get_session_by_id(self, session_id: uuid.UUID) -> ChatSession:
        return self.db.query(ChatSession).filter(ChatSession.id == session_id).first()

    def get_user_sessions(self, user_id: uuid.UUID):
        return self.db.query(ChatSession).filter(ChatSession.user_id == user_id).order_by(ChatSession.created_at.desc()).all()

    def save_message(self, session_id: uuid.UUID, sender_role: str, content: str) -> ChatMessage:
        role_enum = SenderRole.USER if sender_role == "user" else SenderRole.ASSISTANT
        msg = ChatMessage(session_id=session_id, sender_role=role_enum, content=content)
        self.db.add(msg)
        self.db.commit()
        self.db.refresh(msg)
        return msg

    def get_messages_by_session(self, session_id: uuid.UUID, limit: int = 10) -> list[ChatMessage]:
        # Return the last 'limit' messages, ordered by created_at ascending
        messages = self.db.query(ChatMessage)\
            .filter(ChatMessage.session_id == session_id)\
            .order_by(ChatMessage.created_at.desc())\
            .limit(limit)\
            .all()
        return list(reversed(messages))
