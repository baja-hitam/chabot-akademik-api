"""
Chat Endpoint
Provides both standard JSON responses and streaming (SSE) responses
for the RAG-powered academic chatbot, with session management.
"""

import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.schemas.models import ChatRequest, ChatResponse, ChatSessionResponse, ChatMessageDetail
from app.services.ai_logic import ai_logic_service
from app.services.vector_store import vector_store_service
from app.infrastructure.database import get_db
from app.api.dependencies.auth import get_current_user
from app.domain.models.user import User
from app.repositories.chat_repo import ChatRepository

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

@router.get("/sessions", response_model=list[ChatSessionResponse])
def get_chat_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mendapatkan daftar sesi chat milik user saat ini."""
    repo = ChatRepository(db)
    return repo.get_user_sessions(current_user.id)

@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageDetail])
def get_session_messages(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mendapatkan riwayat pesan dari suatu sesi tertentu."""
    repo = ChatRepository(db)
    session = repo.get_session_by_id(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Akses ditolak")
    
    # Passing limit 100 or none to get the full history for UI display
    return repo.get_messages_by_session(session_id, limit=100)

def format_history(messages):
    """Format DB messages into LangChain chat history format."""
    return [
        ("human" if msg.sender_role.value == "user" else "assistant", msg.content)
        for msg in messages
    ]

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with Academic AI",
)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> ChatResponse:
    """Process a question through the RAG pipeline and return a full response."""
    try:
        chat_repo = ChatRepository(db)
        
        # 1. Get or Create Session
        if request.session_id:
            session_id = uuid.UUID(request.session_id)
            session = chat_repo.get_session_by_id(session_id)
            if not session or session.user_id != current_user.id:
                raise HTTPException(status_code=403, detail="Sesi tidak ditemukan atau akses ditolak")
        else:
            session = chat_repo.create_session(user_id=current_user.id, title=request.question[:50])
            session_id = session.id

        # 2. Save User Message
        chat_repo.save_message(session_id=session_id, sender_role="user", content=request.question)

        # 3. Retrieve Chat History
        messages = chat_repo.get_messages_by_session(session_id, limit=10)
        chat_history = format_history(messages)

        # 4. Generate AI Response
        response = await ai_logic_service.get_answer(
            question=request.question,
            category=request.category.value if request.category else None,
            chat_history=chat_history
        )

        # 5. Save AI Response
        chat_repo.save_message(session_id=session_id, sender_role="assistant", content=response.answer)

        # Include session_id in response
        response.session_id = str(session_id)

        return response
    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Gagal memproses pertanyaan: {str(e)}",
        )

@router.post(
    "/chat/stream",
    summary="Chat with Streaming (SSE)",
)
async def chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> StreamingResponse:
    """Stream the RAG answer token-by-token using Server-Sent Events (SSE)."""

    async def event_generator():
        chat_repo = ChatRepository(db)
        try:
            # 1. Get or Create Session
            if request.session_id:
                session_id = uuid.UUID(request.session_id)
                session = chat_repo.get_session_by_id(session_id)
                if not session or session.user_id != current_user.id:
                    yield f"event: error\ndata: {json.dumps({'error': 'Akses ditolak'})}\n\n"
                    return
            else:
                session = chat_repo.create_session(user_id=current_user.id, title=request.question[:50])
                session_id = session.id

            # 2. Save User Message
            chat_repo.save_message(session_id=session_id, sender_role="user", content=request.question)
            
            # Send session_id info
            yield f"event: session\ndata: {json.dumps({'session_id': str(session_id)})}\n\n"

            # 3. Retrieve History
            messages = chat_repo.get_messages_by_session(session_id, limit=10)
            chat_history = format_history(messages)

            # 4. Send source documents info
            retrieved_docs = vector_store_service.search_similar(
                query=request.question,
                category=request.category.value if request.category else None,
            )

            sources_data = [
                {
                    "source": doc["source"],
                    "category": doc["category"],
                    "relevance_score": doc["relevance_score"],
                    "document_year": doc.get("document_year"),
                    "is_latest": doc.get("is_latest", True),
                    "ocr_used": doc.get("ocr_used", False),
                }
                for doc in retrieved_docs
            ]
            yield f"event: sources\ndata: {json.dumps(sources_data, ensure_ascii=False)}\n\n"

            # 5. Stream tokens and collect full answer
            full_answer = ""
            async for token in ai_logic_service.stream_answer(
                question=request.question,
                category=request.category.value if request.category else None,
                chat_history=chat_history
            ):
                full_answer += token
                yield f"event: token\ndata: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

            # 6. Save final AI answer to DB
            chat_repo.save_message(session_id=session_id, sender_role="assistant", content=full_answer)

            # Send done event
            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            logger.error("Streaming error: %s", e, exc_info=True)
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
