import logging
import uuid

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from hybrid_retriever import search
from rag_service import retrieve_context
from auth import get_current_user
from conversation_store import add_message, get_history

app = FastAPI(
    title="Pinecone RAG API",
    version="1.0.0"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    company_id: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    category: str | None = None
    order_id: str | None = None

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    category: str | None = None
    order_id: str | None = None

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "pinecone-rag-api",
    }

@app.post("/search")
def search_knowledge(request: SearchRequest):

    results = search(
        question=request.question,
        company_id=request.company_id,
        top_k=request.top_k,
        alpha=request.alpha,
        category=request.category,
        order_id=request.order_id,
    )

    matches = []

    for match in results.matches:
        matches.append({
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata,
        })

    return {
        "question": request.question,
        "company_id": request.company_id,
        "matches": matches,
    }

@app.post("/ask")
def ask_question(
    request: AskRequest,
    current_user: dict = Depends(get_current_user),
):

    request_id = str(uuid.uuid4())

    user_id = current_user["user_id"]
    company_id = current_user["company_id"]

    logger.info(
        "[%s] RAG request started | user_id=%s | company_id=%s",
        request_id,
        user_id,
        company_id,
    )

    try:
        # Get only the previous conversation
        history = get_history(request.session_id)

        rag_results = retrieve_context(
            question=request.question,
            company_id=company_id,
            request_id=request_id,
            top_k=request.top_k,
            alpha=request.alpha,
            category=request.category,
            order_id=request.order_id,
            history=history,
        )

        # Store the current user message AFTER retrieval
        add_message(
            request.session_id,
            "user",
            request.question,
        )

        # Store the generated answer
        add_message(
            request.session_id,
            "assistant",
            rag_results["answer"],
        )

        logger.info(
            "[%s] RAG request completed",
            request_id,
        )

        return {
            "request_id": request_id,
            "question": request.question,
            "company_id": company_id,
            "answer": rag_results["answer"],
            "context": rag_results["context"],
        }

    except Exception:
        logger.exception(
            "[%s] RAG request failed",
            request_id,
        )

        raise HTTPException(
            status_code=503,
            detail={
                "message": "RAG service is temporarily unavailable",
                "request_id": request_id,
            },
        )
