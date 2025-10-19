from fastapi import FastAPI
import asyncio
from contextlib import asynccontextmanager

from sentence_transformers import SentenceTransformer
import uvicorn
from config.qdrant import qdrant
from qdrant_client.http import models as qmodels
from config.config import (
    EMBEDDING_MODEL,
    QDRANT_CHAT_MESSAGES_COLLECTION_NAME,
    QDRANT_DOCUMENT_COLLECTION_NAME,
    QDRANT_MEETING_TRANSCRIPTS_COLLECTION_NAME,
    SERVER_HOST,
    SERVER_PORT,
    SERVER_RELOAD,
)
from config.models import get_model, set_model
from config.qdrant_indexes import (
    add_indexes_to_existing_document_collection,
    create_indexes_for_chat_messages_collection,
    create_indexes_for_transcript_collection,
)
from helpers.embedding import embedding_worker, embedding_chat_worker
from fastapi.middleware.cors import CORSMiddleware
from routers.embedding import router as embedding_router
from routers.search import router as rag_router
from routers.serve import router as serve_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    task = asyncio.create_task(embedding_worker())
    print("Background worker for transcript processing started.")

    task = asyncio.create_task(embedding_chat_worker())
    print("Background worker for chat processing started.")

    print("embedding model: ", EMBEDDING_MODEL)

    set_model(EMBEDDING_MODEL)
    print("Embedding model loaded.")

    # Create collection if it doesn't exist
    if QDRANT_DOCUMENT_COLLECTION_NAME not in [
        c.name for c in qdrant.get_collections().collections
    ]:
        model = get_model()
        qdrant.recreate_collection(
            collection_name=QDRANT_DOCUMENT_COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=qmodels.Distance.COSINE,
            ),
        )
        print(f"🆕 Created Qdrant collection: {QDRANT_DOCUMENT_COLLECTION_NAME}")

        add_indexes_to_existing_document_collection()

    # Create collection if it doesn't exist
    if QDRANT_MEETING_TRANSCRIPTS_COLLECTION_NAME not in [
        c.name for c in qdrant.get_collections().collections
    ]:
        model = get_model()
        qdrant.recreate_collection(
            collection_name=QDRANT_MEETING_TRANSCRIPTS_COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=qmodels.Distance.COSINE,
            ),
        )
        print(
            f"🆕 Created Qdrant collection: {QDRANT_MEETING_TRANSCRIPTS_COLLECTION_NAME}"
        )

        create_indexes_for_transcript_collection()

    if QDRANT_CHAT_MESSAGES_COLLECTION_NAME not in [
        c.name for c in qdrant.get_collections().collections
    ]:
        model = get_model()
        qdrant.recreate_collection(
            collection_name=QDRANT_CHAT_MESSAGES_COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=qmodels.Distance.COSINE,
            ),
        )
        print(f"🆕 Created Qdrant collection: {QDRANT_CHAT_MESSAGES_COLLECTION_NAME}")

        create_indexes_for_chat_messages_collection()

    # Yield control to the app
    yield

    # Shutdown logic
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("Background worker stopped cleanly.")


app = FastAPI(lifespan=lifespan)

# --- Add CORS Middleware ---
origins = [
    "http://localhost:5173",  # frontend dev server
    # "http://127.0.0.1:3000",  # alternate local address
    "https://your-production-domain.com",  # your deployed frontend
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] to allow all origins (for testing)
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],  # allow all headers
)

app.include_router(rag_router)
app.include_router(embedding_router)
app.include_router(serve_router)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI RAG Search API is running", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run("main:app", host=SERVER_HOST, port=SERVER_PORT, reload=SERVER_RELOAD)
