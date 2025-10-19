import asyncio
from uuid import uuid4
from config.logger import logger
from config.config import QDRANT_MEETING_TRANSCRIPTS_COLLECTION_NAME
from config.qdrant import qdrant
from config.models import get_model

# Simple async queue to decouple ingestion & embedding


embedding_queue = asyncio.Queue()


async def embedding_worker():
    """Continuously embed and push chunks to Qdrant"""
    while True:
        chunk = await embedding_queue.get()
        try:
            _model = get_model()
            logger.info(f"Embedding chunk_id: {chunk.get('block_id')}")
            logger.info(f"Model used for embedding: {_model}")
            embedding = _model.encode(chunk["text"], normalize_embeddings=True).tolist()
            qdrant.upsert(
                collection_name=QDRANT_MEETING_TRANSCRIPTS_COLLECTION_NAME,
                points=[
                    {
                        "id": str(uuid4()),
                        "vector": embedding,
                        "payload": chunk,
                    }
                ],
            )
        except Exception as e:
            logger.error(f"Embedding error: {e}")
        finally:
            embedding_queue.task_done()
