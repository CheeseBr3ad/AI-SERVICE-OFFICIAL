import asyncio
from datetime import datetime
import os
import tempfile
from uuid import UUID, uuid4
from fastapi import (
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    APIRouter,
)
import json
from config.qdrant import qdrant
from config.config import QDRANT_DOCUMENT_COLLECTION_NAME
from config.models import get_model
from helpers.document import chunk_text, extract_text_from_docx
from qdrant_client.http import models as qmodels
from helpers.embedding import embedding_queue, embedding_queue_chat
from config.logger import logger
from helpers.file_serve import get_safe_filename
from helpers.store_file import handle_file_upload

router = APIRouter(tags=["Embedding"], prefix="/api/embedding")


@router.websocket("/ws/meetings/{meeting_id}")
async def meeting_ws(websocket: WebSocket, meeting_id: str):
    """sends
    block_id,
    text -block of text in a json format (stringify)
    to embedding queue"""

    """
    websocket message format example:
{
 "type": "transcript" | "chat",
  "meeting_id": "a3f4b9e1-7c8d-42f1-92a3-bb1c6e2b7d29",
  "block_id": 1,
  "text": {
    "speaker1": {
      "text": "Good morning everyone, let's begin our discussion.",
      "timestamp": "2025-10-18T19:00:05Z",
      "meeting_time": 5
    },
    "speaker2": {
      "text": "Morning! Ready when you are.",
      "timestamp": "2025-10-18T19:00:10Z",
      "meeting_time": 10
    }
  }
}
    """

    await websocket.accept()
    while True:
        try:
            message = await websocket.receive_text()
            data = json.loads(message)
            data["meeting_id"] = meeting_id
            if data.get("timestamp") is None:
                data["timestamp"] = datetime.utcnow().isoformat()
            if data.get("type", "transcript") == "chat":
                await embedding_queue_chat.put(data)
            else:  # transcript
                await embedding_queue.put(data)
            await websocket.send_json(
                {
                    "type": data.get("type", "transcript"),
                    "status": "queued",
                    "block_id": data.get("block_id"),
                    "meeting_id": meeting_id,
                }
            )
        except Exception as e:
            print(f"WebSocket closed: {e}")
            break


@router.post("/document")
async def upload_doc(
    file: UploadFile = File(...),
    meeting_id: UUID = Form(...),
    background_tasks: BackgroundTasks = None,
):
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported.")

    # Read file content once at the beginning
    content = await file.read()

    # Step 1: Upload to server storage
    # Reset file pointer for upload
    await file.seek(0)

    # Fire off upload in the background
    asyncio.create_task(handle_file_upload(meeting_id=str(meeting_id), file=file))

    # Step 2: Process for RAG embedding
    tmp_path = None
    try:
        # Create temp file and ensure it's written before processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            # Explicitly flush to ensure all data is written
            tmp.flush()

        # Now the file is closed but still exists on disk
        # Extract text and chunk
        text = extract_text_from_docx(tmp_path)
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(
                status_code=400, detail="No readable text found in document."
            )

        # Get embeddings
        model = get_model()
        loop = asyncio.get_running_loop()
        text_only = [chunk["text"] for chunk in chunks]
        logger.info(f"Generating embeddings for {len(text_only)} chunks...")
        logger.info(f"Sample chunk text: {text_only[0][:100]}...")
        embeddings = await loop.run_in_executor(None, model.encode, text_only)

        # Create payloads with metadata
        timestamp = datetime.utcnow().isoformat()
        points = [
            qmodels.PointStruct(
                id=str(uuid4()),
                vector=embeddings[i],
                payload={
                    "meeting_id": str(meeting_id),
                    "file_name": file.filename,
                    "chunk_index": i + 1,
                    "text": chunks[i]["text"],
                    "timestamp": timestamp,
                },
            )
            for i in range(len(chunks))
        ]

        # Batch insert for performance
        BATCH_SIZE = 50
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i : i + BATCH_SIZE]
            await loop.run_in_executor(
                None, qdrant.upsert, QDRANT_DOCUMENT_COLLECTION_NAME, batch
            )

        return {
            "status": "success",
            "collection": QDRANT_DOCUMENT_COLLECTION_NAME,
            "meeting_id": str(meeting_id),
            "chunks_stored": len(points),
            "file_name": file.filename,
            "file_id": get_safe_filename(file.filename),
        }

    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")
