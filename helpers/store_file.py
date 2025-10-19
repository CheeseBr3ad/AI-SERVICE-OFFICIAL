import aiofiles
from fastapi import File, HTTPException, UploadFile
from pathlib import Path

from config.config import MAX_FILE_SIZE, UPLOAD_DIR
from helpers.file_serve import get_safe_filename, validate_file
from config.logger import logger


async def handle_file_upload(
    meeting_id: str,
    file: UploadFile,
):
    """Helper function to handle file upload logic."""

    # ✅ Validate file type
    if not validate_file(file.filename):
        raise HTTPException(400, "Only .docx files are allowed")

    # ✅ Validate content type
    if file.content_type not in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]:
        raise HTTPException(400, "Invalid content type")

    # ✅ Create meeting-specific directory
    meeting_dir = UPLOAD_DIR / meeting_id
    meeting_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Generate safe filename and file path
    safe_filename = get_safe_filename(file.filename)
    filepath = meeting_dir / safe_filename

    # ✅ Save file with size limit check
    file_size = 0
    try:
        async with aiofiles.open(filepath, "wb") as f:
            while chunk := await file.read(8192):  # Read in 8KB chunks
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    await f.close()
                    filepath.unlink()  # Delete partially uploaded file
                    raise HTTPException(413, "File too large")
                await f.write(chunk)
        logger.info(f"File uploaded: {filepath}")
    except Exception as e:
        if filepath.exists():
            filepath.unlink()
        raise HTTPException(500, f"Upload failed: {str(e)}")

    return {
        "file_id": safe_filename,
        "meeting_id": meeting_id,
        "original_filename": file.filename,
        "size": file_size,
    }
