import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Path
from fastapi.responses import FileResponse

from config.config import MAX_FILE_SIZE, UPLOAD_DIR
from helpers.file_serve import get_safe_filename, get_safe_filepath, validate_file
import aiofiles


router = APIRouter(tags=["File Serving"], prefix="/api/serve")


@router.post("/upload/{meeting_id}")
async def upload_file(meeting_id: str = Path(...), file: UploadFile = File(...)):
    """Upload a file for a specific meeting"""
    # Validate file type
    if not validate_file(file.filename):
        raise HTTPException(400, "Only .docx files are allowed")

    # Validate content type
    if file.content_type not in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]:
        raise HTTPException(400, "Invalid content type")

    # Create meeting-specific directory
    meeting_dir = UPLOAD_DIR / meeting_id
    meeting_dir.mkdir(parents=True, exist_ok=True)

    # Generate safe filename
    safe_filename = get_safe_filename(file.filename)
    filepath = meeting_dir / safe_filename

    # Save file with size limit check
    file_size = 0
    try:
        async with aiofiles.open(filepath, "wb") as f:
            while chunk := await file.read(8192):  # Read in 8KB chunks
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    await f.close()
                    filepath.unlink()  # Delete the file
                    raise HTTPException(413, "File too large")
                await f.write(chunk)
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


@router.get("/download/{meeting_id}/{file_id}")
async def download_file(meeting_id: str = Path(...), file_id: str = Path(...)):
    """Download a file from a specific meeting"""
    # Get and validate filepath with meeting_id
    filepath = get_safe_filepath(file_id, meeting_id=meeting_id)
    if not filepath:
        raise HTTPException(404, "File not found")

    # Serve file with appropriate headers
    return FileResponse(
        path=filepath,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=file_id,
        headers={
            "Content-Disposition": f"attachment; filename={file_id}",
            "X-Content-Type-Options": "nosniff",  # Prevent MIME sniffing
            "Cache-Control": "private, max-age=3600",  # Cache for 1 hour
        },
    )


@router.delete("/delete/{meeting_id}/{file_id}")
async def delete_file(meeting_id: str = Path(...), file_id: str = Path(...)):
    """Delete a file from a specific meeting"""
    filepath = get_safe_filepath(file_id, meeting_id=meeting_id)
    if not filepath:
        raise HTTPException(404, "File not found")

    try:
        filepath.unlink()
        return {"message": "File deleted successfully", "meeting_id": meeting_id}
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {str(e)}")


@router.get("/list/{meeting_id}")
async def list_meeting_files(meeting_id: str = Path(...)):
    """List all files for a specific meeting"""
    meeting_dir = UPLOAD_DIR / meeting_id

    if not meeting_dir.exists() or not meeting_dir.is_dir():
        return {"meeting_id": meeting_id, "files": []}

    files = []
    for filepath in meeting_dir.iterdir():
        if filepath.is_file() and filepath.suffix == ".docx":
            files.append(
                {
                    "file_id": filepath.name,
                    "size": filepath.stat().st_size,
                    "created_at": datetime.datetime.fromtimestamp(
                        filepath.stat().st_ctime
                    ),
                }
            )

    return {"meeting_id": meeting_id, "files": files}


# @router.delete("/delete-all/{meeting_id}")
# async def delete_all_meeting_files(meeting_id: str = Path(...)):
#     """Delete all files for a specific meeting"""
#     meeting_dir = UPLOAD_DIR / meeting_id

#     if not meeting_dir.exists() or not meeting_dir.is_dir():
#         raise HTTPException(404, "Meeting directory not found")

#     try:
#         deleted_count = 0
#         for filepath in meeting_dir.iterdir():
#             if filepath.is_file():
#                 filepath.unlink()
#                 deleted_count += 1

#         # Remove the directory if empty
#         if not any(meeting_dir.iterdir()):
#             meeting_dir.rmdir()

#         return {
#             "message": "All meeting files deleted successfully",
#             "meeting_id": meeting_id,
#             "deleted_count": deleted_count,
#         }
#     except Exception as e:
#         raise HTTPException(500, f"Delete failed: {str(e)}")
