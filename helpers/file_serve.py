from pathlib import Path as PathLib
import re
from typing import Optional

from config.config import ALLOWED_EXTENSIONS, UPLOAD_DIR


# Security: Validate file extension
def validate_file(filename: str) -> bool:
    return PathLib(filename).suffix.lower() in ALLOWED_EXTENSIONS


def slugify(text: str) -> str:
    # Convert to lowercase, replace spaces and non-alphanumeric chars with hyphens
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text


def get_safe_filename(original_filename: str) -> str:
    ext = PathLib(original_filename).suffix.lower()
    name = PathLib(original_filename).stem
    safe_name = slugify(name)
    return f"{safe_name}{ext}"


# Security: Validate file path to prevent directory traversal
def get_safe_filepath(file_id: str, meeting_id: str = None):
    """Get safe filepath, optionally within a meeting directory"""
    safe_filename = get_safe_filename(file_id)

    if meeting_id:
        # Sanitize meeting_id to prevent directory traversal
        safe_meeting_id = "".join(c for c in meeting_id if c.isalnum() or c in "-_")
        filepath = UPLOAD_DIR / safe_meeting_id / safe_filename
    else:
        filepath = UPLOAD_DIR / safe_filename

    if not filepath.exists() or not filepath.is_file():
        return None

    return filepath
