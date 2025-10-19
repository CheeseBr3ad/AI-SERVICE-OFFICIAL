from docx import Document


def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    text = []
    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text.strip())
    return "\n".join(text)


# def chunk_text(text: str, max_tokens: int = 512):
#     words = text.split()
#     chunks, current_chunk = [], []
#     for word in words:
#         current_chunk.append(word)
#         if len(current_chunk) >= max_tokens:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = []
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
#     return chunks


def chunk_text(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    preserve_speaker_turns: bool = True,
):
    """
    Chunk meeting transcript optimized for RAG with Qdrant.

    Args:
        text: Full transcript
        max_tokens: Target chunk size (512 recommended for embeddings)
        overlap_tokens: Overlap between chunks for context continuity
        preserve_speaker_turns: Try to keep speaker turns intact
    """
    # Convert tokens to approximate word count (1 token ≈ 1.3 words)
    max_words = int(max_tokens * 0.75)
    overlap_words = int(overlap_tokens * 0.75)

    chunks = []
    lines = text.split("\n")

    current_chunk = []
    current_word_count = 0

    for line in lines:
        if not line.strip():
            continue

        line_words = line.split()
        line_word_count = len(line_words)

        # Check if adding this line would exceed max_words
        if current_word_count + line_word_count > max_words and current_chunk:
            # Save current chunk
            chunk_text = "\n".join(current_chunk)
            chunks.append(
                {
                    "text": chunk_text,
                    "word_count": current_word_count,
                    "approx_tokens": int(current_word_count / 0.75),
                }
            )

            # Create overlap: keep last few lines based on overlap_words
            overlap_lines = []
            overlap_count = 0
            for prev_line in reversed(current_chunk):
                overlap_count += len(prev_line.split())
                overlap_lines.insert(0, prev_line)
                if overlap_count >= overlap_words:
                    break

            current_chunk = overlap_lines
            current_word_count = overlap_count

        current_chunk.append(line)
        current_word_count += line_word_count

    # Add final chunk
    if current_chunk:
        chunk_text = "\n".join(current_chunk)
        chunks.append(
            {
                "text": chunk_text,
                "word_count": current_word_count,
                "approx_tokens": int(current_word_count / 0.75),
            }
        )

    return chunks
