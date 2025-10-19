from typing import List, Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from config.models import get_model
from schemas import SearchFilters, SearchResult
from config.qdrant import qdrant
from config.logger import logger


def build_qdrant_filter(
    filters: Optional[SearchFilters], collection_type: str
) -> Optional[Filter]:
    """Build Qdrant filter based on collection type and provided filters"""
    if not filters:
        return None

    conditions = []

    # Common filters
    if filters.meeting_id:
        conditions.append(
            FieldCondition(key="meeting_id", match=MatchValue(value=filters.meeting_id))
        )

    if filters.start_timestamp or filters.end_timestamp:
        # Note: Timestamp filtering would need custom logic based on your format
        if filters.start_timestamp:
            conditions.append(
                FieldCondition(
                    key="timestamp", match=MatchValue(value=filters.start_timestamp)
                )
            )

    # Document collection specific filters
    if collection_type == "documents":
        if filters.file_name:
            conditions.append(
                FieldCondition(
                    key="file_name", match=MatchValue(value=filters.file_name)
                )
            )
        if filters.chunk_index_min is not None or filters.chunk_index_max is not None:
            conditions.append(
                FieldCondition(
                    key="chunk_index",
                    range=Range(
                        gte=filters.chunk_index_min, lte=filters.chunk_index_max
                    ),
                )
            )

    # Transcript collection specific filters
    if collection_type == "transcripts":
        if filters.block_id_min is not None or filters.block_id_max is not None:
            conditions.append(
                FieldCondition(
                    key="block_id",
                    range=Range(gte=filters.block_id_min, lte=filters.block_id_max),
                )
            )

    # chat collection specific filters
    if collection_type == "chat":
        if filters.block_id_min is not None or filters.block_id_max is not None:
            conditions.append(
                FieldCondition(
                    key="block_id",
                    range=Range(gte=filters.block_id_min, lte=filters.block_id_max),
                )
            )

    return Filter(must=conditions) if conditions else None


async def search_collection(
    collection_name: str,
    query_vector: List[float],
    filters: Optional[Filter],
    top_k: int,
    collection_type: str,
) -> List[SearchResult]:
    """Search a single Qdrant collection"""
    try:
        results = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=filters,
            limit=top_k,
            with_payload=True,
        )

        search_results = []
        for result in results:
            payload = result.payload

            # Extract text content based on collection type
            if collection_type == "transcripts":
                # Parse speaker format
                content_parts = []
                for speaker, data in payload.items():
                    if isinstance(data, dict) and "text" in data:
                        content_parts.append(f"{speaker}: {data['text']}")
                content = "\n".join(content_parts)
            else:
                # Document collection - direct text
                content = payload.get("text", str(payload))

            search_results.append(
                SearchResult(
                    collection=collection_name,
                    score=result.score,
                    content=content[:500],  # Truncate for response
                    metadata=payload,
                    meeting_id=payload.get("meeting_id"),
                    timestamp=payload.get("timestamp"),
                )
            )

        return search_results
    except Exception as e:
        print(f"Error searching {collection_name}: {e}")
        return []


def build_rag_prompt(query: str, context_results: List[SearchResult]) -> str:
    """Build comprehensive RAG prompt for Gemini"""

    # Separate results by collection
    transcript_results = [
        r for r in context_results if "transcript" in r.collection.lower()
    ]
    document_results = [
        r for r in context_results if "document" in r.collection.lower()
    ]

    chat_results = [r for r in context_results if "chat" in r.collection.lower()]

    prompt = f"""You are Bridge AI, an intelligent assistant for Bridge - a real-time, multi-modal communication platform. Bridge enables communities to coordinate over audio/video/text with live transcription & translation, gesture cues, and a real-time RAG copilot. Every session becomes a searchable "Meeting Doc" with jump-to-audio timestamps and instant replay.

**Your Role:**
- Answer questions grounded in meeting transcripts and attached documents
- Provide precise, contextual answers when available
- Help users navigate and understand their meeting content
- Synthesize information across multiple sources (conversations + documents + chats)

**User Query:** {query}

**Available Context:**

"""
    # Add chat context
    if chat_results:
        prompt += "**Chat Messages:**\n\n"
        for result in chat_results:
            prompt += f"{result.content}\n\n"

    # Add transcript context
    if transcript_results:
        prompt += "**Meeting Transcripts:**\n\n"
        for result in transcript_results:
            prompt += f"{result.content}\n\n"

    # Add document context
    if document_results:
        prompt += "**Attached Documents:**\n\n"
        for result in document_results:
            prompt += f"{result.content}\n\n"

    prompt += """**Instructions:**
1. Answer the query directly and concisely based on the provided context
2. Synthesize information from multiple sources coherently
3. If the context doesn't contain enough information to answer fully, say so clearly
4. Use a professional but conversational tone appropriate for a team collaboration tool

**Your Answer:**"""

    logger.info(f"✓ RAG prompt constructed with {len(context_results)} context pieces")

    return prompt


def create_embedding(text: str) -> List[float]:
    """Create embedding using Qwen Sentence Transformer"""
    _model = get_model()
    embedding = _model.encode(text, convert_to_numpy=True)
    return embedding.tolist()
