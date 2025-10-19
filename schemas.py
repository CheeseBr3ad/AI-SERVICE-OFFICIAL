from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime


class TimeRange(BaseModel):
    """Time range for filtering results"""

    start: Optional[datetime] = Field(None, description="Start timestamp (ISO format)")
    end: Optional[datetime] = Field(None, description="End timestamp (ISO format)")
    relative: Optional[str] = Field(
        None,
        description="Relative time like 'last_hour', 'last_day', 'last_week', 'last_month'",
    )


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    meeting_id: Optional[str] = Field(None, description="Filter by specific meeting ID")
    search_type: Literal["both", "documents", "transcripts"] = Field(
        "both", description="Search in documents, transcripts, or both"
    )
    time_range: Optional[TimeRange] = Field(None, description="Filter by time range")
    limit: int = Field(5, ge=1, le=20, description="Number of results per collection")
    score_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Minimum similarity score"
    )


class SearchResult(BaseModel):
    source: Literal["document", "transcript"]
    meeting_id: str
    text: str
    score: float
    timestamp: str
    metadata: dict


class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResult]
    total_results: int


class SearchFilters(BaseModel):
    meeting_id: Optional[str] = None
    file_name: Optional[str] = None
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    chunk_index_min: Optional[int] = None
    chunk_index_max: Optional[int] = None
    block_id_min: Optional[int] = None
    block_id_max: Optional[int] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query")
    filters: Optional[SearchFilters] = None
    top_k: int = Field(
        default=5, ge=1, le=50, description="Number of results per collection"
    )
    include_sources: bool = Field(
        default=True, description="Include source documents in response"
    )


class SearchResult(BaseModel):
    collection: str
    score: float
    content: str
    metadata: Dict[str, Any]
    meeting_id: Optional[str] = None
    timestamp: Optional[str] = None


class RAGResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
    query: str
    total_results: int
    processing_time_ms: float
