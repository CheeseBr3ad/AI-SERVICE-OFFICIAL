import asyncio
from datetime import datetime
from fastapi import APIRouter, HTTPException

from config.config import (
    QDRANT_DOCUMENT_COLLECTION_NAME,
    QDRANT_MEETING_TRANSCRIPTS_COLLECTION_NAME,
)
from helpers.rag import (
    build_qdrant_filter,
    build_rag_prompt,
    create_embedding,
    search_collection,
)
from schemas import RAGResponse, SearchRequest
from config.genai import gemini_model
from config.logger import logger

router = APIRouter(tags=["RAG"], prefix="/api/rag")


@router.post("/search", response_model=RAGResponse)
async def rag_search(request: SearchRequest):
    """
    Perform RAG search across both meeting transcripts and document collections.
    Queries both collections in parallel and uses Gemini to generate a contextual answer.
    """
    start_time = datetime.now()
    logger.info(
        f"Starting RAG search for query: '{request.query}' with filters: {request.filters}"
    )

    try:
        # Step 1: Create query embedding
        step_start = datetime.now()
        query_embedding = create_embedding(request.query)
        embedding_time = (datetime.now() - step_start).total_seconds() * 1000
        logger.info(
            f"✓ Query embedding created in {embedding_time:.2f}ms (dimension: {len(query_embedding)})"
        )

        # Step 2: Build filters for both collections
        step_start = datetime.now()
        doc_filter = build_qdrant_filter(request.filters, "documents")
        transcript_filter = build_qdrant_filter(request.filters, "transcripts")
        filter_time = (datetime.now() - step_start).total_seconds() * 1000
        logger.info(f"✓ Filters built in {filter_time:.2f}ms")
        logger.debug(f"Document filter: {doc_filter}")
        logger.debug(f"Transcript filter: {transcript_filter}")

        # Step 3: Search both collections in parallel
        step_start = datetime.now()
        doc_task = search_collection(
            QDRANT_DOCUMENT_COLLECTION_NAME,
            query_embedding,
            doc_filter,
            request.top_k,
            "documents",
        )

        transcript_task = search_collection(
            QDRANT_MEETING_TRANSCRIPTS_COLLECTION_NAME,
            query_embedding,
            transcript_filter,
            request.top_k,
            "transcripts",
        )

        doc_results, transcript_results = await asyncio.gather(
            doc_task, transcript_task
        )
        search_time = (datetime.now() - step_start).total_seconds() * 1000
        logger.info(f"✓ Parallel search completed in {search_time:.2f}ms")
        logger.info(f"  - Documents found: {len(doc_results)} (top_k={request.top_k})")
        logger.info(
            f"  - Transcripts found: {len(transcript_results)} (top_k={request.top_k})"
        )

        # Step 4: Combine and sort results by score
        step_start = datetime.now()
        all_results = doc_results + transcript_results
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Take top results across both collections
        top_results = all_results[: request.top_k * 2]
        merge_time = (datetime.now() - step_start).total_seconds() * 1000
        logger.info(f"✓ Results merged and sorted in {merge_time:.2f}ms")
        logger.info(
            f"  - Total results: {len(all_results)}, Top results selected: {len(top_results)}"
        )

        if top_results:
            logger.debug(f"Top 3 scores: {[f'{r.score:.4f}' for r in top_results[:3]]}")

        # Step 5: Generate answer using Gemini
        step_start = datetime.now()
        if top_results:
            rag_prompt = build_rag_prompt(request.query, top_results)
            logger.debug(f"RAG prompt length: {len(rag_prompt)} characters")

            response = gemini_model.generate_content(
                rag_prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                },
            )

            answer = response.text
            gemini_time = (datetime.now() - step_start).total_seconds() * 1000
            logger.info(f"✓ Gemini answer generated in {gemini_time:.2f}ms")
            logger.info(f"  - Answer length: {len(answer)} characters")
        else:
            answer = "I couldn't find any relevant information in the meeting transcripts or documents to answer your query. Try rephrasing your question or checking if the meeting has been processed."
            gemini_time = 0
            logger.warning("No results found for query")

        # Step 6: Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000

        # Step 7: Build response
        sources = top_results if request.include_sources else []

        logger.info(f"✓ RAG search completed successfully in {processing_time:.2f}ms")
        logger.info(
            f"Time breakdown: Embedding={embedding_time:.2f}ms, Filter={filter_time:.2f}ms, "
            f"Search={search_time:.2f}ms, Merge={merge_time:.2f}ms, Gemini={gemini_time:.2f}ms"
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=request.query,
            total_results=len(all_results),
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        error_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(
            f"✗ RAG search failed after {error_time:.2f}ms: {str(e)}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"RAG search failed: {str(e)}")


# @router.post("/search", response_model=RAGResponse)
# async def rag_search(request: SearchRequest):
#     """
#     Perform RAG search across both meeting transcripts and document collections.
#     Queries both collections in parallel and uses Gemini to generate a contextual answer.
#     """
#     start_time = datetime.now()

#     try:
#         # Step 1: Create query embedding
#         query_embedding = create_embedding(request.query)

#         # Step 2: Build filters for both collections
#         doc_filter = build_qdrant_filter(request.filters, "documents")
#         transcript_filter = build_qdrant_filter(request.filters, "transcripts")

#         # Step 3: Search both collections in parallel
#         doc_task = search_collection(
#             QDRANT_DOCUMENT_COLLECTION_NAME,
#             query_embedding,
#             doc_filter,
#             request.top_k,
#             "documents",
#         )

#         transcript_task = search_collection(
#             QDRANT_MEETING_TRANSCRIPTS_COLLECTION_NAME,
#             query_embedding,
#             transcript_filter,
#             request.top_k,
#             "transcripts",
#         )

#         doc_results, transcript_results = await asyncio.gather(
#             doc_task, transcript_task
#         )

#         # Step 4: Combine and sort results by score
#         all_results = doc_results + transcript_results
#         all_results.sort(key=lambda x: x.score, reverse=True)

#         # Take top results across both collections
#         top_results = all_results[: request.top_k * 2]

#         # Step 5: Generate answer using Gemini
#         if top_results:
#             rag_prompt = build_rag_prompt(request.query, top_results)

#             response = gemini_model.generate_content(
#                 rag_prompt,
#                 generation_config={
#                     "temperature": 0.7,
#                     "top_p": 0.95,
#                     "top_k": 40,
#                     "max_output_tokens": 2048,
#                 },
#             )

#             answer = response.text
#         else:
#             answer = "I couldn't find any relevant information in the meeting transcripts or documents to answer your query. Try rephrasing your question or checking if the meeting has been processed."

#         # Step 6: Calculate processing time
#         end_time = datetime.now()
#         processing_time = (end_time - start_time).total_seconds() * 1000

#         # Step 7: Build response
#         sources = top_results if request.include_sources else []

#         return RAGResponse(
#             answer=answer,
#             sources=sources,
#             query=request.query,
#             total_results=len(all_results),
#             processing_time_ms=round(processing_time, 2),
#         )

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"RAG search failed: {str(e)}")
