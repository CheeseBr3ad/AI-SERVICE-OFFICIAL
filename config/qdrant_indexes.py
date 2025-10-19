from qdrant_client.http import models as qmodels
from config.qdrant import qdrant
from config.config import (
    QDRANT_DOCUMENT_COLLECTION_NAME,
    QDRANT_MEETING_TRANSCRIPTS_COLLECTION_NAME,
)


def add_indexes_to_existing_document_collection():
    """Add indexes to an already created and populated collection"""

    indexes = [
        ("meeting_id", qmodels.PayloadSchemaType.KEYWORD),
        ("file_name", qmodels.PayloadSchemaType.KEYWORD),
        ("chunk_index", qmodels.PayloadSchemaType.INTEGER),
        ("timestamp", qmodels.PayloadSchemaType.KEYWORD),
    ]

    for field_name, schema_type in indexes:
        try:
            qdrant.create_payload_index(
                collection_name=QDRANT_DOCUMENT_COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema_type,
            )
            print(f"✓ Created index for {field_name}")
        except Exception as e:
            # Index might already exist
            print(f"⚠ {field_name}: {e}")

    print("\n✅ Finished adding indexes for document collection!")


def create_indexes_for_transcript_collection():
    """Create indexes for all payload fields in the transcript collection"""

    indexes = [
        ("meeting_id", qmodels.PayloadSchemaType.KEYWORD),
        ("block_id", qmodels.PayloadSchemaType.INTEGER),
        ("timestamp", qmodels.PayloadSchemaType.KEYWORD),
    ]

    for field_name, schema_type in indexes:
        try:
            qdrant.create_payload_index(
                collection_name=QDRANT_MEETING_TRANSCRIPTS_COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema_type,
            )
            print(f"✓ Created index for {field_name}")
        except Exception as e:
            # Index might already exist
            print(f"⚠ {field_name}: {e}")

    print("\n✅ Finished adding indexes for transcript collection!")
