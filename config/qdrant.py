from qdrant_client import QdrantClient
from config.config import QDRANT_URL, QDRANT_API_KEY


qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
