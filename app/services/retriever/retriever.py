from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from models import TextChunk
from services.retriever.embedder import BaseEmbedder
from pydantic import BaseModel
from config import get_session
from sqlmodel import select, col

class RetrieveRequest(BaseModel):
    query: str
    document_whitelist: Optional[List[int]] = None
    top_k: int = 10

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, settings: RetrieveRequest) -> List[Tuple[str, float]]:
        pass

class SentenceTransformerRetriever(BaseRetriever):
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
    def retrieve(self, settings: RetrieveRequest) -> List[Tuple[str, float]]:
        query_embedding = self.embedder.embed(settings.query)
        sql_query = select(TextChunk)
        session = next(get_session())
        chunks = []
        if settings.document_whitelist:
            sql_query = sql_query.filter(col(TextChunk.document_id).in_(settings.document_whitelist))
        sql_query = sql_query.order_by(TextChunk.embedding.l2_distance(query_embedding)).limit(settings.top_k)
        results = session.exec(sql_query).all()
        return [(result.content, 1.0) for result in results]