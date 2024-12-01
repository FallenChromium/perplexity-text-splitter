from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import yake
import torch
from models import Summary, TextChunk
from services.retriever.embedder import BaseEmbedder
from pydantic import BaseModel
from config import get_session
from sqlmodel import select, col
import sqlalchemy as sa
from sqlalchemy import func, cast, ARRAY, String
from sqlalchemy.dialects.postgresql import array

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
        # TODO: actual confidence score
        return [(result.content, 1.0) for result in results]
    

class HypotheticalDocumentRetriever(BaseRetriever):
    def __init__(self, model, tokenizer, embedder: BaseEmbedder):
        self.model = model
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def retrieve(self, settings: RetrieveRequest) -> List[Tuple[str, float]]:
        # Get query embedding
        query_embedding = self.embedder.embed(settings.query)
        
        session = next(get_session())
        # Query summaries first, then get associated chunks
        summary_query = select(Summary)
        if settings.document_whitelist:
            summary_query = summary_query.filter(col(Summary.document_id).in_(settings.document_whitelist))
                
        # Generate hypothetical answer for query
        query_prompt = f"Generate a detailed answer (using the language of the question) to: {settings.query}"
        inputs = self.tokenizer(query_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=768,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        hypothetical_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Get hypothetical answer embedding
        hyp_embedding = self.embedder.embed(hypothetical_answer)
        
        # Calculate similarity scores
        results = []
        
        # Get summaries ordered by distance to query embedding
        query_summaries = session.exec(
            summary_query.order_by(Summary.embedding.l2_distance(query_embedding)).limit(settings.top_k)
        ).all()

        # Get summaries ordered by distance to hypothetical answer embedding
        hyp_summaries = session.exec(
            summary_query.order_by(Summary.embedding.l2_distance(hyp_embedding)).limit(settings.top_k)
        ).all()

        # Get all chunks referenced by these summaries
        all_chunk_ids = set()
        for summary in query_summaries + hyp_summaries:
            all_chunk_ids.update(summary.chunk_ids)

        # Get all chunks in one query
        chunks_query = select(TextChunk).filter(TextChunk.id.in_(all_chunk_ids))
        chunks = {chunk.id: chunk for chunk in session.exec(chunks_query).all()}


        # TODO: scoring
        # the id sorting is mimicking the narration-based-sorting, should be refactored in asynchronous / multi-threaded system
        for chunk in sorted(chunks.values(), key=lambda x: x.id):
            results.append((chunk.content, 1))
            
        # Sort by score descending and return top k
        return results[:settings.top_k]
    

        
class KeywordRetriever(BaseRetriever):
    def retrieve(self, settings: RetrieveRequest) -> List[Tuple[str, float]]:
        session = next(get_session())
        # Get summaries with matching keywords
        # Convert query to keywords using same YAKE extractor as in chunker
        kw_extractor = yake.KeywordExtractor(
            lan="ru",  # TODO: Use language detection
            n=2,
            dedupLim=0.9,
            top=10,
            features=None
        )
        query_keywords = [kw[0].lower() for kw in kw_extractor.extract_keywords(settings.query)]
        # TODO: couldn't figure out how to make this query work on the DB side rather than python, SQLModel is very limited
        # First get all summaries with any keyword overlap
        keyword_summaries = session.exec(
            select(Summary).where(
                Summary.keywords.op('&&')(query_keywords)
            )
        ).all()
        
        # Then sort them by overlap ratio in Python
        scored_summaries = []
        for summary in keyword_summaries:
            matching_keywords = len(set(k.lower() for k in summary.keywords) & set(query_keywords))
            score = matching_keywords / len(summary.keywords) if summary.keywords else 0
            scored_summaries.append((summary, score))
        
        # Sort by score and take top k
        keyword_summaries = [s[0] for s in sorted(scored_summaries, key=lambda x: x[1], reverse=True)[:settings.top_k]]

        # Get chunks referenced by keyword-matching summaries
        keyword_chunk_ids = set()
        for summary in keyword_summaries:
            keyword_chunk_ids.update(summary.chunk_ids)
            
        # Get all chunks in one query
        keyword_chunks_query = select(TextChunk).filter(TextChunk.id.in_(keyword_chunk_ids))
        keyword_chunks = {chunk.id: chunk for chunk in session.exec(keyword_chunks_query).all()}

        # Add keyword-based results with a fixed score
        keyword_results = []
        for chunk in sorted(keyword_chunks.values(), key=lambda x: x.id):
            keyword_results.append((chunk.content, 0.8))  # Fixed score for keyword matches

        # Combine with existing results, keeping highest score for duplicates
        seen_chunks = set()
        final_results = []
        
        for content, score in keyword_results:
            if content not in seen_chunks:
                final_results.append((content, score))
                seen_chunks.add(content)
                
        # Sort by score descending and return top k
        return sorted(final_results, key=lambda x: x[1], reverse=True)[:settings.top_k]
