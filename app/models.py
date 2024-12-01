from typing import List, Optional
from sqlalchemy import Column, JSON, ARRAY, Integer, String, Index
from sqlmodel import SQLModel, Field, Relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
from enum import Enum

class ContentType(str, Enum):
    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    IMAGE = "image"

class Document(SQLModel, table=True):
    __tablename__ = "documents"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str = Field(index=True)
    s3_key: str = Field(unique=True)
    mime_type: str
    content: Optional[str] = Field(default=None)
    content_type: ContentType = Field(default=ContentType.TEXT)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    doc_metadata: dict = Field(sa_column=Column(JSON), default={})
    

class ChunkType(str, Enum):
    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    IMAGE = "image"

class TextChunk(SQLModel, table=True):
    __tablename__ = "text_chunks"
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id")
    content: str
    chunk_type: ChunkType = Field(default=ChunkType.TEXT)
    start_pos: int  # Position in original document
    end_pos: int
    embedding: List[float] = Field(default=None, sa_column=Column(Vector(1024))) #TODO: actually depends on the embedding model, should be configurable       
    # Metadata fields
    related_chunk_ids: List[int] = Field(sa_column=Column(ARRAY(Integer)), default=[])
    chunk_metadata: dict = Field(sa_column=Column(JSON), default={})

    class Config:
        arbitrary_types_allowed = True

class Summary(SQLModel, table=True):
    __tablename__ = "summaries"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id")
    content: str  # The summary text
    keywords: List[str] = Field(sa_column=Column(ARRAY(String)))
    embedding: List[float] = Field(sa_column=Column(Vector(1024)))
    chunk_ids: List[int] = Field(sa_column=Column(ARRAY(Integer)))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    summary_metadata: dict = Field(sa_column=Column(JSON), default={})
    __table_args__ = (
        Index(
            'idx_summaries_keywords',
            'keywords',
            postgresql_using='gin'
        ),
    )
