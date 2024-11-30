from typing import List, Optional
from sqlalchemy import Column, JSON, ARRAY, Integer
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
    content_type: ContentType = Field(default=ContentType.TEXT)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    chunk_metadata: dict = Field(sa_column=Column(JSON), default={})
    
    # Relationships
    chunks: List["TextChunk"] = Relationship(back_populates="document")

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
    embedding: List[float] = Field(default=None, sa_column=Column(Vector(1536)))       
    # Metadata fields
    related_chunk_ids: List[int] = Field(sa_column=Column(ARRAY(Integer)), default=[])
    chunk_metadata: dict = Field(sa_column=Column(JSON), default={})
    
    # Relationships
    document: Document = Relationship(back_populates="chunks")

    class Config:
        arbitrary_types_allowed = True