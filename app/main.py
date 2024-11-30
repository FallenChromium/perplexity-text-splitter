from fastapi import FastAPI, UploadFile, HTTPException, Depends
from sqlmodel import Session, SQLModel, create_engine

from config import POSTGRES_USER, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD
from models import Document, TextChunk
from services.document_processor import DocumentPipeline, S3StorageBackend, PlainTextParser
from services.chunker import PerplexityBasedChunker
import os
from typing import List

app = FastAPI(title="Text Splitter API")
# Read environment variables from .env file


# Construct the database connection string
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}"

# Database configuration
engine = create_engine(DATABASE_URL)

# Create tables
SQLModel.metadata.create_all(engine)

# Initialize document processor components
storage_backend = S3StorageBackend()
markdown_parser = PlainTextParser()
text_chunker = PerplexityBasedChunker()

# Create document processor
doc_processor = DocumentPipeline(
    storage=storage_backend,
    chunker=text_chunker,
    parsers=markdown_parser,
)

def get_session():
    with Session(engine) as session:
        yield session

@app.post("/documents/", response_model=int)
async def upload_document(
    file: UploadFile,
    session: Session = Depends(get_session)
):
    """Upload a document, process it, and store chunks"""
    try:
        # Process document
        document = await doc_processor.process_document(
            file.file,
            file.filename,
            file.content_type or "text/plain"
        )
        
        # Store in database
        session.add(document)
        session.commit()
        session.refresh(document)
        
        return document.id
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}", response_model=Document)
async def get_document(
    document_id: int,
    session: Session = Depends(get_session)
):
    """Get document with its chunks"""
    document = session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@app.post("/documents/{document_id}/process", response_model=bool)
async def process_document(
    document_id: int,
    session: Session = Depends(get_session)
):
    """Process a document and store chunks"""
    document = session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    # Download document from storage
    session.get(Document, document_id)

    document = await doc_processor.process_document(document)

    # Store in database
    session.add(document)
    session.commit()
    session.refresh(document)

    # TODO: Handle errors
    return True

@app.get("/documents/{document_id}/chunks", response_model=List[TextChunk])
async def get_document_chunks(
    document_id: int,
    session: Session = Depends(get_session)
):
    """Get all chunks for a document"""
    document = session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document.chunks 
