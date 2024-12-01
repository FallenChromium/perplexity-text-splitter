from app.services.retriever.retriever import RetrieveRequest, SentenceTransformerRetriever
from fastapi import FastAPI, UploadFile, HTTPException, Depends
from sqlmodel import Session, select, col
from config import get_session

from config import POSTGRES_USER, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_PASSWORD
from models import Document, TextChunk
from services.document_processor import DocumentPipeline, S3StorageBackend, PlainTextParser
from services.chunker import PerplexityBasedChunker
from services.document_processor.parsing import PlainTextParser, HTMLParser
import os
from typing import List, Optional, Tuple

app = FastAPI(title="Text Splitter API")

# Initialize document processor components
storage_backend = S3StorageBackend()
markdown_parser = PlainTextParser()
text_chunker = PerplexityBasedChunker()

# Create document processor
doc_processor = DocumentPipeline(
    storage=storage_backend,
    chunker=text_chunker,
    parsers=[PlainTextParser(), HTMLParser()],
    retrievers=[SentenceTransformerRetriever()]
)



@app.post("/documents/", response_model=int)
async def upload_document(
    file: UploadFile,
    session: Session = Depends(get_session)
):
    """Upload a document, process it, and store chunks"""
    try:
        # Process document
        document = await doc_processor.save_document(
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

@app.get("/documents", response_model=Document)
async def get_document(
    document_id: int,
    session: Session = Depends(get_session)
):
    """Get document with its chunks"""
    documents = Document.query.all()
    if not documents:
        raise HTTPException(status_code=404, detail="Documents not found")
    return documents

@app.post("/documents/{document_id}/process", response_model=bool)
async def process_document(
    document_id: int,
    # TODO: processing settings object
    session: Session = Depends(get_session)
):
    """Process a document and store chunks"""
    document = session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    # Delete all existing text chunks associated with the document (no dupes)
    document, chunks = await doc_processor.process_document(document)
    existing_chunks = session.exec(select(TextChunk).filter(TextChunk.document_id == document.id)).all()
    for chunk in existing_chunks:
        session.delete(chunk)
    session.flush()

    for chunk in chunks:
        session.add(chunk)
    # Store in database
    session.add(document)
    session.flush()
    session.commit()

    return True

@app.get("/documents/{document_id}/chunks", response_model=List[Tuple[str, int, int]])
async def get_document_chunks(
    document_id: int,
    session: Session = Depends(get_session)
):
    """Get all chunks for a document"""
    sql_query = select(TextChunk).filter(col(TextChunk.document_id) == document_id)
    chunks = session.exec(sql_query).all()
    if not chunks:
        raise HTTPException(status_code=404, detail="Chunks not found")
    return [(chunk.content, chunk.start_pos, chunk.end_pos) for chunk in chunks]

@app.post("/retrieve", response_model=List[Tuple[str, float]])
async def get_relevant_chunks(
    settings: RetrieveRequest,
):
    results = await doc_processor.retrieve_chunks(settings)
    return results