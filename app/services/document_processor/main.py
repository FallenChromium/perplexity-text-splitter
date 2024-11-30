from typing import BinaryIO, List

from app.services.chunker.chunkers import BaseTextChunker
from app.services.document_processor.parsing import DocumentParser
from app.services.document_processor.preprocessing import DocumentPreprocessor
from app.services.document_processor.storage import StorageBackend
from ...models import Document, TextChunk, ContentType, ChunkType
from pathlib import Path




class DocumentPipeline:
    def __init__(
        self,
        storage: StorageBackend,
        chunker: BaseTextChunker,
        parsers: List[DocumentParser] | DocumentParser | None = None,
        preprocess: List[DocumentPreprocessor] | DocumentPreprocessor | None = None,
    ):
        self.storage = storage
        self.parser = parsers
        self.preprocess = preprocess
        self.chunker = chunker
    
    async def process_document(
        self,
        file: BinaryIO,
        filename: str,
        mime_type: str
    ) -> Document:
        """Process document and return Document model instance"""
        # Store original file
        s3_key = await self.storage.store_document(file, filename)
        
        # Create document instance
        document = Document(
            filename=filename,
            s3_key=s3_key,
            mime_type=mime_type,
            content_type=ContentType.TEXT,  # TODO: Detect content type
            metadata={}
        )
        
        # Read and parse content
        file.seek(0)
        content = file.read()
        if isinstance(content, str):
            content = content.encode('utf-8')
        
        # Convert to markdown
        # TODO: iterate on parsers
        markdown_text = self.parser.parse_to_markdown(content, mime_type)
        
        # Split into chunks
        chunks = self.chunker.split(markdown_text)
        # TODO: linked chunks
        # Create chunk instances
        document.chunks = []
        for chunk_text, start_pos, end_pos in chunks:
            chunk = TextChunk(
                content=chunk_text,
                chunk_type=ChunkType.TEXT,
                start_pos=start_pos,
                end_pos=end_pos,
                metadata={}
            )
            document.chunks.append(chunk)
        
        return document