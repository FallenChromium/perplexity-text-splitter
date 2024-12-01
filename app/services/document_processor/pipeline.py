from typing import BinaryIO, List, Tuple

from services.retriever.retriever import BaseRetriever, RetrieveRequest
from services.retriever.embedder import SentenceTransformerEmbedder
from services.chunker.chunkers import BaseTextChunker
from services.document_processor.parsing import DocumentParser
from services.document_processor.preprocessing import DocumentPreprocessor
from services.document_processor.storage import StorageBackend
from services.retriever.embedder import BaseEmbedder
from models import Document, TextChunk, ContentType, ChunkType
from pathlib import Path

text_mimetypes = [
    "text/plain",
    "text/markdown",
    "text/html",
]


class DocumentPipeline:
    def __init__(
        self,
        storage: StorageBackend,
        chunker: BaseTextChunker,
        parsers: List[DocumentParser] | DocumentParser | None = None,
        preprocess: List[DocumentPreprocessor] | DocumentPreprocessor | None = None,
        retrievers: List[BaseRetriever] | BaseRetriever | None = None,
        embedder: BaseEmbedder = SentenceTransformerEmbedder(),
    ):
        self.storage = storage
        self.parsers: List[DocumentParser] | DocumentParser | None = parsers
        self.preprocess = preprocess
        self.chunker = chunker
        self.retrievers: BaseRetriever = retrievers
        self.embedder: BaseEmbedder = embedder

    async def save_document(
        self, file: BinaryIO, filename: str, mime_type: str
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
            doc_metadata={},
        )

        return document

    async def process_document(self, document: Document, embed_model: BaseEmbedder = None) -> Document:
        embedder = embed_model or self.embedder
        doc = document
        if not doc.content:
            doc_file: bytes = await self.storage.get_document(doc.s3_key)
            content = doc_file

            mime_type: str = doc.mime_type

            if isinstance(self.parsers, list):
                for parser in self.parsers:
                    if parser.can_handle(mime_type):
                        doc.content = parser.parse_to_markdown(content, mime_type)
                        break
                if not doc.content:
                    raise(Exception("We do not support parsing this file as of yet"))
            elif isinstance(self.parsers, DocumentParser) and self.parser.can_handle(mime_type):
                doc.content = self.parser.parse_to_markdown(content, mime_type)
            else:
                print("assuming plaintext")
                doc.content = content.decode("utf-8")

            if isinstance(self.preprocess, list):
                for preprocessor in self.preprocess:
                    preprocessor.preprocess(doc)
            elif isinstance(self.preprocess, DocumentPreprocessor):
                self.preprocess.preprocess(doc)
            else:
                pass

        markdown_text = doc.content
        # Split into chunks
        chunks = self.chunker.split(markdown_text)
        # TODO: linked chunks
        # Create chunk instances
        embeddings = embedder.embed_batch([chunk[0] for chunk in chunks])
        chunks = zip(chunks, embeddings)
        final_chunks = []
        for chunk, embedding in chunks:
            print(chunk)
            chunk_text, start_pos, end_pos = chunk
            chunk_obj = TextChunk(
                document_id = doc.id,
                content=chunk_text,
                chunk_type=ChunkType.TEXT,
                start_pos=start_pos,
                end_pos=end_pos,
                embedding=embedding.tolist(),
                chunk_metadata={"embedder_model_name": embedder.model_name},
            )
            final_chunks.append(chunk_obj)


        return doc, final_chunks
    
    async def retrieve_chunks(self, settings: RetrieveRequest) -> List[Tuple[str, float]]:
        if isinstance(self.retrievers, list):
            results = {}
            for retriever in self.retrievers:
                # TODO: conditional continuation of retrieval if confidence is low
                res = retriever.retrieve(settings)
                results.update({k: results[k]+v for k, v in res})
            results.update({key: value/len(self.retrievers) for key, value in results.items()})
            # sort by confidence and return as list
            return sorted(tuple(results.items()), key=lambda x: x[1], reverse=True)
        elif isinstance(self.retrievers, BaseRetriever):
            return self.retrievers.retrieve(settings)
        else:
            raise Exception("No retriever found")
