
from typing import Protocol


class DocumentParser(Protocol):
    """Protocol for document parsers that convert different file formats to markdown"""
    def can_handle(self, mime_type: str) -> bool:
        """Check if this parser can handle the given mime type"""
        ...
    
    def parse_to_markdown(self, content: bytes, mime_type: str) -> str:
        """Convert document content to markdown format"""
        ...

class PlainTextParser(DocumentParser):
    def can_handle(self, mime_type: str) -> bool:
        return mime_type in ["text/markdown", "text/plain"]
    
    def parse_to_markdown(self, content: bytes, mime_type: str) -> str:
        return content.decode('utf-8')
