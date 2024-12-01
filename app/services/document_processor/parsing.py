
from abc import ABC
from markdownify import markdownify as md


class DocumentParser(ABC):
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
    
class HTMLParser(DocumentParser):
    def can_handle(self, mime_type: str) -> bool:
        return mime_type in ["text/html"]
    
    def parse_to_markdown(self, content: bytes, mime_type: str) -> str:
        return md(content.decode('utf-8'))
