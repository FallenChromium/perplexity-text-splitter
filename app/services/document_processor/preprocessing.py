
from typing import Protocol


class DocumentPreprocessor(Protocol):
    """Protocol for document parsers that convert different file formats to markdown"""
    def can_handle(self, content: str) -> bool:
        """Check if this parser can handle the given markdown text"""
        ...
    
    def preprocess(self, content: str) -> str:
        """Convert document content to markdown format"""
        ...