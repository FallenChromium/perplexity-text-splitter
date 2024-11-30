
from abc import ABC
from models import Document

class DocumentPreprocessor(ABC):
    def can_handle(self, content: str) -> bool:
        """Check if this parser can handle the given markdown text"""
        ...
    
    def preprocess(self, content: Document) -> str:
        """Convert document content to markdown format"""
        ...
