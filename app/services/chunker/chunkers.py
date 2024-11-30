from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseTextChunker(ABC):
    """Abstract base class for text splitting strategies"""
    @abstractmethod
    def split(self, text: str, prefered_chunk_size: int = 400) -> List[Tuple[str, int, int]]:
        """Split text into chunks and return (chunk, start_pos, end_pos) tuples"""
        pass

class PerplexityBasedChunker(BaseTextChunker):
    def __init__(self, scorer = None):
        self.scorer = scorer
    
    def split(self, text: str, min_chunk_size: int = 100) -> List[Tuple[str, int, int]]:
        # TODO: to be implemented
        pass