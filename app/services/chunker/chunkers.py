from abc import ABC, abstractmethod
from typing import List, Tuple
from .ppl_chunking import llm_chunker_ppl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseTextChunker(ABC):
    """Abstract base class for text splitting strategies"""

    @abstractmethod
    def split(
        self, text: str, prefered_chunk_size: int = 400
    ) -> List[Tuple[str, int, int]]:
        """Split text into chunks and return (chunk, start_pos, end_pos) tuples"""
        pass


class PerplexityBasedChunker(BaseTextChunker):
    def __init__(
        self,
        model_name: str = "Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct",
        dynamic_merge: str = "yes",
        threshold: float = 0.2,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.small_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.small_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)
        self.dynamic_merge = dynamic_merge
        self.threshold = threshold
        self.batch_size=4096
        self.max_txt_size=9000

    def split(self, text: str, min_chunk_size: int = 100) -> List[Tuple[str, int, int]]:
        chunks = llm_chunker_ppl(
            text,
            self.small_model,
            self.small_tokenizer,
            self.threshold,
            # TODO language recognition?
            "russian",
            dynamic_merge=self.dynamic_merge,
            target_size=min_chunk_size,
            batch_size=self.batch_size,
            max_txt_size=self.max_txt_size,
        )
        to_return = []
        prev_pos = 0
        for chunk in chunks:
            # TODO: use normal not cursed position findings
            start_pos = text.find(chunk[5:40], prev_pos)
            prev_pos = start_pos
            start_pos -= 5
            to_return.append((chunk, start_pos, start_pos + len(chunk)+1))
        return to_return