from abc import ABC, abstractmethod
from typing import List, Tuple
from models import Summary, TextChunk
import yake

from .ppl_chunking import llm_chunker_ppl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseTextChunker(ABC):
    """Abstract base class for text splitting strategies"""

    def __init__(
        self,
        summary_window: int = 3,  # Number of chunks to summarize together
        summary_overlap: bool = True,  # Whether summaries should overlap
        summary_stride: int = 1,  # How many chunks to move window by
    ):
        self.summary_window = summary_window
        self.summary_overlap = summary_overlap
        self.summary_stride = summary_stride if summary_overlap else summary_window

    @abstractmethod
    def split(
        self, text: str, prefered_chunk_size: int = 400
    ) -> List[Tuple[str, int, int]]:
        """Split text into chunks and return (chunk, start_pos, end_pos) tuples"""
        pass

    @abstractmethod
    def create_summaries(self, chunks: List[TextChunk]) -> List[Summary]:
        """Create summaries from chunks"""
        pass


class PerplexityBasedChunker(BaseTextChunker):
    def __init__(
        self,
        model_name: str = "Vikhrmodels/Vikhr-Qwen-2.5-0.5B-Instruct",
        dynamic_merge: str = "yes",
        threshold: float = 0.2,
        summary_window: int = 3,
        summary_stride: int = 2,
        summary_overlap: bool = False
    ):
        super().__init__(summary_window, summary_overlap, summary_stride)
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

    def split(self, text: str, min_chunk_size: int = 784) -> List[Tuple[str, int, int]]:
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
        return chunks

    async def create_summaries(self, chunks: List[str], document_id: int) -> List[Summary]:
        summaries = []
        
        for i in range(0, len(chunks), self.summary_stride):
            window = chunks[i:i + self.summary_window]
            combined_text = "\n".join([chunk.content for chunk in window])
            
            # Prepare input for the model
            prompt = "Create a concise summary (using the language of the provided text) from the following text:\n\n" + combined_text
            inputs = self.small_tokenizer(prompt, return_tensors="pt").to(self.device)
            print(f"generating summary for {i}")
            # Generate summary
            with torch.no_grad():
                outputs = self.small_model.generate(
                    **inputs,
                    max_new_tokens=384,
                    num_return_sequences=1,
                    pad_token_id=self.small_tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_text = self.small_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            summary_text = generated_text
            # Extract keywords using YAKE (Yet Another Keyword Extractor)
            # It's language-independent and doesn't require training

            # Initialize keyword extractor
            # max_ngram_size=2 allows for single words and bigrams
            # deduplication_threshold=0.9 removes similar keywords
            kw_extractor = yake.KeywordExtractor(
                lan="ru", # Automatic language detection # TODO: use info from langdetect
                n=2, # Max ngram size 
                dedupLim=0.9,
                top=10, # Number of keywords to extract
                features=None
            )

            # Extract keywords from the summary text
            keywords = [kw[0] for kw in kw_extractor.extract_keywords(summary_text)]
                
            summary = Summary(
                document_id = document_id,
                content=summary_text,
                keywords=keywords,
                chunk_ids=[chunk.id for chunk in window]
            )
            summaries.append(summary)
            
        return summaries