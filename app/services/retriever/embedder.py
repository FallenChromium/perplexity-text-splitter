from abc import ABC, abstractmethod
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from typing import List
class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        pass


class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])
        
        return embeddings.cpu().numpy()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)