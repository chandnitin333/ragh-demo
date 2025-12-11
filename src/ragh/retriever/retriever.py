from typing import List, Dict
import numpy as np
from ragh.embeddings.embedder import Embedder
from ragh.vectordb.faiss_store import FaissStore
from ragh.config import settings
from loguru import logger

class Retriever:
    def __init__(self, embedder: Embedder, store: FaissStore, top_k: int = None):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k or settings.TOP_K

    def retrieve(self, query: str, k: int = None):
        k = k or self.top_k
        q_emb = self.embedder.embed_texts([query])
        hits = self.store.search(q_emb, top_k=k)
        results = []
        for idx, score in hits:
            # fetch metadata by idx (example, assume store.ids maps)
            metadata = {"index": idx}
            results.append({"score": score, "metadata": metadata})
        return results
