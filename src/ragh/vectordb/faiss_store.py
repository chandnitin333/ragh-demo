from typing import List, Tuple, Dict, Optional
import faiss
import numpy as np
from pathlib import Path
from loguru import logger

class FaissStore:
    def __init__(self, dim: int, index_path: Optional[str] = None):
        self.dim = dim
        self.index_path = index_path or "./data/faiss.index"
        self.index = faiss.IndexFlatIP(dim)  # inner product (requires normalized vectors)
        self.ids = []  # simple mapping; for production use an sqlite/kv store
        logger.info("Initialized FAISS index dim=%s", dim)

    def add(self, embeddings: np.ndarray, metadatas: List[dict], ids: List[str]):
        assert embeddings.shape[1] == self.dim
        self.index.add(embeddings.astype(np.float32))
        self.ids.extend(ids)
        # store metadatas in a separate persistent store (json, sqlite, or kv)
        # Here we simply write to disk for sample
        logger.debug("Added %d vectors", embeddings.shape[0])

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        D, I = self.index.search(query_emb.astype(np.float32), top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            results.append((int(idx), float(dist)))
        return results

    def save(self):
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        logger.info("Saved index to %s", self.index_path)

    def load(self):
        if Path(self.index_path).exists():
            self.index = faiss.read_index(self.index_path)
            logger.info("Loaded index from %s", self.index_path)
