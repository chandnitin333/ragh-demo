# src/ragh/embeddings/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
from loguru import logger

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info("Loading embedder: %s", model_name)
        self.model = SentenceTransformer(model_name)
        # get dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts):
        # returns numpy array shape (n, dim)
        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        if isinstance(embs, list):
            embs = np.array(embs, dtype="float32")
        return embs
