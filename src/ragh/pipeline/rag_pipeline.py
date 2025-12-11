from ragh.retriever.retriever import Retriever
from ragh.reader.reader import Reader
from typing import List, Dict
from loguru import logger

class RAGPipeline:
    def __init__(self, retriever: Retriever, reader: Reader):
        self.retriever = retriever
        self.reader = reader

    def query(self, question: str, top_k: int = 5) -> Dict:
        logger.info("Query received: %s", question)
        hits = self.retriever.retrieve(question, k=top_k)
        contexts = []
        for h in hits:
            # fetch chunk text via metadata/id mapping
            contexts.append(self._fetch_chunk_text(h["metadata"]))
        answer = self.reader.answer(question, contexts)
        return {
            "answer": answer,
            "retrieved": hits,
            "provenance": [c for c in contexts]
        }

    def _fetch_chunk_text(self, metadata):
        # stub: replace with metadata-store lookup
        return "CHUNK TEXT for index " + str(metadata["index"])
