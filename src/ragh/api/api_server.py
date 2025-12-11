from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from pathlib import Path
import uuid
import asyncio
from pydantic import BaseModel
from loguru import logger

# Local Imports (PYTHONPATH=./src required)
from ragh.ingestion.loaders import extract_text_from_bytes
from ragh.ingestion.chunker import chunk_text
from ragh.embeddings.embedder import Embedder
from ragh.vectordb.faiss_store import FaissStore
from ragh.retriever.retriever import Retriever
from ragh.reader.reader import Reader
from ragh.pipeline.rag_pipeline import RAGPipeline

# -----------------------------
# FastAPI App Initialization
# -----------------------------
app = FastAPI(title="RAGH API")

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("./data")
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Global Components
# -----------------------------
embedder = Embedder()   # Loads SentenceTransformer model
dim = embedder.embedding_dim
store = FaissStore(dim=dim)

retriever = Retriever(embedder, store)
reader = Reader()
pipeline = RAGPipeline(retriever, reader)

# -----------------------------
# Request/Response Models
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    retrieved: list
    provenance: list

# -----------------------------
# Query Endpoint
# -----------------------------
@app.post("/v1/query", response_model=QueryResponse)
def query_q(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        resp = pipeline.query(req.query, top_k=req.top_k)
        return resp
    except Exception as e:
        logger.exception("Query failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Upload Endpoint
# -----------------------------
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload files (PDF, DOCX, TXT, PNG, JPG, JPEG, MP3, WAV, MP4, MOV)
    Extract → Chunk → Embed → Store to FAISS.
    """
    results = []

    for file in files:
        try:
            # -----------------------------
            # Save File
            # -----------------------------
            unique_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
            out_path = UPLOAD_DIR / unique_name
            content = await file.read()
            out_path.write_bytes(content)
            logger.info(f"Saved file: {file.filename} -> {out_path}")

            # -----------------------------
            # Extract Text
            # -----------------------------
            text = await asyncio.get_event_loop().run_in_executor(
                None,
                extract_text_from_bytes,
                file.filename,
                content
            )

            if not text or not text.strip():
                results.append({
                    "file": file.filename,
                    "indexed": 0,
                    "note": "no extractable text found"
                })
                continue

            # -----------------------------
            # Chunk
            # -----------------------------
            chunks = chunk_text(text, max_chars=1800, overlap=200)
            texts = [c["text"] for c in chunks]
            ids = [f"{unique_name}_c{i}" for i in range(len(texts))]

            # -----------------------------
            # Embed (non-blocking)
            # -----------------------------
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None,
                embedder.embed_texts,
                texts
            )

            # -----------------------------
            # Metadata
            # -----------------------------
            metas = [
                {
                    "id": ids[i],
                    "source_file": file.filename,
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                    "text_preview": chunk["text"][:200]
                }
                for i, chunk in enumerate(chunks)
            ]

            # -----------------------------
            # Store in FAISS
            # -----------------------------
            store.add(embeddings, metas, ids)

            results.append({
                "file": file.filename,
                "indexed": len(texts)
            })

        except Exception as e:
            logger.exception(f"Error processing {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed: {file.filename}, {e}")

    return {"status": "ok", "results": results}
