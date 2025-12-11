from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENV: str = "dev"
    DEBUG: bool = True

    VECTOR_DB: str = "faiss"  # or "milvus"
    FAISS_INDEX_PATH: str = "./data/faiss.index"

    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    READER_MODEL: str = "google/flan-t5-large"

    TOP_K: int = 5
    MAX_CHUNK_TOKENS: int = 400
    CHUNK_OVERLAP: int = 50
    MAX_DOC_SIZE_MB: int = 50

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

settings = Settings()
