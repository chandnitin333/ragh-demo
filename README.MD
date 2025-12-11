mkdir -p ragh/infra/docker
mkdir -p ragh/infra/compose
mkdir -p ragh/infra/k8s

mkdir -p ragh/docs

mkdir -p ragh/src/ragh/ingestion
mkdir -p ragh/src/ragh/embeddings
mkdir -p ragh/src/ragh/vectordb
mkdir -p ragh/src/ragh/retriever
mkdir -p ragh/src/ragh/reader
mkdir -p ragh/src/ragh/pipeline
mkdir -p ragh/src/ragh/api
mkdir -p ragh/src/ragh/utils
mkdir -p ragh/src/ragh/tests

mkdir -p ragh/notebooks

<!-- ============================= -->

touch ragh/src/ragh/**init**.py
touch ragh/src/ragh/config.py
touch ragh/src/ragh/logging_config.py

touch ragh/src/ragh/ingestion/**init**.py
touch ragh/src/ragh/ingestion/loaders.py
touch ragh/src/ragh/ingestion/chunker.py

touch ragh/src/ragh/embeddings/**init**.py
touch ragh/src/ragh/embeddings/embedder.py

touch ragh/src/ragh/vectordb/**init**.py
touch ragh/src/ragh/vectordb/faiss_store.py
touch ragh/src/ragh/vectordb/milvus_store.py

touch ragh/src/ragh/retriever/**init**.py
touch ragh/src/ragh/retriever/retriever.py

touch ragh/src/ragh/reader/**init**.py
touch ragh/src/ragh/reader/reader.py

touch ragh/src/ragh/pipeline/**init**.py
touch ragh/src/ragh/pipeline/rag_pipeline.py

touch ragh/src/ragh/api/**init**.py
touch ragh/src/ragh/api/api_server.py

touch ragh/src/ragh/utils/**init**.py
touch ragh/src/ragh/tests/**init**.py

========================
touch ragh/requirements.txt
touch ragh/pyproject.toml
touch ragh/Dockerfile
touch ragh/docker-compose.yml

export PYTHONPATH=./src && uvicorn ragh.api.api_server:app --reload

cd /Users/nitinchandekar/Personal/AI_PROJECT/ragh/src
uvicorn ragh.api.api_server:app --reload

python3 -m venv venv
source venv/bin/activate
 pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

python -m uvicorn ragh.api.api_server:app --reload


pip install pytesseract pdfplumber python-docx pillow speechrecognition moviepy
pip install pydantic-settings

