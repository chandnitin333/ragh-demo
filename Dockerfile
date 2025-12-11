FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY src /app/src
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "ragh.api.api_server:app", "--host", "0.0.0.0", "--port", "8000"]

