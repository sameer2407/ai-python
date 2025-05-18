FROM python:3.10-slim

WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Set environment variables for memory optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV TORCH_HOME=/tmp/torch_cache

# Copy requirements first to leverage Docker cache
COPY requirements-minimal.txt requirements.txt

# Install Python packages with memory optimizations
RUN pip install --no-cache-dir -r requirements.txt && \
  rm -rf /root/.cache/pip/* && \
  rm -rf /root/.cache/torch/* && \
  rm -rf /root/.cache/huggingface/* && \
  rm -rf /root/.cache/numpy/* && \
  rm -rf /root/.cache/scipy/* && \
  rm -rf /root/.cache/sklearn/*

# Copy only necessary files
COPY main.py .
COPY rag_engine.py .
COPY generated/ ./generated/
COPY proto/ ./proto/

# Expose the gRPC port
EXPOSE 50051

CMD ["python", "main.py"]
