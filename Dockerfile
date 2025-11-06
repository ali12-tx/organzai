FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend and web
COPY backend/ /app/backend/
COPY web/ /app/web/

# Install Python deps (PyTorch from CUDA 12.1 wheel index)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

EXPOSE 8000
WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

