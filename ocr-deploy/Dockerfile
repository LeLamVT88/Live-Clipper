FROM python:3.11-slim

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# - tesseract-ocr: required by pytesseract
# - ffmpeg: required by scenedetect / video loading
# - libglib2.0-0, libgl1-mesa-glx: required by OpenCV (cv2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/laffey/Projects/work/ocr

# Install uv to manage dependencies
RUN pip install --no-cache-dir uv

# Copy project specification files
COPY pyproject.toml uv.lock ./

# Install project dependencies system-wide (using system flag in uv)
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy the rest of the application code
COPY src/ ./src/
COPY .env ./

# Pre-downloaded PaddleOCR models so inference doesn't download at runtime
COPY .cache/paddlex/ /home/laffey/Projects/work/ocr/.cache/paddlex/

# Set environment variables for model caches to be in writable workspace paths
ENV EASYOCR_DOWNLOAD_DIR=/home/laffey/Projects/work/ocr/.cache/easyocr
ENV PADDLEOCR_DOWNLOAD_DIR=/home/laffey/Projects/work/ocr/.cache/paddleocr
ENV TORCH_HOME=/home/laffey/Projects/work/ocr/.cache/torch
ENV PADDLE_PDX_CACHE_HOME=/home/laffey/Projects/work/ocr/.cache/paddlex

# Create cache directories and set permissions
RUN mkdir -p /home/laffey/Projects/work/ocr/.cache/easyocr \
             /home/laffey/Projects/work/ocr/.cache/paddleocr \
             /home/laffey/Projects/work/ocr/.cache/torch
RUN mkdir -p /home/laffey/Projects/work/ocr/.cache/paddlex/temp \
             /home/laffey/Projects/work/ocr/.cache/paddlex/locks \
             /home/laffey/Projects/work/ocr/.cache/paddlex/func_ret

# Create non-root system user for security and set ownership
RUN groupadd -r appgroup && useradd -m -r -g appgroup -u 10001 appuser
RUN chown -R appuser:appgroup /home/laffey/Projects/work/ocr
USER 10001:10001

ENTRYPOINT ["python3", "-m", "src.main"]
