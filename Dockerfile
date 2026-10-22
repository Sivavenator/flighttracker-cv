# FlightTracker CV — Docker Image
# GPU-accelerated aircraft detection with YOLOv8 + FR24 integration
# Base: CUDA 12.1 + cuDNN 8 on Ubuntu 22.04

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="Sivasuriyan Sundaresan <slr007suriya@gmail.com>"
LABEL description="Real-time aircraft detection, classification, and FR24 identification"

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    ffmpeg \
    v4l-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python3 -m pip install --upgrade pip

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY flight_tracker_cv/ .

# Create runtime directories
RUN mkdir -p logs screenshots test_videos weights/detector

# Weights are NOT baked into the image.
# Mount them at runtime:  -v /path/to/best.pt:/app/weights/detector/best.pt
# Or download from GitHub Releases in the entrypoint.

# Web dashboard port
EXPOSE 5000

# Default: web dashboard, no OpenCV window (Docker has no display)
# Override with:  docker run ... python main.py --source /app/test_videos/video.mp4 --web --no-window
ENTRYPOINT ["python", "main.py"]
CMD ["--no-window", "--web"]
