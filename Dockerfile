# syntax=docker/dockerfile:1

# CPU-only image for the prebenchmark UI and CLI.
#
# torch is installed first from the PyTorch CPU index. Installing
# requirements.txt straight from PyPI on Linux pulls the CUDA build (~2-3 GB)
# even when no GPU is available, because PyPI's default wheel is the CUDA one.
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG TORCH_VERSION=2.14.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# libgomp1 is loaded by the CPU builds of torch and scikit-learn at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"

COPY requirements.txt requirements-server.txt ./
RUN pip install -r requirements-server.txt

# Create the runtime user and the two writable mount points before anything is
# copied, so bind mounts and named volumes inherit usable ownership on first
# use. The image never writes results itself; both paths are meant to be mounted.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/results /app/.cache \
    && chown -R app:app /app

COPY --chown=app:app prebenchmark ./prebenchmark
COPY --chown=app:app server ./server
COPY --chown=app:app pyproject.toml ./

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status == 200 else 1)"

# --reload is deliberately absent: it watches files and restarts the process,
# which would kill an in-flight training run.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
