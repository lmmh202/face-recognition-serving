# Build for linux/amd64: Azure Container Apps does not run arm64 images.
FROM python:3.14.6-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1
WORKDIR /srv
COPY pyproject.toml uv.lock ./
# TensorFlow's C++ headers are only needed to compile against it.
RUN uv sync --frozen --no-dev --no-install-project \
    && rm -rf /opt/venv/lib/python3.14/site-packages/tensorflow/include

FROM python:3.14.6-slim AS app
RUN useradd --uid 10001 --create-home app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /srv
COPY app ./app
COPY ml ./ml
COPY models ./models
COPY artifacts ./artifacts
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

# docker build --target test: the app image plus dev dependencies and tests.
FROM app AS test
USER root
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1 \
    uv sync --frozen --no-install-project
COPY tests ./tests
USER app
CMD ["python", "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"]

FROM app AS runtime
