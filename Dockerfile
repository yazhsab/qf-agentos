# syntax=docker/dockerfile:1
# Multi-stage build for a small, reproducible QF-AgentOS runtime image.

FROM python:3.12-slim AS builder
WORKDIR /build
RUN pip install --no-cache-dir build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# The wheel force-includes examples/ (bundled as Studio presets), so the build
# needs them present in the builder context — not just the runtime stage below.
COPY examples ./examples
RUN python -m build --wheel

FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="QF-AgentOS" \
      org.opencontainers.image.source="https://github.com/yazhsab/qf-agentos" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QF_LOG_FORMAT=json \
    QF_EVIDENCE_DIR=/app/evidence

RUN useradd --create-home --uid 10001 qf
WORKDIR /app

COPY --from=builder /build/dist/*.whl /tmp/
# Install the built wheel with the server + gate-model simulator extras.
RUN WHL="$(ls /tmp/*.whl)" && pip install --no-cache-dir "${WHL}[server,qiskit]" && rm -rf /tmp/*.whl

COPY examples ./examples
RUN mkdir -p /app/evidence && chown -R qf:qf /app
USER qf

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

# Override the CMD to run the CLI, e.g. `docker run --rm qf-agentos qf-agent version`.
CMD ["qf-agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
