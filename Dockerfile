FROM rust:1.91.1-bookworm AS builder

WORKDIR /app
# Install Python 3.11 and development headers for linking
RUN apt-get update && apt-get install -y pkg-config libssl-dev python3.11 python3.11-dev

COPY . .

RUN cargo build --release

FROM debian:bookworm-slim

# Install Python 3.11 runtime, dev headers (for shared lib), and pip/venv if needed
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends ca-certificates python3.11 python3.11-dev python3.11-venv python3.11-distutils python3-pip \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user
RUN useradd -m appuser

# Install Python dependencies in a virtual environment (portable)
COPY --from=builder /app/requirements.txt /app/requirements.txt

RUN python3.11 -m venv /app/venv \
    && /app/venv/bin/pip install --no-cache-dir -r /app/requirements.txt

# Ensure Python can find local_inference.py
ENV PATH="/app/venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Confirm Python dependencies are installed
RUN /app/venv/bin/pip list

COPY --from=builder /app/local_inference.py /app/local_inference.py
COPY --from=builder /app/merged-model-new /app/merged-model-new

COPY --from=builder /app/target/release/rpc_client /usr/local/bin/rpc_client

EXPOSE 5500

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD curl -f http://localhost:5500/health || exit 1
USER appuser

CMD ["rpc_client"]
