FROM rust:1.89-bookworm AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y pkg-config libssl-dev

COPY . .

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends ca-certificates \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user
RUN useradd -m appuser
COPY --from=builder /app/target/release/rpc_client /usr/local/bin/rpc_client
EXPOSE 5500

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD curl -f http://localhost:5500/health || exit 1
USER appuser

CMD ["rpc_client"]
