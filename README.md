# RPC Client

This project is an RPC (Remote Procedure Call) server written in Rust. It is designed to communicate with an RPC server, receiving requests and replying swiftly and securely.

## Features
- Written in Rust for performance and safety
- Communicates over RPC
- Handles request/response serialization
- Error handling with `anyhow` and `thiserror`
- Uses `serde` and `serde_json` for data serialization

## Getting Started

### Prerequisites
- Rust (latest stable version recommended)
- Cargo (comes with Rust)

### Building the Project
```
cargo build --release
```

### Running the Client
```
cargo run --release
```

## Project Structure
- `src/main.rs` — Main entry point for the RPC client
- `Cargo.toml` — Project manifest and dependencies
