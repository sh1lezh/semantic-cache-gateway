# Semantic Cache Gateway

## Overview

The Semantic Cache Gateway is a fast, locally-hosted middleware layer designed to intercept chat completion requests before they reach a downstream Large Language Model (LLM). By generating vector embeddings of incoming user prompts and storing them in a vector database, the gateway can detect semantically similar questions and serve pre-calculated responses. This drastically reduces API costs and latency for repetitive queries.

This application provides an OpenAI-compatible API endpoint (`/v1/chat/completions`), allowing it to serve as a drop-in replacement for existing LLM integrations.

## Proof of Work & Business Capability Disclaimer

**Note:** This repository serves as a **Proof of Work (PoC)**. It is not currently optimized for a production environment. It lacks features such as persistent error handling, authentication, robust logging, and connection pooling.

However, this project demonstrates the foundational capability to design and implement an advanced ML-driven caching layer. It proves that custom AI pipelines—involving tokenization, localized embedding models, and vector database integrations—can be successfully engineered and adapted to suit specific, scalable business use cases.

## Core Architecture and Flow

The application is built using Rust (Axum framework) for high performance and low memory footprint. The request lifecycle follows these steps:

1. **Interception:** The server receives a JSON payload formatted as a standard chat completion request.
2. **Tokenization:** The user prompt is processed using Hugging Face's `tokenizers`, converting the text into `input_ids`, `attention_mask`, and `token_type_ids`.
3. **Embedding Generation (ONNX):** The tokens are passed to a locally hosted embedding model via `ort` (ONNX Runtime). The system performs mean-pooling on the `last_hidden_state` to generate a single 384-dimensional dense vector representing the semantic meaning of the prompt.
4. **Vector Search:** The generated vector is used to query a Qdrant vector database using Cosine Similarity.
5. **Resolution:**
* **Cache Hit:** If a previously stored prompt has a similarity score of 95% (`0.95`) or higher, the gateway immediately returns the cached response.
* **Cache Miss:** If no similar prompt exists, the system dynamically generates a response (simulating an LLM call), stores the new vector and payload in Qdrant, and returns the response to the user.



## Prerequisites & System Setup

To run this project locally, ensure your environment is set up with the necessary dependencies.

### 1. System Dependencies

You will need the Rust toolchain and Docker (to run the Qdrant database). You can install these directly via your package manager:

```bash
sudo pacman -S rustup docker
rustup default stable
sudo systemctl enable --now docker

```

### 2. Model Assets

The application expects an ONNX-formatted embedding model (e.g., a variant of `all-MiniLM-L6-v2`) and its associated tokenizer configuration to be present in a `model/` directory at the project root.

* `model/model.onnx`
* `model/tokenizer.json`

### 3. Running Qdrant

The gateway requires a local instance of the Qdrant vector database. Start it using Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

```

## Running the Gateway

Once Qdrant is running and the model files are in place, compile and run the Rust application:

```bash
cargo run --release

```

The server will initialize the ONNX runtime, connect to Qdrant (creating the `semantic-cache` collection if it does not exist), and bind to `127.0.0.1:3000`.

## API Usage Example

You can interact with the gateway using standard HTTP clients. Send a POST request to the completions endpoint:

```bash
curl -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'

```

**First Request (Cache Miss):**
The server will route the request, embed it, and store the output. The response text will indicate dynamic generation.

**Subsequent Request (Cache Hit):**
If you send the exact same prompt—or one with highly similar phrasing—the server will bypass the generation step and return the payload prefixed with `[CACHED]`.
