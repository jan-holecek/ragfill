# RagFill

RagFill is a self-hosted Retrieval-Augmented Generation (RAG) backend. It ingests documents, indexes them for hybrid search, and answers questions grounded strictly in that content. It has no dependency on a third-party LLM API, since every model (LLM and embedding) runs locally through Ollama or vLLM and is fronted by LiteLLM.

## What it does

- **Document ingestion** - loads source documents and converts them into clean, Markdown-flavoured text (headings, lists, bold, tables) so structure survives into the chunks. Currently supports `.docx` (`src/loaders/docx.py`); `.pdf` and `.xlsx` loaders are stubbed and planned.
- **Chunking** - splits documents on Markdown headers first, then recursively by character count for any section that's still too large, so chunks stay both semantically coherent and size-bounded (`src/pipeline/chunking.py`).
- **Embedding & indexing** - embeds chunks through LiteLLM and stores them in Elasticsearch alongside their metadata (`src/pipeline/embedding.py`, `src/db/elasticsearch.py`).
- **Hybrid search** - retrieves candidates with both BM25 (keyword) and kNN (vector) search, then fuses the two rankings with Reciprocal Rank Fusion (RRF) so lexical and semantic matches both surface (`src/pipeline/search.py`).
- **Grounded generation** - answers are generated only from retrieved context. The system prompt instructs the model to say when the answer isn't in the documents rather than guessing, and to always reply in Czech. Both blocking and streaming generation are supported (`src/pipeline/generation.py`, `src/prompts.py`).
- **Model-agnostic backend** - the same pipeline runs on CPU, AMD/ROCm GPU, or NVIDIA/CUDA GPU by swapping the Docker Compose profile. LiteLLM abstracts Ollama vs. vLLM behind one OpenAI-compatible interface, so pipeline code never changes.

Why this shape: keeping the LLM/embedding backend swappable via LiteLLM means the same codebase runs on a laptop CPU during development and on a CUDA box in production without touching `src/pipeline/*`. Hybrid search with RRF exists because BM25 alone misses paraphrases and kNN alone misses exact terms like IDs and names, so combining them covers both cases.

## Project structure

```
RagFill/
├── docker-compose.yaml      # Mongo, Elasticsearch, LiteLLM, Ollama, vLLM services
├── litellm_config.yaml      # LiteLLM model routing (llm + embedding routes)
├── pyproject.toml           # uv project + dependencies
├── requirements.txt         # plain pip fallback
├── .env.example             # documented environment template
├── scripts/
│   ├── setup.sh             # creates ./data volumes and fixes permissions
│   └── start.sh             # brings the stack up for the chosen DEVICE profile
└── src/
    ├── main.py              # example / test pipeline
    ├── config.py            # pydantic-settings configuration
    ├── prompts.py           # RAG system prompt
    ├── api/                 # (planned) REST API layer
    ├── core/
    │   ├── db.py            # BaseDB interface
    │   └── loader.py        # BaseLoader interface + loader factory
    ├── db/
    │   ├── mongo.py         # MongoDB client wrapper
    │   ├── elasticsearch.py # Elasticsearch client wrapper (index/search/delete)
    │   └── collections/     # (planned) Mongo collection schemas
    ├── loaders/
    │   ├── docx.py          # .docx -> Markdown-flavoured Document loader
    │   ├── pdf.py           # (planned)
    │   └── xlsx.py          # (planned)
    ├── models/
    │   ├── response.py      # RAGResponse / EmbeddingResponse / StreamChunkResponse
    │   └── search.py        # SearchResult
    └── pipeline/
        ├── chunking.py      # Markdown-header + recursive character chunking
        ├── embedding.py     # embeddings via LiteLLM
        ├── generation.py    # LLM generation (sync + streaming) via LiteLLM
        ├── search.py        # hybrid BM25 + kNN search with RRF fusion
        ├── rerank.py        # (planned)
        └── query_rewrite.py # (planned)
```

## Configuration

Settings are loaded by `src/config.py` (pydantic-settings) from a `.env` file, using `__` as the nested delimiter. For example, `MONGO__URL` fills `settings.mongo.url`. Copy the template to get started:

```bash
cp .env.example .env
```

### Choosing a device profile

The `DEVICE` variable in `.env` selects both the Docker Compose profile and which serving stack `scripts/start.sh` wires up:

| `DEVICE` | Hardware | Serving stack | Notes |
|---|---|---|---|
| `cpu` | No GPU | Ollama | `OLLAMA_NUM_GPU=0` to guarantee CPU-only inference |
| `rocm` | AMD GPU | Ollama | Same Ollama container, with `/dev/kfd` and `/dev/dri` passed through; `OLLAMA_NUM_GPU=999` lets Ollama auto-detect the GPU |
| `cuda` | NVIDIA GPU | vLLM | Two separate vLLM OpenAI-compatible servers (one for the LLM, one for embeddings), requires the NVIDIA Container Toolkit |

`scripts/start.sh` brings up `docker compose --profile ${DEVICE}`, waits for the relevant service(s) to become healthy, and then rewrites the auto-managed `.env` keys (`LLM__API_BASE`, `EMBEDDING__API_BASE`, `LLM__LITELLM_MODEL`, `EMBEDDING__LITELLM_MODEL`) to point at whichever stack it just started. Don't edit those four keys by hand, since they get overwritten on every start.

Whichever profile is active, application code always talks to **LiteLLM** (`http://localhost:4000` by default via `LLM__API_BASE`/`EMBEDDING__API_BASE`), which forwards to Ollama or vLLM underneath. `litellm_config.yaml` defines the two routes (`llm`, `embedding`) it proxies.

### Tuning search & chunking

`src/config.py` also exposes (with sensible defaults, overridable via `.env`):

- `CHUNKING__CHUNK_SIZE` / `CHUNKING__CHUNK_OVERLAP` - recursive character splitter bounds (default 2000 / 200).
- `SEARCH__BM25_K` / `SEARCH__KNN_K` - how many hits each retriever contributes before RRF fusion (default 3 / 3).
- `SEARCH__NUM_CANDIDATES` - kNN candidate pool size (default 200).
- `SEARCH__KNN_SCORE_THRESHOLD` - minimum kNN similarity (default 0.5).

## Managing models

Models are configured entirely through `.env`. Nothing is hardcoded in `src/`. Example:

```dotenv
# Device profile: cpu, rocm, cuda
DEVICE=cpu

# MongoDB
MONGO__URL=mongodb://localhost:27017
MONGO__DATABASE=ragfill

# Elasticsearch
ELASTICSEARCH__URL=http://localhost:9200
ELASTICSEARCH__INDEX=ragfill

# LLM model
# Ollama model name (used on cpu / rocm)
LLM__MODEL=gemma3:12b
# Custom Ollama model created with an extended context window (optional)
LLM__CUSTOM_MODEL=gemma3-128k
LLM__CONTEXT=131072
# HuggingFace model ID (used on cuda, served by vLLM)
LLM__HF_MODEL=Qwen/Qwen3-8B-AWQ
HF_TOKEN=hf_your_token_here

# Embedding model
# Ollama model name (cpu / rocm)
EMBEDDING__MODEL=bge-m3
# HuggingFace model ID (cuda)
EMBEDDING__HF_MODEL=BAAI/bge-m3

# Ollama GPU (999 = auto, 0 = cpu only)
OLLAMA_NUM_GPU=999

# LiteLLM gateway auth
LITELLM_KEY=ragfill
LLM__OPEN_API_KEY=ragfill
EMBEDDING__OPEN_API_KEY=ragfill

# Auto-set by scripts/start.sh - do not edit manually
LLM__API_BASE=http://localhost:11434
EMBEDDING__API_BASE=http://localhost:11434
LLM__LITELLM_MODEL=ollama/gemma3-128k
EMBEDDING__LITELLM_MODEL=ollama/bge-m3
```

Swapping a model is a two-line change:

- **cpu / rocm** - set `LLM__MODEL` / `EMBEDDING__MODEL` to any Ollama tag; `scripts/start.sh` pulls it automatically on next start. Set `LLM__CUSTOM_MODEL` + `LLM__CONTEXT` if you want a longer context window than the model's default (Ollama's `num_ctx` is capped at 4096 unless a custom Modelfile raises it, which the script does for you).
- **cuda** - set `LLM__HF_MODEL` / `EMBEDDING__HF_MODEL` to any HuggingFace repo ID vLLM can serve; set `HF_TOKEN` if the repo is gated.

## Running it

Requires Docker, and either `uv` or `pip` with Python 3.11+.

1. **Configure**
   ```bash
   cp .env.example .env
   # edit DEVICE and the model variables above
   ```

2. **Prepare data directories** (Elasticsearch/MongoDB volumes need uid 1000):
   ```bash
   ./scripts/setup.sh
   ```

3. **Start the stack** for your chosen device profile:
   ```bash
   ./scripts/start.sh
   ```
   This starts MongoDB, Elasticsearch, LiteLLM, and either Ollama (cpu/rocm) or the two vLLM servers (cuda). It waits for health checks, pulls/creates the configured Ollama models, and finalizes `.env`.

4. **Install Python dependencies**:
   ```bash
   uv sync
   # or: pip install -r requirements.txt
   ```

5. **Run the example pipeline** to see the full flow (load, chunk, embed, index, search, generate) end-to-end. `src/main.py` is a static example, so edit `EXAMPLE_FILE` and `EXAMPLE_QUERY` at the top of the file to point at a real document before running:
   ```bash
   uv run src/main.py
   # or: python src/main.py
   ```

## Services and ports
 
| Service | Port | Profile | Description |
|---------|------|---------|-------------|
| Streamlit UI | 8501 | all | Test interface |
| Elasticsearch | 9200 | all | Vector + full-text database |
| MongoDB | 27017 | all | Metadata storage |
| LiteLLM | 4000 | all | Unified LLM proxy |
| Ollama | 11434 | cpu, rocm | Local inference |
| vLLM generation | 8000 | cuda | NVIDIA LLM inference |
| vLLM embedding | 8001 | cuda | NVIDIA embedding inference |

## TODO
- Query rewriting step before retrieval
- Reranking of hybrid search results - BGE reranker planned, waiting for GPU
- Additional loaders: PDF, XLSX, etc
- Template filling - extract placeholders from a Word template, run a RAG query per placeholder, write values back
- REST API so the pipeline can be used without a Python entrypoint
- Use MongoDB for chat history and uploaded file metadata, currently connected but unused
- LLMOps - integrate Langfuse for prompt/response logging, latency tracking and RAG quality evaluation
- Automated tests
