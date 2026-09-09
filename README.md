# RagFill

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Docker Compose](https://img.shields.io/badge/docker-compose-2496ED.svg)](docker-compose.yaml)

RagFill is a self-hosted Retrieval-Augmented Generation (RAG) backend. It ingests documents, indexes them for hybrid search, and answers questions grounded strictly in that content. It has no dependency on a third-party LLM API, since every model (LLM and embedding) runs locally through Ollama or vLLM and is fronted by LiteLLM.

## Table of contents

- [What it does](#what-it-does)
- [Project structure](#project-structure)
- [Configuration](#configuration)
  - [Choosing a device profile](#choosing-a-device-profile)
  - [Enabling ROCm iGPU support](#enabling-rocm-igpu-support)
- [Managing models](#managing-models)
- [Running it](#running-it)
- [Services and ports](#services-and-ports)
- [TODO](#todo)
- [License](#license)

## What it does

- **Document ingestion** - loads source documents and converts them into clean, Markdown-flavoured text (headings, lists, bold, tables) so structure survives into the chunks. Supports `.docx`, `.pdf`, and `.xlsx`.
- **Chunking** - splits on Markdown headers first, keeps tables intact (an oversized table is split on row boundaries instead, repeating the header row), carries a sentence-aware overlap into the next chunk instead of a blind character cut, and merges chunks that end up too short instead of dropping them.
- **Query rewrite & decomposition** - before searching, an LLM call turns the raw question into one or more standalone, self-contained search queries, resolving pronouns/references from conversation history. Multi-fact questions (e.g. comparing two dates, or joining facts about different entities) are split into independent sub-questions instead of being blended into one blurred embedding.
- **Embedding & indexing** - embeds chunks through LiteLLM and stores them in Elasticsearch alongside their metadata.
- **Hybrid search** - retrieves candidates with both BM25 (keyword) and kNN (vector) search, then fuses the two rankings with Reciprocal Rank Fusion (RRF) so lexical and semantic matches both surface. When the rewrite step decomposes a question into sub-queries, each one is searched independently and the results are merged by RRF score.
- **Template filling** - extracts `{{placeholder}}` markers and their fill instructions from Word comment ranges or Excel cell comments, runs a RAG query per placeholder (or one batched query that fills every placeholder from the merged context), and writes back the extracted values with per-placeholder token/timing stats.
- **Grounded generation** - answers are generated only from retrieved context. The system prompt instructs the model to say when the answer isn't in the documents rather than guessing, and to always reply in Czech. Both blocking and streaming generation are supported, and the same generator powers template filling via a swappable prompt builder.
- **Model-agnostic backend** - the same pipeline runs on CPU, AMD/ROCm GPU, or NVIDIA/CUDA GPU by swapping the Docker Compose profile. The `litellm` SDK abstracts Ollama vs. vLLM behind one call shape, so pipeline code never changes.

## Project structure

```
RagFill/
├── docker-compose.yaml      # Mongo, Elasticsearch, LiteLLM, Ollama, vLLM services
├── litellm_config.yaml      # LiteLLM proxy model routing (llm + embedding routes; not yet used by the pipeline)
├── pyproject.toml           # uv project + dependencies
├── requirements.txt         # plain pip fallback
├── .env.example             # documented environment template
├── scripts/
│   ├── setup.sh             # creates ./data volumes and fixes permissions
│   └── start.sh             # brings the stack up for the chosen DEVICE profile
└── src/
    ├── main.py              # example / test pipeline
    ├── config.py            # pydantic-settings configuration
    ├── prompts.py           # RAG, query-rewrite, and template-fill prompt builders
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
    │   ├── pdf.py           # .pdf -> Markdown via pymupdf4llm
    │   └── xlsx.py          # .xlsx -> one Markdown table per sheet
    ├── models/
    │   ├── response.py      # RAGResponse / EmbeddingResponse / TemplateFillResponse / ...
    │   └── search.py        # SearchResult
    └── pipeline/
        ├── chunking.py      # header-aware Markdown chunking: intact tables, sentence overlap, short-chunk merging
        ├── embedding.py     # embeddings via litellm
        ├── generation.py    # LLM generation (sync + streaming) via litellm
        ├── query_rewrite.py # standalone-query rewriting + multi-fact question decomposition
        ├── search.py        # hybrid BM25 + kNN search, RRF fusion, multi-query search over decomposed sub-queries
        └── template_fill.py # extracts {{placeholders}} from .docx/.xlsx comments and fills them via RAG
```

## Configuration

Settings are loaded from a `.env` file (via pydantic-settings), using `__` as the nested delimiter. For example, `MONGO__URL` fills `settings.mongo.url`. Copy the template to get started:

```bash
cp .env.example .env
```

### Choosing a device profile

The `DEVICE` variable in `.env` selects both the Docker Compose profile and which serving stack `scripts/start.sh` wires up:

| `DEVICE` | Hardware | Serving stack | Notes |
|---|---|---|---|
| `cpu` | No GPU | Ollama | `OLLAMA_NUM_GPU=0` to guarantee CPU-only inference |
| `rocm` | AMD GPU | Ollama | Same Ollama container, with `/dev/kfd` and `/dev/dri` passed through; `OLLAMA_NUM_GPU=999` lets Ollama auto-detect the GPU. Integrated GPUs need extra config - see [Enabling ROCm iGPU support](#enabling-rocm-igpu-support) |
| `cuda` | NVIDIA GPU | vLLM | Two separate vLLM OpenAI-compatible servers (one for the LLM, one for embeddings), requires the NVIDIA Container Toolkit |

`scripts/start.sh` brings up `docker compose --profile ${DEVICE}`, waits for the relevant service(s) to become healthy, and then rewrites the auto-managed `.env` keys (`LLM__API_BASE`, `EMBEDDING__API_BASE`, `REWRITE__API_BASE`, `LLM__LITELLM_MODEL`, `EMBEDDING__LITELLM_MODEL`, `REWRITE__LITELLM_MODEL`) to point at whichever stack it just started. Don't edit those six keys by hand, since they get overwritten on every start.

Whichever profile is active, application code talks directly to the underlying inference server through the `litellm` Python SDK: the auto-managed `*_API_BASE` keys point at Ollama (`http://localhost:11434`) or vLLM (`http://localhost:8000`/`8001`), and the `*_LITELLM_MODEL` keys carry the `ollama/`- or `openai/`-prefixed model id litellm uses to pick the right provider, so pipeline code never has to know which one is running. A standalone LiteLLM proxy also runs alongside the stack on port 4000 for use as a shared gateway later, but the pipeline doesn't route through it yet.

### Enabling ROCm iGPU support

ROCm's GPU detection targets discrete cards. Most integrated GPUs report a chip ID ROCm doesn't recognize and get silently skipped (falling back to CPU), and some iGPUs aren't supported by ROCm at all no matter what you configure. Before enabling this, check that your APU's GPU is RDNA2/RDNA3-class with at least community-reported ROCm/Ollama iGPU support - e.g. the Radeon 780M/880M/890M found in Ryzen 7040/8040/AI 300 ("Phoenix"/"Hawk Point"/"Strix Point") mobile chips. Older or unrelated iGPUs (older RDNA/Vega APUs, non-AMD iGPUs, etc.) generally won't work here regardless of configuration.

For a supported chip, ROCm can usually be coerced into treating the iGPU as a similar, officially-supported one via `HSA_OVERRIDE_GFX_VERSION`. To enable it:

1. In `.env`, set `DEVICE=rocm` and `ROCM_IGPU=1`.
2. Check `HSA_OVERRIDE_GFX_VERSION` / `HCC_AMDGPU_TARGET` in `.env`. The defaults (`11.0.0` / `gfx1100`) target RDNA3 iGPUs like the 780M/880M, which are natively `gfx1103` but run fine reporting as `gfx1100`. If your chip's real `gfx` target differs (e.g. via `rocminfo | grep gfx` on a ROCm-capable machine, or AMD's docs), override these to the closest supported target instead.
3. Run `./scripts/start.sh` as usual. When `ROCM_IGPU=1`, it writes the override values into `.env` and passes them into the Ollama container's environment before the container starts; `OLLAMA_NUM_GPU=999` still handles device auto-detection.
4. Check `sudo docker logs ragfill-ollama` for GPU detection. If the iGPU still doesn't show up, that's most likely a genuinely unsupported chip rather than a configuration problem.

Leave `ROCM_IGPU=0` (the default) if you're running ROCm on a discrete AMD GPU - it's already detected correctly without an override, and forcing a mismatched `HSA_OVERRIDE_GFX_VERSION` onto a different-generation card can break it.

## Managing models

Models are configured entirely through `.env`. Nothing is hardcoded in the app. Example:

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

# Query rewrite model - a small/fast model with a short context window is enough,
# since it only rewrites and decomposes the incoming question (uses the same base
# model as LLM__MODEL, just with a different context window)
REWRITE__CUSTOM_MODEL=gemma3-rewrite
REWRITE__CONTEXT=2048

# Ollama GPU (999 = auto, 0 = cpu only)
OLLAMA_NUM_GPU=999

# ROCm integrated GPU support (DEVICE=rocm only) - see "Enabling ROCm iGPU support"
# in this README before turning this on; not every iGPU is supported by ROCm
ROCM_IGPU=0
# Only applied when ROCM_IGPU=1 - defaults target RDNA3 iGPUs (e.g. Radeon 780M/880M)
# reporting themselves as gfx1100
HSA_OVERRIDE_GFX_VERSION=11.0.0
HCC_AMDGPU_TARGET=gfx1100

# LiteLLM gateway auth
LITELLM_KEY=ragfill
LLM__OPEN_API_KEY=ragfill
EMBEDDING__OPEN_API_KEY=ragfill
REWRITE__OPEN_API_KEY=ragfill

# Auto-set by scripts/start.sh - do not edit manually
LLM__API_BASE=http://localhost:11434
EMBEDDING__API_BASE=http://localhost:11434
REWRITE__API_BASE=http://localhost:11434
LLM__LITELLM_MODEL=ollama/gemma3-128k
EMBEDDING__LITELLM_MODEL=ollama/bge-m3
REWRITE__LITELLM_MODEL=ollama/gemma3-rewrite
```

Swapping a model is a two-line change:

- **cpu / rocm** - set `LLM__MODEL` / `EMBEDDING__MODEL` to any Ollama tag; `scripts/start.sh` pulls it automatically on next start. Set `LLM__CUSTOM_MODEL` + `LLM__CONTEXT` if you want a longer context window than the model's default (Ollama's `num_ctx` is capped at 4096 unless a custom Modelfile raises it, which the script does for you). `REWRITE__CUSTOM_MODEL` + `REWRITE__CONTEXT` work the same way for the query-rewrite model.
- **cuda** - set `LLM__HF_MODEL` / `EMBEDDING__HF_MODEL` to any HuggingFace repo ID vLLM can serve; set `HF_TOKEN` if the repo is gated. Query rewrite reuses `LLM__HF_MODEL` on this profile.

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
| Elasticsearch | 9200 | all | Vector + full-text database |
| MongoDB | 27017 | all | Metadata storage |
| LiteLLM | 4000 | all | Unified LLM proxy |
| Ollama | 11434 | cpu, rocm | Local inference |
| vLLM generation | 8000 | cuda | NVIDIA LLM inference |
| vLLM embedding | 8001 | cuda | NVIDIA embedding inference |

## TODO

- Reranking of hybrid search results
- FastAPI REST API
- Directory crawler - recursively index all files from a local path or network share
- Use MongoDB for uploaded file metadata, currently connected but unused
- LLMOps - integrate Langfuse for prompt/response logging, latency tracking and per-request observability
- Evaluation - build a test dataset and run RAGAS metrics (faithfulness, answer relevancy, context precision) offline to measure and compare pipeline changes
- Automated tests

## License

Licensed under the [MIT License](LICENSE).