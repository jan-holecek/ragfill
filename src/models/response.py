from dataclasses import dataclass
from langchain_core.documents import Document
from models.search import SearchResult

@dataclass
class EmbeddingResponse:
    total_tokens: int
    elapsed: float
    vectors: list[list[float]] | None = None
    chunks: list[tuple[Document, list[float]]] | None = None

    def __post_init__(self):
        self.elapsed = round(self.elapsed, 2)

@dataclass
class RewriteResponse:
    original_query: str
    rewritten_query: str
    elapsed: float

    def __post_init__(self):
        self.elapsed = round(self.elapsed, 2)

@dataclass
class RAGResponse:
    answer: str
    chunks: list[SearchResult]
    completion_tokens: int
    prompt_tokens: int
    elapsed: float
    embedding: EmbeddingResponse | None = None
    rewrite: RewriteResponse | None = None
    used_queries: list[str] | None = None

    def __post_init__(self):
        self.elapsed = round(self.elapsed, 2)

@dataclass
class StreamChunkResponse:
    token: str | None = None
    stats: RAGResponse | None = None
    is_done: bool = False
    rewrite: RewriteResponse | None = None
    embedding: EmbeddingResponse | None = None

@dataclass
class PlaceholderResponse:
    placeholder: str
    prompt: str
    answer: str
    completion_tokens: int
    prompt_tokens: int
    elapsed: float
    rewrite: RewriteResponse | None = None
    chunks: list = None
    used_queries: list[str] | None = None

@dataclass
class TemplateFillResponse:
    results: list[PlaceholderResponse]
    total_elapsed: float
    total_completion_tokens: int
    total_prompt_tokens: int