from dataclasses import dataclass

from langchain_core.documents import Document
from litellm.types.utils import EmbeddingResponse

from models.search import SearchResult

@dataclass
class RAGResponse:
    answer: str
    chunks: list[SearchResult]
    completion_tokens: int
    prompt_tokens: int
    elapsed: float
    embedding: EmbeddingResponse | None = None

    def __post_init__(self):
        self.elapsed = round(self.elapsed, 2)

@dataclass
class EmbeddingResponse:
    total_tokens: int
    elapsed: float
    vectors: list[list[float]] | None = None
    chunks: list[tuple[Document, list[float]]] | None = None

    def __post_init__(self):
        self.elapsed = round(self.elapsed, 2)

@dataclass
class StreamChunkResponse:
    token: str | None = None
    stats: RAGResponse | None = None
    is_done: bool = False