import litellm
import time
from langchain_core.documents import Document
from config import EmbeddingSettings
from models.response import EmbeddingResponse

class Embedding:
    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings

    def embed_documents(self, chunks: list[Document]) -> EmbeddingResponse:
        texts = [chunk.page_content for chunk in chunks]
        start = time.time()

        response = litellm.embedding(
            model=self.settings.litellm_model,
            input=texts,
            api_base=self.settings.api_base,
        )

        elapsed = time.time() - start
        vectors = [item["embedding"] for item in response.data]

        return EmbeddingResponse(
            vectors=vectors,
            chunks=list(zip(chunks, vectors)),
            total_tokens=response.usage.total_tokens,
            elapsed=elapsed,
        )

    def embed_query(self, query: str) -> EmbeddingResponse:
        start = time.time()

        response = litellm.embedding(
            model=self.settings.litellm_model,
            input=[query],
            api_base=self.settings.api_base,
        )

        elapsed = time.time() - start
        vectors = [item["embedding"] for item in response.data]

        return EmbeddingResponse(
            vectors=vectors,
            chunks=None,
            total_tokens=response.usage.total_tokens,
            elapsed=elapsed,
        )