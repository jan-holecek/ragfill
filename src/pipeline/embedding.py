import time
import litellm
from langchain_core.documents import Document
from config import EmbeddingSettings
from models.response import EmbeddingResponse

class Embedding:
    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings

    def _call_embedding(self, texts: list[str]) -> litellm.EmbeddingResponse:
        provider = self.settings.litellm_model.split("/")[0]

        return litellm.embedding(
            model=self.settings.litellm_model,
            input=texts,
            api_base=self.settings.api_base,
            api_key=self.settings.open_api_key,
            custom_llm_provider=provider,
            timeout=self.settings.timeout,
            num_retries=2,
        )

    def embed_documents(self, chunks: list[Document]) -> EmbeddingResponse:
        texts = [chunk.page_content for chunk in chunks]
        start = time.time()
        response = self._call_embedding(texts)
        elapsed = time.time() - start
        vectors = [item["embedding"] for item in response.data]

        return EmbeddingResponse(
            vectors=vectors,
            chunks=list(zip(chunks, vectors)),
            total_tokens=response.usage.total_tokens,
            elapsed=elapsed,
        )

    def embed_query(self, query: str) -> EmbeddingResponse:
        return self.embed_queries([query])

    def embed_queries(self, queries: list[str]) -> EmbeddingResponse:
        start = time.time()
        response = self._call_embedding(queries)
        elapsed = time.time() - start
        vectors = [item["embedding"] for item in response.data]

        return EmbeddingResponse(
            vectors=vectors,
            chunks=None,
            total_tokens=response.usage.total_tokens,
            elapsed=elapsed,
        )