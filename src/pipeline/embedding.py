import time
import litellm
from langchain_core.documents import Document
from config import EmbeddingSettings
from models.response import EmbeddingResponse

litellm.suppress_debug_info = True

# litellm's Ollama embeddings handler calls logging_obj.debug()/.warning() when a
# response is missing prompt_eval_count, but litellm's own Logging class doesn't
# define those methods (present through at least litellm 1.100.0) - patch them in
# so that a normal Ollama response quirk doesn't crash the whole embedding call.
try:
    from litellm._logging import verbose_logger as _litellm_verbose_logger
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLogging

    if not hasattr(_LiteLLMLogging, "debug"):
        _LiteLLMLogging.debug = lambda self, msg, *a, **kw: _litellm_verbose_logger.debug(msg)
    if not hasattr(_LiteLLMLogging, "warning"):
        _LiteLLMLogging.warning = lambda self, msg, *a, **kw: _litellm_verbose_logger.warning(msg)
except ImportError:
    pass

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