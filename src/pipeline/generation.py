import time
from typing import Generator
from models.response import RAGResponse, EmbeddingResponse, StreamChunkResponse, RewriteResponse
from config import LLMSettings
from models.search import SearchResult
from prompts import build_rag_prompt
import litellm

litellm.cache = litellm.Cache(type="local")

class Generation:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def generate(self, query: str, chunks: list[SearchResult], embedding_response: EmbeddingResponse | None = None, rewrite_response: RewriteResponse | None = None, prompt_builder=build_rag_prompt) -> RAGResponse:
        if not chunks:
            return RAGResponse(
                answer="Nenašel jsem relevantní informace k vašemu dotazu.",
                chunks=None,
                completion_tokens=0,
                prompt_tokens=0,
                elapsed=0,
                rewrite=rewrite_response,
                embedding=embedding_response
            )

        context = "\n\n".join([chunk.text for chunk in chunks])
        messages = [
            {"role": "system", "content": prompt_builder(context, no_think=self.settings.enable_thinking)},
            {"role": "user", "content": query}
        ]

        start = time.time()

        response = litellm.completion(
            model=self.settings.litellm_model,
            messages=messages,
            temperature=self.settings.temperature,
            api_base=self.settings.api_base,
            api_key=self.settings.open_api_key,
            extra_body=self._get_extra_body(),
            caching=True,
            timeout=self.settings.timeout,
        )

        end = time.time()

        return RAGResponse(
            answer=response.choices[0].message["content"],
            chunks=chunks,
            completion_tokens=response.usage.completion_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            elapsed=end - start,
            rewrite=rewrite_response,
            embedding=embedding_response,
        )

    def stream_generate(self, query: str, chunks: list[SearchResult], embedding_response: EmbeddingResponse | None = None, rewrite_response: RewriteResponse | None = None, prompt_builder=build_rag_prompt) -> Generator[StreamChunkResponse, None, None]:
        if not chunks:
            yield StreamChunkResponse(
                token="Nenašel jsem relevantní informace k vašemu dotazu.",
                rewrite=rewrite_response,
                embedding=embedding_response,
                is_done=True
            )
            return

        context = "\n\n".join([chunk.text for chunk in chunks])
        messages = [
            {"role": "system", "content": prompt_builder(context, no_think=self.settings.enable_thinking)},
            {"role": "user", "content": query}
        ]
        start = time.time()
        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        response = litellm.completion(
            model=self.settings.litellm_model,
            messages=messages,
            temperature=self.settings.temperature,
            api_base=self.settings.api_base,
            api_key=self.settings.open_api_key,
            extra_body=self._get_extra_body(),
            caching=True,
            stream=True,
            stream_options={"include_usage": True},
            timeout=self.settings.timeout,
        )

        for chunk in response:
            delta = chunk.choices[0].delta.content

            if delta:
                full_text += delta

                yield StreamChunkResponse(token=delta)

            if hasattr(chunk, "usage") and chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

        elapsed = time.time() - start

        yield StreamChunkResponse(
            is_done=True,
            stats=RAGResponse(
                answer=full_text,
                chunks=chunks,
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                elapsed=elapsed,
                rewrite=rewrite_response,
                embedding=embedding_response,
            )
        )

    def _get_extra_body(self) -> dict:
        model = self.settings.litellm_model.lower()

        if not self.settings.enable_thinking:
            if "qwen" in model:
                return {"chat_template_kwargs": {"enable_thinking": False}}

            if "deepseek" in model:
                return {}

        return {}