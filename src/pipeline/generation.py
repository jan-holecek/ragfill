import time
from models.response import RAGResponse, EmbeddingResponse
from config import LLMSettings
from models.search import SearchResult
from prompts import build_rag_prompt
import litellm

class Generation:
    def __init__(self, settings: LLMSettings):
        self.settings = settings

    def generate(self, query: str, chunks: list[SearchResult], embedding_response: EmbeddingResponse | None = None) -> RAGResponse:
        context = "\n\n".join([chunk.text for chunk in chunks])
        start = time.time()

        messages = [
            {
                "role": "system",
                "content": build_rag_prompt(context)
            },
            {
                "role": "user",
                "content": query
            }
        ]

        response = litellm.completion(
            model=self.settings.litellm_model,
            messages=messages,
            temperature=self.settings.temperature,
            api_base=self.settings.api_base,
        )

        end = time.time()

        return RAGResponse(
            answer=response.choices[0].message["content"],
            chunks=chunks,
            completion_tokens=response.usage.completion_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            elapsed=end-start,
            embedding=embedding_response
        )