import litellm
import time
from config import RewriteSettings
from prompts import build_query_rewrite_prompt
from models.response import RewriteResponse

class QueryRewrite:
    def __init__(self, settings: RewriteSettings):
        self.settings = settings

    def rewrite(self, query: str) -> RewriteResponse:
        start = time.time()
        messages = [
            {
                "role": "system",
                "content": build_query_rewrite_prompt(self.settings.enable_thinking)
            },
            {
                "role": "user",
                "content": f"Rewrite this query for search: {query}"
            }
        ]

        response = litellm.completion(
            model=self.settings.litellm_model,
            messages=messages,
            temperature=self.settings.temperature,
            api_base=self.settings.api_base,
            api_key=self.settings.open_api_key,
            extra_body=self._get_extra_body(),
            timeout = self.settings.timeout,
        )

        return RewriteResponse(
            original_query=query,
            rewritten_query=response.choices[0].message.content.strip(),
            elapsed=time.time() - start
        )

    def _get_extra_body(self) -> dict:
        model = self.settings.litellm_model.lower()

        if not self.settings.enable_thinking:
            if "qwen" in model:
                return {"chat_template_kwargs": {"enable_thinking": False}}

            if "deepseek" in model:
                return {}

        return {}