from pydantic import BaseModel
from pydantic_settings import BaseSettings

class MongoSettings(BaseModel):
    url: str
    database: str

class ElasticSettings(BaseModel):
    url: str
    index: str
    username: str | None = None
    password: str | None = None

class LLMSettings(BaseModel):
    model: str
    hf_model: str
    litellm_model: str
    open_api_key: str
    api_base: str
    temperature: float = 0.0
    enable_thinking: bool = False
    timeout: int = 3600

class EmbeddingSettings(BaseModel):
    open_api_key: str
    model: str
    hf_model: str
    litellm_model: str
    api_base: str
    timeout: int = 60

class ChunkingSettings(BaseModel):
    chunk_size: int = 1200
    chunk_overlap: int = 150
    min_chunk_chars: int = 20
    table_size_multiplier: float = 2.0

class SearchSettings(BaseModel):
    bm25_K: int = 4
    knn_K: int = 4
    num_candidates: int = 200
    knn_score_threshold: float = 0.5
    max_subqueries: int = 3
    decomposed_bm25_K: int = 2
    decomposed_knn_K: int = 2
    max_total_chunks: int = 12

class RewriteSettings(BaseModel):
    custom_model: str | None = None
    context: int | None = None
    litellm_model: str
    open_api_key: str
    api_base: str
    enable_thinking: bool = False
    temperature: float = 0.0
    timeout: int = 3600

class Settings(BaseSettings):
    mongo: MongoSettings
    elasticsearch: ElasticSettings
    llm: LLMSettings
    embedding: EmbeddingSettings
    chunking: ChunkingSettings = ChunkingSettings()
    search: SearchSettings = SearchSettings()
    rewrite: RewriteSettings

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        extra = "allow"

settings = Settings()