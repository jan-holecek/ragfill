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
    api_base: str
    temperature: float = 0.7

class EmbeddingSettings(BaseModel):
    model: str
    hf_model: str
    litellm_model: str
    api_base: str

class ChunkingSettings(BaseModel):
    chunk_size: int = 2000
    chunk_overlap: int = 200

class SearchSettings(BaseModel):
    bm25_K: int = 5
    knn_K: int = 5
    num_candidates: int = 100
    knn_score_threshold: float = 0.5

class Settings(BaseSettings):
    mongo: MongoSettings
    elasticsearch: ElasticSettings
    llm: LLMSettings
    embedding: EmbeddingSettings
    chunking: ChunkingSettings = ChunkingSettings()
    search: SearchSettings = SearchSettings()

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        extra = "allow"

settings = Settings()