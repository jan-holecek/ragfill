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
    api_base: str

class EmbeddingSettings(BaseModel):
    model: str
    api_base: str

class Settings(BaseSettings):
    mongo: MongoSettings
    elastic: ElasticSettings
    llm: LLMSettings
    embedding: EmbeddingSettings

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

settings = Settings()