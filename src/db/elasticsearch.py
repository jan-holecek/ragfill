from core.db import BaseDB
from elasticsearch import Elasticsearch
from config import ElasticSettings

class ElasticSearchDB(BaseDB):
    def __init__(self, settings: ElasticSettings) -> None:
        self.settings = settings
        self.client = Elasticsearch(
            self.settings.url,
            verify_certs=False,
            ssl_show_warn=False,
            basic_auth=(self.settings.username, self.settings.password) if self.settings.username else None,
        )
    
    def disconnect(self) -> None:
        self.client.close()

    def get_client(self) -> Elasticsearch:
        return self.client

    def get_index(self) -> str:
        return self.settings.index

    def index_knowledge(self, chunk_id: str, text: str, embedding: list[float], metadata: dict) -> None:
        self.client.index(
            index=self.settings.index,
            id=chunk_id,
            document={
                "text": text,
                "embedding": embedding,
                **metadata,
            }
        )

    def get_knowledge(self, chunk_id: str) -> dict | None:
        try:
            response = self.client.get(index=self.settings.index, id=chunk_id)

            return response["_source"]
        except Exception:
            return None

    def delete_knowledge_by_id(self, chunk_id: str) -> bool:
        result = self.client.delete(
            index=self.settings.index,
            id=chunk_id
        )

        return result["result"] == "deleted"

    def delete_knowledge_by_source(self, source_file: str) -> int:
        result = self.client.delete_by_query(
            index=self.settings.index,
            body={
                "query": {
                    "term": {"source": source_file}
                }
            }
        )

        return result["deleted"]

    def delete_all_knowledge(self) -> int:
        result = self.client.delete_by_query(
            index=self.settings.index,
            body={
                "query": {"match_all": {}}
            }
        )

        return result["deleted"]
    
    def ping(self) -> bool:
        return self.client.ping()