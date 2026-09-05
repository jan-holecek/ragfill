from core.BaseDB import BaseDB
from elasticsearch import Elasticsearch
from config import ElasticSettings

class ElasticSearchDB(BaseDB):
    def __init__(self, settings: ElasticSettings) -> None:
        self.client = Elasticsearch(
            settings.url,
            verify_certs=False,
            ssl_show_warn=False,
            basic_auth=(settings.username, settings.password) if settings.username else None,
        )
        self.index = settings.index
    
    def disconnect(self) -> None:
        self.client.close()

    def get_client(self) -> Elasticsearch:
        return self.client

    def get_index(self) -> str:
        return self.index
    
    def ping(self) -> bool:
        return self.client.ping()