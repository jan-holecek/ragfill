from abc import ABC, abstractmethod
from config import MongoSettings, ElasticSettings

class BaseDB(ABC):
    @abstractmethod
    def __init__(self, settings: MongoSettings | ElasticSettings) -> None:
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def get_client(self) -> MongoSettings | ElasticSettings:
        pass
    
    @abstractmethod
    def ping(self) -> bool:
        pass