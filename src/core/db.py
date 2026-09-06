from abc import ABC, abstractmethod
from typing import Any

from config import MongoSettings, ElasticSettings

class BaseDB(ABC):
    @abstractmethod
    def __init__(self, settings: MongoSettings | ElasticSettings) -> None:
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def get_client(self) -> Any:
        pass
    
    @abstractmethod
    def ping(self) -> bool:
        pass