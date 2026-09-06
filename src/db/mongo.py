from pymongo.synchronous.database import Database
from pymongo.synchronous.collection import Collection
from core.db import BaseDB
from pymongo import MongoClient
from config import MongoSettings

class MongoDB(BaseDB):
    def __init__(self, settings: MongoSettings) -> None:
        self.settings = settings
        self.client = MongoClient(self.settings.url)
        self.database = self.client.get_database(self.settings.database)

    def disconnect(self) -> None:
        self.client.close()

    def get_client(self) -> MongoClient:
        return self.client

    def get_db(self) -> Database:
        return self.database

    def get_collection(self, collection_name) -> Collection:
        return self.get_db().get_collection(collection_name)

    def ping(self) -> bool:
        return self.client.admin.command("ping")["ok"] == 1
