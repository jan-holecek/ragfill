from db.mongo.connection import MongoDB
from db.elasticsearch import ElasticSearchDB
from config import settings
from loaders.docx import DocxLoader

def main() -> None:
    mongodb_client = MongoDB(settings.mongo)
    elasticsearch_client = ElasticSearchDB(settings.elastic)

    print(mongodb_client.ping())
    print(elasticsearch_client.ping())

    docx_loader = DocxLoader()

    docx_loader.lazy_load("test")

if __name__ == "__main__":
    main()