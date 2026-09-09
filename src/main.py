from config import settings
from db.elasticsearch import ElasticSearchDB
from db.mongo import MongoDB
from pipeline.chunking import Chunking
from pipeline.embedding import Embedding
from pipeline.generation import Generation
from pipeline.search import Search
from pipeline.query_rewrite import QueryRewrite
from pipeline.template_fill import TemplateFill

def main() -> None:
    mongodb_client = MongoDB(settings.mongo)
    elasticsearch_client = ElasticSearchDB(settings.elasticsearch)

    chunking = Chunking(settings.chunking)
    embedding = Embedding(settings.embedding)
    search = Search(elasticsearch_client)
    generation = Generation(settings.llm)
    query_rewrite = QueryRewrite(settings.rewrite)
    template_fill = TemplateFill(elasticsearch_client, embedding, generation, query_rewrite)

    mongodb_client.disconnect()
    elasticsearch_client.disconnect()

if __name__ == "__main__":
    main()