from config import settings
from core.loader import BaseLoader
from db.elasticsearch import ElasticSearchDB
from db.mongo import MongoDB
from pipeline.chunking import Chunking
from pipeline.embedding import Embedding
from pipeline.generation import Generation
from pipeline.search import Search

EXAMPLE_FILE = "example.docx"
EXAMPLE_QUERY = "Test prompt"

def main() -> None:
    mongodb_client = MongoDB(settings.mongo)
    elasticsearch_client = ElasticSearchDB(settings.elasticsearch)

    loader = BaseLoader.get_loader(EXAMPLE_FILE)
    chunking = Chunking()
    embedding = Embedding(settings.embedding)
    search = Search(elasticsearch_client)
    generation = Generation(settings.llm)

    documents = loader.load()
    chunks = chunking.chunk_documents(documents)
    embedded_chunks = embedding.embed_documents(chunks)
    print(f"Embedding: {embedded_chunks.elapsed}s | {embedded_chunks.total_tokens} tokens")

    for chunk, vector in embedded_chunks.chunks:
        elasticsearch_client.index_knowledge(chunk.metadata["chunk_id"], chunk.page_content, vector, chunk.metadata)

    query_vector = embedding.embed_query(EXAMPLE_QUERY)
    results = search.search(EXAMPLE_QUERY, query_vector.vectors[0])
    response = generation.generate(EXAMPLE_QUERY, results, query_vector)
    print(response.answer)

    mongodb_client.disconnect()
    elasticsearch_client.disconnect()

if __name__ == "__main__":
    main()
