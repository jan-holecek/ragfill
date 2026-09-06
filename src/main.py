from core.loader import BaseLoader
from db.mongo import MongoDB
from db.elasticsearch import ElasticSearchDB
from pipeline.chunking import Chunking
from pipeline.embedding import Embedding
from pipeline.generation import Generation
from pipeline.search import Search
from config import settings

def main() -> None:
    mongodb_client = MongoDB(settings.mongo)
    elasticsearch_client = ElasticSearchDB(settings.elasticsearch)

    loader = BaseLoader.get_loader("cesko.docx")
    chunking = Chunking()
    embedding = Embedding(settings.embedding)

    documents = loader.load()
    chunks = chunking.chunk_documents(documents)
    embedded_chunks = embedding.embed_documents(chunks)
    search = Search(elasticsearch_client)
    generation = Generation(settings.llm)
    print(f"Embedding: {embedded_chunks.elapsed}s | {embedded_chunks.total_tokens} tokens")

    for chunk, vector in embedded_chunks.chunks:
        elasticsearch_client.index_knowledge(chunk.metadata["chunk_id"], chunk.page_content, vector, chunk.metadata)

    query = "Test prompt"
    query_vector = embedding.embed_query(query)

    results = search.search(query, query_vector.vectors[0])
    response = generation.generate(query, results, query_vector)
    print(response)

if __name__ == "__main__":
    main()