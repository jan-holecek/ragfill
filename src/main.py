from config import settings
from core.loader import BaseLoader
from db.elasticsearch import ElasticSearchDB
from db.mongo import MongoDB
from pipeline.chunking import Chunking
from pipeline.embedding import Embedding
from pipeline.generation import Generation
from pipeline.search import Search
from pipeline.query_rewrite import QueryRewrite
from pipeline.template_fill import TemplateFill

EXAMPLE_FILE = "test.pdf"
EXAMPLE_QUERY = "Test prompt"

def main() -> None:
    mongodb_client = MongoDB(settings.mongo)
    elasticsearch_client = ElasticSearchDB(settings.elasticsearch)

    loader = BaseLoader.get_loader(EXAMPLE_FILE)
    chunking = Chunking(settings.chunking)
    embedding = Embedding(settings.embedding)
    search = Search(elasticsearch_client)
    generation = Generation(settings.llm)
    query_rewrite = QueryRewrite(settings.rewrite)

    template_fill = TemplateFill(embedding, elasticsearch_client, search, generation, query_rewrite)
    template_fill.fill(("template.docx"), "output.docx")


    """
    documents = loader.load()
    chunks = chunking.chunk_documents(documents)
    embedded_chunks = embedding.embed_documents(chunks)
    print(f"Embedding: {embedded_chunks.elapsed}s | {embedded_chunks.total_tokens} tokens")

    for chunk, vector in embedded_chunks.chunks:
        elasticsearch_client.index_knowledge(chunk.metadata["chunk_id"], chunk.page_content, vector, chunk.metadata)

    rewrite_response = query_rewrite.rewrite(EXAMPLE_QUERY)
    print(f"Rewrite: {rewrite_response.elapsed}s | '{rewrite_response.original_query}' -> '{rewrite_response.rewritten_query}'")

    query_vector = embedding.embed_query(rewrite_response.rewritten_query)
    results = search.search(rewrite_response.rewritten_query, query_vector.vectors[0])
    response = generation.generate(EXAMPLE_QUERY, results, query_vector, rewrite_response)
    print(response.answer)
 """
    mongodb_client.disconnect()
    elasticsearch_client.disconnect()

if __name__ == "__main__":
    main()