from typing import Any
from elastic_transport import ObjectApiResponse
from config import SearchSettings
from db.elasticsearch import ElasticSearchDB
from models.search import SearchResult
from pipeline.embedding import Embedding

class Search:
    def __init__(self, db: ElasticSearchDB, settings: SearchSettings = SearchSettings()) -> None:
        self.db = db
        self.client = db.get_client()
        self.settings = settings

    def _BM25_search(self, query, bm25_K: int | None = None) -> list[SearchResult]:
        results = self.client.search(
            index=self.db.get_index(),
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["*"]
                    }
                },
                "size": bm25_K if bm25_K is not None else self.settings.bm25_K,
                "_source": {"excludes": ["embedding"]},
            }
        )

        return self._return_results(results)

    def _knn_search(self, query_vector: list[float], knn_K: int | None = None) -> list[SearchResult]:
        results = (
            self.client.search(
            index=self.db.get_index(),
            body={
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": knn_K if knn_K is not None else self.settings.knn_K,
                    "num_candidates": self.settings.num_candidates,
                    "similarity": self.settings.knn_score_threshold,
                },
                "_source": {"excludes": ["embedding"]},
            }
        ))

        return self._return_results(results)

    def _parse_subqueries(self, raw: str) -> list[str]:
        queries = []
        for line in raw.split("\n"):
            line = line.strip()

            if not line:
                continue

            queries.append(line)

        if len(queries) > self.settings.max_subqueries:
            queries = queries[:self.settings.max_subqueries]

        return queries

    def _multi_query_search(self, queries: list[str], embedding: Embedding, bm25_K: int | None = None, knn_K: int | None = None) -> list[SearchResult]:
        merged: dict[str, SearchResult] = {}
        query_vectors = embedding.embed_queries(queries).vectors

        for query, query_vector in zip(queries, query_vectors):
            results = self.search(
                query,
                query_vector,
                bm25_K=bm25_K,
                knn_K=knn_K,
            )

            for result in results:
                result.metadata["matched_subquery"] = query

                if result.id not in merged:
                    merged[result.id] = result

        final_results = sorted(merged.values(), key=lambda result: result.rrf_score, reverse=True)

        return final_results[:self.settings.max_total_chunks]

    def _rrf(self, bm25_results: list[SearchResult], knn_results: list[SearchResult]) -> list[SearchResult]:
        k = 60
        scores = {}
        documents = {}
        final_results = []

        for i, result in enumerate(bm25_results):
            scores[result.id] = scores.get(result.id, 0) + 1 / (k + i + 1)
            documents[result.id] = result

        for i, result in enumerate(knn_results):
            scores[result.id] = scores.get(result.id, 0) + 1 / (k + i + 1)
            documents[result.id] = result

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        for doc_id, rrf_score in ranked:
            doc = documents[doc_id]
            doc.rrf_score = rrf_score

            final_results.append(doc)

        return final_results

    def _return_results(self, results: ObjectApiResponse[Any]) -> list[SearchResult]:
        return [
            SearchResult(
                id=hit["_id"],
                score=hit["_score"],
                text=hit["_source"]["text"],
                metadata={key: value for key, value in hit["_source"].items() if key != "text"}
            )

            for hit in results["hits"]["hits"]
        ]

    def search(self, query: str, query_vector: list[float], bm25_K: int | None = None, knn_K: int | None = None) -> list[SearchResult]:
        knn_results = self._knn_search(query_vector, knn_K)
        bm25_results = self._BM25_search(query, bm25_K)

        return self._rrf(bm25_results, knn_results)

    def parse_and_search(self, raw_rewrite_output: str, embedding: Embedding) -> tuple[list[SearchResult], list[str]]:
        queries = self._parse_subqueries(raw_rewrite_output)

        if len(queries) == 1:
            query_vector = embedding.embed_query(queries[0]).vectors[0]

            return self.search(queries[0], query_vector, self.settings.bm25_K, self.settings.knn_K), queries

        return self._multi_query_search(queries, embedding, self.settings.decomposed_bm25_K, self.settings.decomposed_knn_K), queries