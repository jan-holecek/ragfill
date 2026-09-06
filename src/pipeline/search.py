from typing import Any
from elastic_transport import ObjectApiResponse
from config import SearchSettings
from db.elasticsearch import ElasticSearchDB
from models.search import SearchResult

class Search:
    def __init__(self, db: ElasticSearchDB, settings: SearchSettings = SearchSettings()) -> None:
        self.db = db
        self.client = db.get_client()
        self.settings = settings

    def BM25_search(self, query) -> list[SearchResult]:
        results = self.client.search(
            index=self.db.get_index(),
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["*"]
                    }
                },
                "size": self.settings.bm25_K,
                "_source": {"excludes": ["embedding"]},
            }
        )

        return self._return_results(results)

    def knn_search(self, query_vector: list[float]) -> list[SearchResult]:
        results = (
            self.client.search(
            index=self.db.get_index(),
            body={
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": self.settings.knn_K,
                    "num_candidates": self.settings.num_candidates,
                    "similarity": self.settings.knn_score_threshold,
                },
                "_source": {"excludes": ["embedding"]},
            }
        ))

        return self._return_results(results)

    def search(self, query: str, query_vector: list[float]):
        knn_results = self.knn_search(query_vector)
        bm25_results = self.BM25_search(query)

        return self._rrf(bm25_results, knn_results)

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