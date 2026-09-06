from dataclasses import dataclass

@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    metadata: dict
    rrf_score: float = 0.0