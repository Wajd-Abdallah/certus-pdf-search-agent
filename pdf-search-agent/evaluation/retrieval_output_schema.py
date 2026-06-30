from dataclasses import dataclass
from typing import List


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float


@dataclass
class RetrievalResult:
    question: str
    retrieved_chunks: List[RetrievedChunk]