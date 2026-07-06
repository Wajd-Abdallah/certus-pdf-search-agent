from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Prediction:
    question: str
    answer: str
    citations: list
    abstained: bool
    abstention_reason: Optional[str]
    retrieved_contexts: list
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    def toDict(self) -> dict:
        return asdict(self)


@dataclass
class Citation:
    document: str
    page_number: int | str
    chunk_id: Optional[str] = None

    def toDict(self) -> dict:
        return asdict(self)