from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class RetrievedChunk:
    text: str
    score: float


class SimpleTokenRetriever:
    """A lightweight lexical retriever based on token overlap.

    This retriever avoids external vector DB dependencies and is suitable
    for a small local corpus demonstration of RAG.
    """

    def __init__(self, tokenizer, chunks: Sequence[str], min_chunk_chars: int = 1):
        self.tokenizer = tokenizer
        self.chunks: List[str] = [c.strip() for c in chunks if len(c.strip()) >= min_chunk_chars]
        self.chunk_token_sets: List[set[int]] = [set(tokenizer.encode(c)) for c in self.chunks]

    @classmethod
    def from_text(cls, tokenizer, text: str, chunk_chars: int = 300):
        if chunk_chars <= 0:
            raise ValueError("chunk_chars must be positive")

        chunks = []
        start = 0
        text = text.strip()
        while start < len(text):
            chunks.append(text[start : start + chunk_chars])
            start += chunk_chars
        return cls(tokenizer, chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        if top_k <= 0:
            return []

        query_tokens = set(self.tokenizer.encode(query))
        if not query_tokens:
            return []

        scored: List[RetrievedChunk] = []
        for chunk, token_set in zip(self.chunks, self.chunk_token_sets):
            if not token_set:
                continue
            overlap = len(query_tokens & token_set)
            if overlap == 0:
                continue
            # Overlap ratio keeps scores roughly in [0, 1].
            score = overlap / len(query_tokens)
            scored.append(RetrievedChunk(text=chunk, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


def build_augmented_prompt(query: str, retrieved: Sequence[RetrievedChunk]) -> str:
    if not retrieved:
        return query

    context_blocks = [item.text for item in retrieved]
    context_text = "\n\n".join(context_blocks)
    return (
        "You are given reference context. Use it when relevant.\n\n"
        f"[Context]\n{context_text}\n\n"
        f"[Question]\n{query}\n\n"
        "[Answer]\n"
    )
