from __future__ import annotations

from typing import Iterable, Protocol

from lib.schemas import EmbeddingRecord


class EmbeddingStoreProtocol(Protocol):
    def all(self) -> list[EmbeddingRecord]:
        ...

    def append(self, record: EmbeddingRecord) -> None:
        ...

    def replace(self, records: Iterable[EmbeddingRecord]) -> None:
        ...
