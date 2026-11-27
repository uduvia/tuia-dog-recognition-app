from __future__ import annotations

import json
import os
import tempfile
from typing import Iterable
from pathlib import Path

from lib.schemas import EmbeddingRecord


class EmbeddingStore:
    """Base vectorial simple sobre un archivo JSON (alternativa a pgvector)."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self.storage_path.write_text("[]", encoding="utf-8")

    def all(self) -> list[EmbeddingRecord]:
        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        return [EmbeddingRecord.model_validate(item) for item in payload]

    def save(self, records: list[EmbeddingRecord]) -> None:
        raw = [item.model_dump() for item in records]
        self.storage_path.write_text(
            json.dumps(raw, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def append(self, record: EmbeddingRecord) -> None:
        records = self.all()
        records.append(record)
        self.save(records)

    def replace(self, records: Iterable[EmbeddingRecord]) -> None:
        with tempfile.NamedTemporaryFile(mode="w", dir=self.storage_path.parent, encoding="utf-8",
                                         delete=False) as stream:
            temp_path = Path(stream.name)
        try:
            with temp_path.open("w", encoding="utf-8") as stream:
                stream.write("[")
                for i, record in enumerate(records):
                    if i:
                        stream.write(",")
                    stream.write(record.model_dump_json())
                stream.write("]")
            os.replace(temp_path, self.storage_path)
        finally:
            temp_path.unlink(missing_ok=True)
