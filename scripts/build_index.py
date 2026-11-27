"""Construye la base vectorial indexando el dataset con extract_embedding (Etapa 1).

Requiere haber implementado SimilarityService.extract_embedding (o, si
EMBEDDING_MODEL != baseline, ClassifierService.extract_custom_embedding).
Si USE_PGVECTOR=true, la base de datos debe estar corriendo
(`docker compose up postgres -d`).

Uso:
    python scripts/build_index.py [--split train] [--limit 0]

Dentro de Docker (con `docker compose up -d` corriendo):
    docker compose exec backend python /scripts/build_index.py [--split train]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if not (SRC / "lib").is_dir():
    # Dentro del contenedor del backend: src/ esta montado en /app (WORKDIR) y scripts/ en /scripts.
    ROOT = SRC = Path.cwd()
sys.path.insert(0, str(SRC))

# Usa la misma configuracion que el backend local (src/.env, con paths relativos a src/).
# Si src/.env no existe, se usan los defaults con paths relativos a la raiz del repo.
os.chdir(SRC if (SRC / ".env").is_file() else ROOT)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", default="train", choices=["train"], help="Galería del manifiesto final."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Maximo de imagenes por raza (0 = todas)."
    )
    args = parser.parse_args()

    from lib.bootstrap import build_classifier, build_similarity, build_store
    from lib.config import settings

    store = build_store(settings)
    similarity = build_similarity(settings, store)

    if settings.embedding_model == "baseline":
        extractor = similarity.extract_embedding
    else:
        classifier = build_classifier(settings)
        classifier.set_active_model(settings.embedding_model)
        extractor = classifier.extract_custom_embedding
    similarity.extract_embedding = extractor  # type: ignore[method-assign]

    from collections import Counter
    from lib.tp2_data import rutas_galeria

    counts = Counter()
    def records():
        for path, breed in rutas_galeria(settings.dataset_path):
            if args.limit and counts[breed] >= args.limit:
                continue
            record = similarity.image_record(str(path), breed, {"split": "train", "model": settings.embedding_model})
            if len(record.embedding) != settings.embedding_dim:
                raise ValueError("La dimensión del extractor difiere de EMBEDDING_DIM")
            counts[breed] += 1
            yield record

    if args.limit < 0:
        parser.error("--limit debe ser mayor o igual a cero")
    store.replace(records())
    print(f"Galería: {sum(counts.values())} imágenes, {len(counts)} razas, modelo {settings.embedding_model}")


if __name__ == "__main__":
    main()
