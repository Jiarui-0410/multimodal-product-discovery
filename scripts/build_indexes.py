"""Build local CLIP text/image indexes from one validated catalog, in row order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

import faiss
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import ProductSearchPipeline


def catalog_text(row: dict) -> str:
    # Match the fields and cleaning used by the original preprocessing notebook.
    text = (
        f"{row['productDisplayName']}. Category : {row['masterCategory']}. "
        f"Type : {row['articleType']}. Color: {row['baseColour']}"
    )
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", text.lower())).strip()


def build(catalog_path: Path, images: Path, output: Path, batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    table = pd.read_csv(catalog_path).fillna("")
    required = {"id", "productDisplayName", "masterCategory", "articleType", "baseColour"}
    if required.difference(table.columns):
        raise ValueError(f"Missing columns: {sorted(required.difference(table.columns))}")
    if table.empty or table["id"].duplicated().any():
        raise ValueError("Catalog must contain products with unique IDs")
    paths = [images / f"{int(product_id)}.jpg" for product_id in table["id"]]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / f"{kind}_index.faiss" for kind in ("text", "image")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Indexes already exist. Use a new --output directory.")

    pipeline = ProductSearchPipeline.from_environment()
    text_index = image_index = None
    for start in range(0, len(table), batch_size):
        batch = table.iloc[start : start + batch_size]
        text = pipeline.embed_text_batch(
            [catalog_text(row) for row in batch.to_dict("records")], batch_size
        )
        image = pipeline.embed_image_batch(paths[start : start + batch_size], batch_size)
        if text_index is None:
            text_index = faiss.IndexFlatIP(text.shape[1])
            image_index = faiss.IndexFlatIP(image.shape[1])
        text_index.add(text)
        image_index.add(image)
        print(f"Embedded {min(start + batch_size, len(table))}/{len(table)}", flush=True)

    # Finish both encoders before publishing either index.
    with tempfile.TemporaryDirectory(prefix="index-build-", dir=output) as temp:
        for index, target in zip((text_index, image_index), targets):
            staged = Path(temp) / target.name
            faiss.write_index(index, str(staged))
        manifest = {
            "model": pipeline.model_name,
            "products": len(table),
            "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            "normalized": True,
            "dimension": int(text_index.d),
        }
        for target in targets:
            (Path(temp) / target.name).replace(target)
        (output / "build_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    print(f"Built both indexes in {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/image_validated.csv"))
    parser.add_argument("--images", type=Path, default=Path("images"))
    parser.add_argument("--output", type=Path, default=Path("indexes"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    build(args.catalog, args.images, args.output, args.batch_size)


if __name__ == "__main__":
    main()
