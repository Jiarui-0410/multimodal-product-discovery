"""Validate catalog images; optionally check existing index row counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import pandas as pd
from PIL import Image


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--styles", type=Path, default=Path("data/styles.csv"))
    parser.add_argument("--images", type=Path, default=Path("images"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/image_validated.csv")
    )
    parser.add_argument("--indexes", type=Path, default=Path("indexes"))
    parser.add_argument(
        "--skip-index-check",
        action="store_true",
        help="Prepare a fresh catalog before building indexes; preserve CSV order.",
    )
    args = parser.parse_args()

    if not args.styles.is_file():
        raise SystemExit(f"Missing source CSV: {args.styles.resolve()}")
    if not args.images.is_dir():
        raise SystemExit(f"Missing image directory: {args.images.resolve()}")

    catalog = pd.read_csv(args.styles, on_bad_lines="skip")
    image_paths = catalog["id"].map(lambda value: args.images / f"{int(value)}.jpg")
    keep = image_paths.map(valid_image)
    prepared = catalog.loc[keep].reset_index(drop=True)

    if not args.skip_index_check:
        for filename in ("text_index.faiss", "image_index.faiss"):
            index_path = args.indexes / filename
            if not index_path.is_file():
                raise SystemExit(
                    f"Missing index: {index_path}. For a fresh checkout, use "
                    "--skip-index-check, then run scripts/build_indexes.py."
                )
            index = faiss.read_index(str(index_path))
            if len(prepared) != int(index.ntotal):
                raise SystemExit(
                    f"Catalog/index mismatch: valid rows={len(prepared)}, "
                    f"{filename} vectors={index.ntotal}. Rebuild both indexes."
                )

    prepared["images_path"] = image_paths.loc[keep].map(
        lambda path: str(path.resolve())
    ).to_numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)
    print(
        f"Prepared {len(prepared)} aligned products at {args.output.resolve()} "
        f"({len(catalog) - len(prepared)} invalid images removed)."
    )


if __name__ == "__main__":
    main()
