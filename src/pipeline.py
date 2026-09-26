"""CLIP embedding and FAISS candidate retrieval.

The module intentionally knows nothing about users, interactions, SQL, or
personalized scoring. It is the ML boundary consumed by Spring Boot.
"""

from __future__ import annotations

import io
import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

import faiss
import numpy as np
import torch
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor


class ProductSearchPipeline:
    def __init__(
        self,
        project_dir: str | Path = ".",
        model_name: str = "openai/clip-vit-base-patch32",
    ) -> None:
        self.project_dir = Path(project_dir)
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: CLIPModel | None = None
        self.processor: CLIPProcessor | None = None
        self.text_index: Any | None = None
        self.image_index: Any | None = None
        self.query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.query_cache_size = max(0, int(os.getenv("QUERY_CACHE_SIZE", "512")))
        self.max_image_bytes = max(
            1, int(os.getenv("MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))
        )
        self._model_lock = Lock()
        self._index_lock = Lock()
        self._cache_lock = Lock()

    @classmethod
    def from_environment(cls) -> "ProductSearchPipeline":
        return cls(
            project_dir=os.getenv("PROJECT_DIR", "."),
            model_name=os.getenv(
                "CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"
            ),
        )

    def health(self) -> dict[str, Any]:
        text_path, image_path = self._index_paths()
        counts: dict[str, int | None] = {"text": None, "image": None}
        if text_path.exists() and image_path.exists():
            self.load_indexes()
            counts = {
                "text": int(self.text_index.ntotal),
                "image": int(self.image_index.ntotal),
            }
        return {
            "status": "ok" if text_path.exists() and image_path.exists() else "degraded",
            "model": self.model_name,
            "device": self.device,
            "indexes": {
                "text": text_path.exists(),
                "image": image_path.exists(),
            },
            "vector_counts": counts,
        }

    def load_models(self) -> None:
        if self.model is None or self.processor is None:
            with self._model_lock:
                if self.model is None or self.processor is None:
                    self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                    self.model.eval()
                    self.processor = CLIPProcessor.from_pretrained(self.model_name)

    def _index_paths(self) -> tuple[Path, Path]:
        return (
            self.project_dir / "indexes" / "text_index.faiss",
            self.project_dir / "indexes" / "image_index.faiss",
        )

    def load_indexes(self) -> None:
        if self.text_index is None or self.image_index is None:
            with self._index_lock:
                if self.text_index is None or self.image_index is None:
                    text_path, image_path = self._index_paths()
                    if not text_path.exists() or not image_path.exists():
                        raise FileNotFoundError(
                            "FAISS indexes are missing. Expected indexes/text_index.faiss "
                            "and indexes/image_index.faiss under PROJECT_DIR."
                        )
                    # Original prototype indexes may contain unnormalised CLIP vectors.
                    # Inner-product search on those vectors is affected by vector length,
                    # so build normalized in-memory indexes once at startup. The files on
                    # disk remain unchanged and index positions still map to catalog rows.
                    self.text_index = self._normalized_copy(
                        faiss.read_index(str(text_path))
                    )
                    self.image_index = self._normalized_copy(
                        faiss.read_index(str(image_path))
                    )

    @staticmethod
    def _normalized_copy(index: Any, batch_size: int = 4096) -> Any:
        if int(index.ntotal) == 0:
            return index
        sample_size = min(256, int(index.ntotal))
        sample = np.asarray(index.reconstruct_n(0, sample_size), dtype="float32")
        if np.allclose(np.linalg.norm(sample, axis=1), 1.0, rtol=1e-4, atol=1e-5):
            return index

        normalized = faiss.IndexFlatIP(int(index.d))
        for start in range(0, int(index.ntotal), batch_size):
            count = min(batch_size, int(index.ntotal) - start)
            vectors = np.asarray(index.reconstruct_n(start, count), dtype="float32")
            faiss.normalize_L2(vectors)
            normalized.add(vectors)
        return normalized

    @staticmethod
    def _normalize(embedding: Any) -> np.ndarray:
        # transformers <=4.x returned a Tensor from get_*_features(), while
        # newer releases return a model-output object whose projected CLIP
        # vector is stored in pooler_output.
        if hasattr(embedding, "pooler_output"):
            embedding = embedding.pooler_output
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return embedding.detach().cpu().numpy().astype("float32")

    def embed_text(self, text: str) -> np.ndarray:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Text must not be blank")
        with self._cache_lock:
            if normalized_text in self.query_cache:
                cached = self.query_cache.pop(normalized_text)
                self.query_cache[normalized_text] = cached
                return cached

        self.load_models()
        assert self.model is not None and self.processor is not None
        inputs = self.processor(
            text=[normalized_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        result = self._normalize(features)[0]
        with self._cache_lock:
            self.query_cache[normalized_text] = result
            while len(self.query_cache) > self.query_cache_size:
                self.query_cache.popitem(last=False)
        return result

    def embed_image(self, image: Image.Image) -> np.ndarray:
        self.load_models()
        assert self.model is not None and self.processor is not None
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            features = self.model.get_image_features(**inputs)
        return self._normalize(features)[0]

    async def read_upload(self, file: UploadFile) -> Image.Image:
        if file.content_type and not file.content_type.startswith("image/"):
            raise ValueError("Uploaded file must be an image")
        contents = await file.read(self.max_image_bytes + 1)
        if not contents:
            raise ValueError("Uploaded image is empty")
        if len(contents) > self.max_image_bytes:
            raise ValueError(
                f"Uploaded image exceeds the {self.max_image_bytes // (1024 * 1024)} MB limit"
            )
        try:
            return Image.open(io.BytesIO(contents)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Uploaded file is not a valid image") from exc

    @staticmethod
    def _reconstruct(index: Any, index_id: int) -> list[float]:
        try:
            vector = np.asarray(index.reconstruct(int(index_id)), dtype="float32")
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            return vector.tolist()
        except RuntimeError as exc:
            raise RuntimeError(
                "The configured FAISS index does not support vector reconstruction; "
                "rebuild it with a reconstructable index type."
            ) from exc

    def _retrieve(
        self,
        search_index: Any,
        embedding_index: Any,
        query: np.ndarray,
        top_k: int,
    ) -> list[dict]:
        limit = min(top_k, int(search_index.ntotal))
        if limit <= 0:
            return []
        scores, indices = search_index.search(query.reshape(1, -1), limit)
        results = []
        for score, index_id in zip(scores[0], indices[0]):
            if int(index_id) < 0:
                continue
            # A single canonical representation makes user profiles stable:
            # the same product always contributes its normalized text/catalog
            # vector, regardless of whether it was found by text or image.
            embedding = self._reconstruct(embedding_index, int(index_id))
            results.append(
                {
                    "index_id": int(index_id),
                    "score": float(score),
                    "embedding": embedding,
                }
            )
        return results

    def retrieve_text(self, query: str, top_k: int = 50) -> list[dict]:
        self.load_indexes()
        return self._retrieve(
            self.text_index, self.text_index, self.embed_text(query), top_k
        )

    def retrieve_image(self, image: Image.Image, top_k: int = 50) -> list[dict]:
        self.load_indexes()
        return self._retrieve(
            self.image_index, self.text_index, self.embed_image(image), top_k
        )

    def embed_text_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.load_models()
        assert self.model is not None and self.processor is not None
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            inputs = self.processor(
                text=texts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.inference_mode():
                batches.append(self._normalize(self.model.get_text_features(**inputs)))
        return np.vstack(batches)

    def embed_image_batch(
        self, image_paths: list[str | Path], batch_size: int = 32
    ) -> np.ndarray:
        self.load_models()
        assert self.model is not None and self.processor is not None
        batches: list[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            images = [
                Image.open(path).convert("RGB")
                for path in image_paths[start : start + batch_size]
            ]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                batches.append(self._normalize(self.model.get_image_features(**inputs)))
        return np.vstack(batches)
