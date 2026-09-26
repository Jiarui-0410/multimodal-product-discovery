import unittest

import faiss
import numpy as np

from src.pipeline import ProductSearchPipeline


class ProductSearchPipelineTest(unittest.TestCase):
    def test_normalized_copy_removes_vector_length_bias(self) -> None:
        raw = faiss.IndexFlatIP(2)
        raw.add(np.asarray([[2.0, 0.0], [100.0, 100.0]], dtype="float32"))

        normalized = ProductSearchPipeline._normalized_copy(raw)
        scores, indices = normalized.search(
            np.asarray([[1.0, 0.0]], dtype="float32"), 2
        )

        self.assertEqual([0, 1], indices[0].tolist())
        self.assertAlmostEqual(1.0, float(scores[0][0]), places=6)
        self.assertAlmostEqual(2**-0.5, float(scores[0][1]), places=6)

    def test_retrieval_returns_embedding_from_canonical_index(self) -> None:
        search = faiss.IndexFlatIP(2)
        canonical = faiss.IndexFlatIP(2)
        search.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
        canonical.add(np.asarray([[0.6, 0.8], [0.8, 0.6]], dtype="float32"))

        pipeline = ProductSearchPipeline()
        result = pipeline._retrieve(
            search, canonical, np.asarray([1.0, 0.0], dtype="float32"), 1
        )

        self.assertEqual(0, result[0]["index_id"])
        np.testing.assert_allclose([0.6, 0.8], result[0]["embedding"], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
