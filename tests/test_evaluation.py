import unittest

import numpy as np

from src.evaluation import BM25Index, normalized_mean, ranking_metrics


class EvaluationTest(unittest.TestCase):
    def test_bm25_ranks_matching_document_first(self) -> None:
        index = BM25Index(["black sports shoes", "red summer dress"])
        _, ranking = index.search("black shoes", 2)
        self.assertEqual([0, 1], ranking.tolist())

    def test_metrics_use_rank_position(self) -> None:
        result = ranking_metrics([7, 2, 3], {2, 9}, 3)
        self.assertAlmostEqual(1 / 3, result.precision_at_k)
        self.assertAlmostEqual(0.5, result.recall_at_k)
        self.assertEqual(1, result.hits)
        self.assertAlmostEqual(0.5, result.reciprocal_rank)

    def test_normalized_mean_has_unit_length(self) -> None:
        result = normalized_mean(np.asarray([[2.0, 0.0], [0.0, 2.0]]))
        self.assertAlmostEqual(1.0, float(np.linalg.norm(result)), places=6)


if __name__ == "__main__":
    unittest.main()
