import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import faiss
import numpy as np
import pandas as pd

from scripts.build_indexes import build, catalog_text


class IndexBuildTest(unittest.TestCase):
    def test_preserves_catalog_order_across_batches_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                dict(id=identifier, productDisplayName="A T-Shirt",
                     masterCategory="Apparel", articleType="Tshirts", baseColour="Blue")
                for identifier in (42, 7, 99)
            ]
            catalog = root / "catalog.csv"
            pd.DataFrame(rows).to_csv(catalog, index=False)
            for row in rows:
                (root / f"{row['id']}.jpg").touch()
            vectors = np.eye(3, dtype="float32")
            with patch("scripts.build_indexes.ProductSearchPipeline.from_environment") as factory:
                pipeline = factory.return_value
                pipeline.model_name = "test-model"
                pipeline.embed_text_batch.side_effect = [vectors[:2], vectors[2:]]
                pipeline.embed_image_batch.side_effect = [vectors[:2], vectors[2:]]
                output = root / "indexes"
                build(catalog, root, output, 2)
                for kind in ("text", "image"):
                    index = faiss.read_index(str(output / f"{kind}_index.faiss"))
                    np.testing.assert_array_equal(index.reconstruct_n(0, 3), vectors)
                paths = pipeline.embed_image_batch.call_args_list[0].args[0]
                self.assertEqual(["42.jpg", "7.jpg"], [p.name for p in paths])
                self.assertEqual(
                    "a tshirt category apparel type tshirts color blue",
                    catalog_text(rows[0]),
                )
                with self.assertRaises(FileExistsError):
                    build(catalog, root, output, 2)


if __name__ == "__main__":
    unittest.main()
