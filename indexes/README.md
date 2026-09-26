# Local FAISS indexes

Generated binary indexes are excluded from Git. From the repository root:

```powershell
python scripts/prepare_catalog.py --skip-index-check
python scripts/build_indexes.py
```

Download the catalog and images first, as described in the root README. Building
both indexes encodes the entire catalog and can take substantial time on CPU.
Each index row maps to the same row of `data/image_validated.csv`; do not reorder
the catalog independently. The model downloads on first use.

The published evaluation tables were measured with the original prototype
indexes. Rebuilding with different library/model revisions can change rankings;
rerun evaluation before making claims about your rebuilt version.
