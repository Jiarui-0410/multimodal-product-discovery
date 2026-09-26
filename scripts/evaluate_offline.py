"""Run reproducible retrieval and personalization benchmarks.

The personalization benchmark uses simulated preference profiles because the
Fashion Product Images dataset has no real user-event history. Results must be
reported with that limitation, never as an online user experiment.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    BM25Index,
    bootstrap_mean_interval,
    normalized_mean,
    ranking_metrics,
    tokenize,
    unit_interval,
)
from src.pipeline import ProductSearchPipeline


PARAPHRASES = {
    "Sports Shoes": "workout sneakers",
    "Casual Shoes": "everyday footwear",
    "Shirts": "button-down tops",
    "Tshirts": "short-sleeve tees",
    "Jeans": "denim trousers",
    "Watches": "wrist timepieces",
    "Handbags": "carry purses",
    "Backpacks": "rucksacks",
    "Dresses": "one-piece outfits",
    "Heels": "high-heeled footwear",
    "Sandals": "open-toe footwear",
    "Track Pants": "jogging bottoms",
}

GENDER_PHRASES = {
    "Men": "men",
    "Women": "women",
    "Boys": "boys",
    "Girls": "girls",
    "Unisex": "anyone",
}

METRIC_COLUMNS = [
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "hit_rate_at_k",
    "reciprocal_rank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("data/image_validated.csv"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries-per-type", type=int, default=3)
    parser.add_argument("--minimum-relevant", type=int, default=20)
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument("--candidate-pool", type=int, default=500)
    parser.add_argument(
        "--max-personas",
        type=int,
        default=0,
        help="0 evaluates every eligible preference combination.",
    )
    return parser.parse_args()


def clean_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(path).fillna("").reset_index(drop=True)
    required = {
        "id",
        "gender",
        "masterCategory",
        "articleType",
        "baseColour",
        "usage",
        "productDisplayName",
    }
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"Catalog is missing columns: {sorted(missing)}")
    return catalog


def eligible_combinations(catalog: pd.DataFrame, minimum: int) -> pd.DataFrame:
    selected = catalog[catalog["articleType"].isin(PARAPHRASES)]
    grouped = (
        selected.groupby(["articleType", "baseColour", "gender"], dropna=False)
        .size()
        .reset_index(name="catalog_relevant_count")
    )
    return grouped[grouped["catalog_relevant_count"] >= minimum].reset_index(drop=True)


def relevant_indices(
    catalog: pd.DataFrame, article_type: str, color: str, gender: str
) -> np.ndarray:
    mask = (
        catalog["articleType"].eq(article_type)
        & catalog["baseColour"].eq(color)
        & catalog["gender"].eq(gender)
    )
    return np.flatnonzero(mask.to_numpy())


def build_query_benchmark(
    combinations: pd.DataFrame,
    queries_per_type: int,
    seed: int,
) -> pd.DataFrame:
    random = np.random.default_rng(seed)
    rows: list[dict] = []
    query_id = 0
    for article_type in PARAPHRASES:
        candidates = combinations[combinations["articleType"].eq(article_type)]
        sample_size = min(queries_per_type, len(candidates))
        chosen = random.choice(candidates.index.to_numpy(), sample_size, replace=False)
        for row_index in chosen:
            target = combinations.loc[row_index]
            color = str(target["baseColour"])
            gender = str(target["gender"])
            query_id += 1
            shared = {
                "target_id": query_id,
                "article_type": article_type,
                "color": color,
                "gender": gender,
                "relevant_count": int(target["catalog_relevant_count"]),
            }
            rows.append(
                {
                    **shared,
                    "query_id": f"Q{query_id:03d}-literal",
                    "cohort": "literal",
                    "query": f"{gender} {color} {article_type}",
                }
            )
            rows.append(
                {
                    **shared,
                    "query_id": f"Q{query_id:03d}-paraphrased",
                    "cohort": "paraphrased",
                    "query": (
                        f"{color.lower()} {PARAPHRASES[article_type]} "
                        f"for {GENDER_PHRASES.get(gender, gender.lower())}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def catalog_search_text(catalog: pd.DataFrame) -> list[str]:
    columns = ["productDisplayName", "masterCategory", "articleType", "baseColour"]
    return catalog[columns].astype(str).agg(" ".join, axis=1).tolist()


def summarize(
    frame: pd.DataFrame,
    group_columns: list[str],
    seed: int,
) -> pd.DataFrame:
    random = np.random.default_rng(seed)
    rows: list[dict] = []
    group_key = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in frame.groupby(group_key, sort=False):
        key_values = (key,) if len(group_columns) == 1 else key
        row = dict(zip(group_columns, key_values))
        row["observations"] = len(group)
        for metric in METRIC_COLUMNS:
            values = group[metric].astype(float).to_numpy()
            low, high = bootstrap_mean_interval(values, random)
            row[metric] = float(values.mean())
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_retrieval(
    catalog: pd.DataFrame,
    queries: pd.DataFrame,
    pipeline: ProductSearchPipeline,
    k: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vocabulary = {token for query in queries["query"] for token in tokenize(query)}
    bm25 = BM25Index(catalog_search_text(catalog), vocabulary)
    clip_queries = pipeline.embed_text_batch(queries["query"].tolist(), batch_size=32)
    clip_scores, clip_rankings = pipeline.text_index.search(clip_queries, k)
    rows: list[dict] = []

    for position, query in queries.iterrows():
        relevant = set(
            relevant_indices(
                catalog, query["article_type"], query["color"], query["gender"]
            ).tolist()
        )
        bm25_scores, bm25_ranking = bm25.search(query["query"], k)
        methods = {
            "BM25": (bm25_ranking, bm25_scores),
            "CLIP": (clip_rankings[position], clip_scores[position]),
        }
        for method, (ranking, scores) in methods.items():
            metrics = ranking_metrics(ranking.tolist(), relevant, k)
            rows.append(
                {
                    "query_id": query["query_id"],
                    "cohort": query["cohort"],
                    "query": query["query"],
                    "method": method,
                    "relevant_count": len(relevant),
                    "top_index_ids": "|".join(str(int(value)) for value in ranking),
                    "top_scores": "|".join(f"{float(value):.6f}" for value in scores),
                    **metrics.to_dict(),
                }
            )

    details = pd.DataFrame(rows)
    summary = summarize(details, ["cohort", "method"], seed + 1)
    overall = summarize(details, ["method"], seed + 2)
    overall.insert(0, "cohort", "overall")
    return details, pd.concat([summary, overall], ignore_index=True)


def normalized_frequency(values: pd.Series) -> dict[str, float]:
    counts = values.astype(str).value_counts()
    total = float(counts.sum())
    return {str(key): float(value / total) for key, value in counts.items()}


def evaluate_personalization(
    catalog: pd.DataFrame,
    combinations: pd.DataFrame,
    pipeline: ProductSearchPipeline,
    text_vectors: np.ndarray,
    k: int,
    candidate_pool: int,
    history_size: int,
    seed: int,
    max_personas: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random = np.random.default_rng(seed)
    personas = combinations.copy()
    if max_personas > 0 and len(personas) > max_personas:
        chosen = random.choice(personas.index.to_numpy(), max_personas, replace=False)
        personas = personas.loc[sorted(chosen)].reset_index(drop=True)

    broad_queries = list(PARAPHRASES.values())
    query_vectors = pipeline.embed_text_batch(broad_queries, batch_size=32)
    query_by_type = dict(zip(PARAPHRASES, query_vectors))
    candidate_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    retrieval_limit = min(len(catalog), candidate_pool + history_size)
    for article_type, query_vector in query_by_type.items():
        scores, indices = pipeline.text_index.search(query_vector.reshape(1, -1), retrieval_limit)
        candidate_cache[article_type] = (indices[0], scores[0])

    detail_rows: list[dict] = []
    persona_rows: list[dict] = []
    for persona_number, (_, persona) in enumerate(personas.iterrows(), 1):
        article_type = str(persona["articleType"])
        color = str(persona["baseColour"])
        gender = str(persona["gender"])
        pool = relevant_indices(catalog, article_type, color, gender)
        if len(pool) <= history_size:
            continue
        history = np.sort(random.choice(pool, history_size, replace=False))
        relevant = set(np.setdiff1d(pool, history).tolist())
        raw_candidates, raw_query_scores = candidate_cache[article_type]
        keep = ~np.isin(raw_candidates, history)
        candidate_ids = raw_candidates[keep][:candidate_pool]
        query_scores = raw_query_scores[keep][:candidate_pool]
        candidate_vectors = text_vectors[candidate_ids]

        semantic_profile = normalized_mean(text_vectors[history])
        semantic_scores = unit_interval(candidate_vectors @ semantic_profile)
        query_unit_scores = unit_interval(query_scores)
        history_rows = catalog.iloc[history]
        usage_preferences = normalized_frequency(history_rows["usage"])
        candidate_rows = catalog.iloc[candidate_ids]
        metadata_scores = (
            0.5 * candidate_rows["articleType"].eq(article_type).to_numpy(dtype=float)
            + 0.3 * candidate_rows["baseColour"].eq(color).to_numpy(dtype=float)
            + 0.2
            * candidate_rows["usage"]
            .map(lambda value: usage_preferences.get(str(value), 0.0))
            .to_numpy(dtype=float)
        )

        method_scores = {
            "Random": random.random(len(candidate_ids)),
            "CLIP": query_unit_scores,
            "CLIP + Metadata": (0.6 * query_unit_scores + 0.1 * metadata_scores) / 0.7,
            "CLIP + Semantic": (0.6 * query_unit_scores + 0.3 * semantic_scores) / 0.9,
            "Full Hybrid": (
                0.6 * query_unit_scores
                + 0.3 * semantic_scores
                + 0.1 * metadata_scores
            ),
        }
        persona_id = f"U{persona_number:03d}"
        candidates_relevant = sum(int(value in relevant) for value in candidate_ids)
        persona_rows.append(
            {
                "persona_id": persona_id,
                "query": PARAPHRASES[article_type],
                "preferred_article_type": article_type,
                "preferred_color": color,
                "preferred_gender": gender,
                "history_product_ids": "|".join(
                    str(int(value)) for value in catalog.iloc[history]["id"]
                ),
                "history_size": len(history),
                "held_out_relevant_count": len(relevant),
                "relevant_in_candidate_pool": candidates_relevant,
                "candidate_recall": candidates_relevant / len(relevant),
            }
        )
        for method, scores in method_scores.items():
            order = np.argsort(-scores, kind="stable")
            ranking = candidate_ids[order]
            metrics = ranking_metrics(ranking.tolist(), relevant, k)
            detail_rows.append(
                {
                    "persona_id": persona_id,
                    "query": PARAPHRASES[article_type],
                    "preferred_article_type": article_type,
                    "preferred_color": color,
                    "preferred_gender": gender,
                    "method": method,
                    "relevant_count": len(relevant),
                    "relevant_in_candidate_pool": candidates_relevant,
                    "top_index_ids": "|".join(str(int(value)) for value in ranking[:k]),
                    "top_product_ids": "|".join(
                        str(int(value)) for value in catalog.iloc[ranking[:k]]["id"]
                    ),
                    **metrics.to_dict(),
                }
            )

    details = pd.DataFrame(detail_rows)
    summary = summarize(details, ["method"], seed + 10)
    return details, summary, pd.DataFrame(persona_rows)


def paired_improvement(
    details: pd.DataFrame,
    key: str,
    baseline: str,
    treatment: str,
    cohort: str | None,
    seed: int,
) -> dict:
    selected = details if cohort is None else details[details["cohort"].eq(cohort)]
    pivot = selected.pivot(index=key, columns="method", values="ndcg_at_k").dropna()
    difference = (pivot[treatment] - pivot[baseline]).to_numpy()
    random = np.random.default_rng(seed)
    low, high = bootstrap_mean_interval(difference, random)
    baseline_mean = float(pivot[baseline].mean())
    treatment_mean = float(pivot[treatment].mean())
    return {
        "baseline": baseline,
        "treatment": treatment,
        "cohort": cohort or "overall",
        "observations": len(pivot),
        "baseline_ndcg_at_k": baseline_mean,
        "treatment_ndcg_at_k": treatment_mean,
        "absolute_change": treatment_mean - baseline_mean,
        "relative_change_percent": (
            100.0 * (treatment_mean - baseline_mean) / baseline_mean
            if baseline_mean
            else None
        ),
        "paired_absolute_change_ci95_low": low,
        "paired_absolute_change_ci95_high": high,
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_bar_chart(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    width, height = 900, 480
    left, right, top, bottom = 90, 30, 70, 120
    chart_width, chart_height = width - left - right, height - top - bottom
    maximum = max(values) if values else 1.0
    maximum = max(maximum * 1.15, 0.01)
    bar_width = chart_width / max(len(values), 1) * 0.65
    gap = chart_width / max(len(values), 1)
    colors = ["#2563eb", "#7c3aed", "#059669", "#ea580c", "#dc2626"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="36" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top+chart_height}" x2="{left+chart_width}" y2="{top+chart_height}" stroke="#334155"/>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = top + chart_height - chart_height * tick / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+chart_width}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.2f}</text>')
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + index * gap + (gap - bar_width) / 2
        bar_height = chart_height * value / maximum
        y = top + chart_height - bar_height
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{colors[index % len(colors)]}" rx="3"/>')
        parts.append(f'<text x="{x+bar_width/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="Arial" font-size="13">{value:.3f}</text>')
        parts.append(f'<text x="{x+bar_width/2:.1f}" y="{top+chart_height+24}" text-anchor="middle" font-family="Arial" font-size="12" transform="rotate(18 {x+bar_width/2:.1f} {top+chart_height+24})">{html.escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_report(
    output: Path,
    retrieval_summary: pd.DataFrame,
    personalization_summary: pd.DataFrame,
    comparisons: dict,
    query_count: int,
    persona_count: int,
    mean_candidate_recall: float,
    k: int,
) -> None:
    retrieval_display = retrieval_summary[
        ["cohort", "method", "observations", "precision_at_k", "recall_at_k", "ndcg_at_k", "reciprocal_rank"]
    ]
    personalization_display = personalization_summary[
        ["method", "observations", "precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate_at_k"]
    ]
    full = comparisons["personalization_full_vs_clip"]
    paraphrased = comparisons["retrieval_clip_vs_bm25_paraphrased"]
    overall = comparisons["retrieval_clip_vs_bm25_overall"]
    report = f"""# Offline Evaluation Report

This report is generated by `scripts/evaluate_offline.py` from the prepared
catalog and locally supplied FAISS indexes.

## Protocol

- Ranking cutoff: K={k}
- Retrieval benchmark: {query_count} queries split equally between literal and paraphrased wording.
- Personalization benchmark: {persona_count} simulated preference profiles.
- Ground truth: exact article-type, color, and gender matches in catalog metadata.
- Profile/test separation: profile history products are removed from both the candidate list and held-out relevant set.
- Uncertainty: 95% percentile bootstrap confidence intervals are stored in the summary CSV files.

## Retrieval results

{markdown_table(retrieval_display, retrieval_display.columns.tolist())}

Across all queries, CLIP changed NDCG@{k} by
**{overall['relative_change_percent']:.2f}%** relative to BM25
({overall['baseline_ndcg_at_k']:.4f} → {overall['treatment_ndcg_at_k']:.4f});
the paired absolute-change 95% CI is
[{overall['paired_absolute_change_ci95_low']:.4f}, {overall['paired_absolute_change_ci95_high']:.4f}].

On paraphrased queries, CLIP increased NDCG@{k} by
**{paraphrased['absolute_change']:.4f}**
({paraphrased['baseline_ndcg_at_k']:.4f} → {paraphrased['treatment_ndcg_at_k']:.4f});
the paired 95% CI is
[{paraphrased['paired_absolute_change_ci95_low']:.4f}, {paraphrased['paired_absolute_change_ci95_high']:.4f}].

## Personalization ablation

{markdown_table(personalization_display, personalization_display.columns.tolist())}

The full hybrid reranker increased NDCG@{k} by
**{full['absolute_change']:.4f}** relative to the non-personalized CLIP baseline
({full['baseline_ndcg_at_k']:.4f} → {full['treatment_ndcg_at_k']:.4f});
the paired 95% CI is
[{full['paired_absolute_change_ci95_low']:.4f}, {full['paired_absolute_change_ci95_high']:.4f}].

The 500-item CLIP candidate pool contained an average of
**{mean_candidate_recall:.2%}** of each persona's complete held-out relevant set.
The reranker cannot recover relevant products that are absent from this pool,
so candidate generation remains an important bottleneck.

## Interpretation limits

This is a deterministic simulated-user offline benchmark, not an online A/B
test and not evidence of real-user satisfaction. The Fashion Product Images
dataset contains no genuine user histories. Metadata-derived relevance labels
also favor systems that use those attributes. These results are suitable for
regression testing and architecture comparison; a real interaction dataset or
user study is required before making production claims.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = clean_catalog(args.catalog)
    combinations = eligible_combinations(catalog, args.minimum_relevant)
    if combinations.empty:
        raise SystemExit("No eligible preference combinations; lower --minimum-relevant")

    queries = build_query_benchmark(combinations, args.queries_per_type, args.seed)
    pipeline = ProductSearchPipeline()
    pipeline.load_indexes()
    if int(pipeline.text_index.ntotal) != len(catalog):
        raise SystemExit(
            f"Catalog/index mismatch: catalog={len(catalog)}, index={pipeline.text_index.ntotal}"
        )

    retrieval_details, retrieval_summary = evaluate_retrieval(
        catalog, queries, pipeline, args.k, args.seed
    )
    text_vectors = np.asarray(
        pipeline.text_index.reconstruct_n(0, int(pipeline.text_index.ntotal)),
        dtype="float32",
    )
    personalization_details, personalization_summary, personas = evaluate_personalization(
        catalog,
        combinations,
        pipeline,
        text_vectors,
        args.k,
        args.candidate_pool,
        args.history_size,
        args.seed + 100,
        args.max_personas,
    )

    comparisons = {
        "retrieval_clip_vs_bm25_overall": paired_improvement(
            retrieval_details, "query_id", "BM25", "CLIP", None, args.seed + 200
        ),
        "retrieval_clip_vs_bm25_literal": paired_improvement(
            retrieval_details, "query_id", "BM25", "CLIP", "literal", args.seed + 201
        ),
        "retrieval_clip_vs_bm25_paraphrased": paired_improvement(
            retrieval_details, "query_id", "BM25", "CLIP", "paraphrased", args.seed + 202
        ),
        "personalization_full_vs_clip": paired_improvement(
            personalization_details,
            "persona_id",
            "CLIP",
            "Full Hybrid",
            None,
            args.seed + 203,
        ),
    }

    queries.to_csv(args.output / "query_benchmark.csv", index=False)
    retrieval_details.to_csv(args.output / "retrieval_per_query.csv", index=False)
    retrieval_summary.to_csv(args.output / "retrieval_summary.csv", index=False)
    personas.to_csv(args.output / "synthetic_personas.csv", index=False)
    personalization_details.to_csv(
        args.output / "personalization_per_persona.csv", index=False
    )
    personalization_summary.to_csv(
        args.output / "personalization_summary.csv", index=False
    )
    personalization_by_type = summarize(
        personalization_details,
        ["preferred_article_type", "method"],
        args.seed + 11,
    )
    personalization_by_type.to_csv(
        args.output / "personalization_by_type.csv", index=False
    )
    mean_candidate_recall = float(personas["candidate_recall"].mean())
    summary = {
        "dataset": {
            "catalog_products": len(catalog),
            "query_count": len(queries),
            "simulated_persona_count": len(personas),
        },
        "configuration": {
            "seed": args.seed,
            "k": args.k,
            "minimum_relevant": args.minimum_relevant,
            "history_size": args.history_size,
            "candidate_pool": args.candidate_pool,
            "mean_candidate_recall": mean_candidate_recall,
        },
        "comparisons": comparisons,
        "limitations": [
            "Simulated users, not real interaction histories.",
            "Relevance labels are derived from catalog metadata.",
            "Offline results do not establish online user satisfaction.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    paraphrased_chart = retrieval_summary[
        retrieval_summary["cohort"].eq("paraphrased")
    ]
    write_bar_chart(
        args.output / f"retrieval_ndcg_at_{args.k}.svg",
        f"Paraphrased Retrieval NDCG@{args.k}",
        paraphrased_chart["method"].tolist(),
        paraphrased_chart["ndcg_at_k"].tolist(),
    )
    write_bar_chart(
        args.output / f"personalization_ndcg_at_{args.k}.svg",
        f"Personalization Ablation NDCG@{args.k}",
        personalization_summary["method"].tolist(),
        personalization_summary["ndcg_at_k"].tolist(),
    )
    write_report(
        args.output,
        retrieval_summary,
        personalization_summary,
        comparisons,
        len(queries),
        len(personas),
        mean_candidate_recall,
        args.k,
    )
    print(f"Evaluation complete: {args.output.resolve()}")
    print(json.dumps(comparisons, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
