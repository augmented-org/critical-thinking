"""
Misclassification error analysis for subskill classification results.

Given a results JSON (e.g., llama_results_wrubric_splits/rstate_47/test_results.json)
and the ground-truth labels from combined_essays.xlsx, this script:

- Aggregates misclassification types (e.g., below emerging -> emerging)
- Counts occurrences overall and per subskill
- Extracts common keywords/phrases from the generated justifications for each misclassification type
- Dumps example essay IDs per misclassification category
- Exports detailed CSVs and a concise Markdown report for paper writing

Usage:
    python misclassification_error_analysis.py \
        --results-file llama_results_wrubric_splits/rstate_47/test_results.json \
        --gt-file combined_essays.xlsx \
        --output-dir llama_results_wrubric_splits/rstate_47/error_analysis

Outputs (under --output-dir):
- summary_by_type.csv: counts per (gt -> pred) with per-subskill breakdown
- misclassified_records.csv: row-wise listing with essay, subskill, gt, pred, justification
- per_type/<gt>__to__<pred>.csv: records for each misclassification category
- report.md: human-readable summary with top keywords and examples
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_ground_truth(gt_path_xlsx: str = "combined_essays.xlsx") -> pd.DataFrame:
    """Load ground-truth labels from combined_essays.xlsx.

    Expects columns: 'essay_id', 'Subskill 2.1', 'Subskill 2.2', ..., as in training code.
    """
    if not os.path.exists(gt_path_xlsx):
        raise FileNotFoundError(
            f"Ground truth file not found: {gt_path_xlsx}. Ensure it is present in the workspace."
        )

    gt_df = pd.read_excel(gt_path_xlsx)
    gt_df.columns = gt_df.iloc[0]
    gt_df = gt_df[1:].reset_index(drop=True)
    # Normalize labels to lowercase str
    for col in gt_df.columns:
        if str(col).startswith("Subskill "):
            gt_df[col] = gt_df[col].astype(str).str.strip().str.lower()
    gt_df['essay_id'] = gt_df['essay_id'].astype(str)
    return gt_df


SUBSKILL_MAP = {
    '2.1': '2.1: Synthesizing Multiple Sources',
    '2.2': '2.2: Evaluating Evidence Strength',
    '3.1': '3.1: Using Counterarguments',
    '3.2': '3.2: Using Facts and Opinions',
    '4.1': '4.1: Drawing Conclusions',
    '4.2': '4.2: Using Logical Fallacies',
}


def normalize_label(text: str | None) -> str | None:
    if text is None:
        return None
    t = str(text).strip().lower()
    # Occasionally predictions may contain parsing markers; filter them
    if t in {"few-shot example", "parsing error", "none", "nan"}:
        return None
    return t


def iter_records(pred_results: Dict, gt_df: pd.DataFrame, only_subskills: List[str] | None = None) -> List[Dict]:
    """Join predictions with ground truth into row-wise records.

    Returns rows with: essay_id, subskill, subskill_full, gt, pred, justification
    """
    rows: List[Dict] = []
    # Build quick access GT per essay
    gt_per_essay: Dict[str, Dict[str, str]] = {}
    for _, row in gt_df.iterrows():
        essay_id = str(row['essay_id'])
        gt_per_essay[essay_id] = {}
        for col, val in row.items():
            if isinstance(col, str) and col.startswith('Subskill '):
                gt_per_essay[essay_id][col] = normalize_label(val)

    # Walk over essays that have predictions
    for essay_id, rubric in pred_results.items():
        if essay_id not in gt_per_essay:
            # Essay not in GT split; skip
            continue
        for subskill_short, subskill_full in SUBSKILL_MAP.items():
            if only_subskills is not None and subskill_short not in only_subskills:
                continue
            short_col = f"Subskill {subskill_short}"
            gt_label = gt_per_essay[essay_id].get(short_col)

            pred_obj = rubric.get(subskill_full, {}) if isinstance(rubric, dict) else {}
            pred_label = normalize_label(pred_obj.get('label'))
            justification = pred_obj.get('justification')

            if gt_label is None or pred_label is None:
                continue

            rows.append({
                'essay_id': essay_id,
                'subskill': short_col,
                'subskill_full': subskill_full,
                'ground_truth': gt_label,
                'predicted': pred_label,
                'justification': justification,
            })
    return rows


def top_keywords(texts: List[str], top_k: int = 15) -> List[Tuple[str, int]]:
    """Compute simple keyword frequencies as a quick, dependency-light proxy for insights.

    - Lowercases
    - Removes punctuation and numbers
    - Splits on whitespace
    - Removes short tokens and common stopwords
    - Returns top_k most frequent unigrams
    """
    STOPWORDS = {
        'the','a','an','and','or','of','to','in','on','for','with','by','as','at','from','this','that','it','is','are',
        'was','were','be','been','being','so','we','they','their','there','our','i','you','he','she','his','her','them',
        'but','not','no','yes','do','does','did','have','has','had','having','into','about','over','more','most','much',
        'many','few','some','any','all','also','can','could','should','would','may','might','than','then','because',
        'such','these','those','however','therefore','thus','which','who','whom','what','when','where','why','how',
        'just','very','like','one','two','three','first','second','third','etc','e.g','eg','ie','i.e','s','t','ll','re',
        'if','while','though','although','both','each','per','vs','vs.'
    }
    counter: Dict[str, int] = collections.Counter()
    for txt in texts:
        if not isinstance(txt, str) or not txt:
            continue
        t = txt.lower()
        t = re.sub(r"[^a-z\s]", " ", t)
        tokens = [w for w in t.split() if len(w) > 2 and w not in STOPWORDS]
        counter.update(tokens)
    return counter.most_common(top_k)


def generate_report(output_dir: str,
                    summary_df: pd.DataFrame,
                    per_type_examples: Dict[Tuple[str, str], List[str]],
                    per_type_keywords: Dict[Tuple[str, str], List[Tuple[str, int]]],
                    min_count: int) -> None:
    """Write a concise Markdown report with counts, keywords, and examples."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'report.md')

    lines: List[str] = []
    lines.append(f"# Misclassification Error Analysis\n")
    lines.append(f"Min count filter: {min_count}\n")

    if summary_df.empty:
        lines.append("No misclassifications found given the filters.")
    else:
        lines.append("## Summary by Type (GT -> Pred)\n")
        # Sorted summary
        tmp = summary_df.sort_values(['count_total'], ascending=False)
        lines.append("| GT | Pred | Total | Per-Subskill Counts |")
        lines.append("|---|---:|---:|---|")
        for _, row in tmp.iterrows():
            lines.append(
                f"| {row['ground_truth']} | {row['predicted']} | {row['count_total']} | {row['per_subskill']} |"
            )

        lines.append("\n## Category Insights\n")
        for (gt, pred), kw in per_type_keywords.items():
            ex_ids = per_type_examples.get((gt, pred), [])
            lines.append(f"### {gt} → {pred}\n")
            lines.append(f"Top keywords (from justifications): ")
            if kw:
                kws = ", ".join([f"{w} ({c})" for w, c in kw])
            else:
                kws = "<none>"
            lines.append(f"{kws}\n")
            if ex_ids:
                lines.append(f"Example essay IDs: {', '.join(ex_ids[:10])}\n")
            lines.append("")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Misclassification error analysis with justification insights")
    parser.add_argument("--results-file", required=True, help="Path to test_results.json")
    parser.add_argument("--gt-file", default="combined_essays.xlsx", help="Path to ground truth Excel (combined_essays.xlsx)")
    parser.add_argument("--output-dir", default=None, help="Directory to write outputs; default alongside results")
    parser.add_argument("--min-count", type=int, default=5, help="Only include categories with at least this many errors")
    parser.add_argument("--examples-per-type", type=int, default=5, help="Number of example essay IDs per category")
    parser.add_argument("--only-subskill", default=None, help="Restrict analysis to a subskill short code (e.g., '4.2')")
    args = parser.parse_args()

    results_path = args.results_file
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    # Load predictions JSON
    with open(results_path, 'r', encoding='utf-8') as f:
        pred_results = json.load(f)

    # Load GT
    gt_df = load_ground_truth(args.gt_file)

    # Join predictions with GT
    only_subskills = None
    if args.only_subskill is not None:
        code = args.only_subskill.strip()
        if code not in SUBSKILL_MAP:
            raise ValueError(f"Unknown subskill '{code}'. Available: {', '.join(sorted(SUBSKILL_MAP.keys()))}")
        only_subskills = [code]

    records = iter_records(pred_results, gt_df, only_subskills=only_subskills)
    if not records:
        print("No matching records between predictions and ground truth. Nothing to analyze.")
        return

    df = pd.DataFrame(records)
    # Identify misclassifications
    mis = df[df['ground_truth'] != df['predicted']].copy()
    if mis.empty:
        print("Model is perfectly accurate for the provided split. No misclassifications found.")
        return

    # Save all misclassified rows
    out_dir = args.output_dir
    if out_dir is None:
        base_dir = os.path.dirname(results_path)
        out_dir = os.path.join(base_dir, 'error_analysis')
    os.makedirs(out_dir, exist_ok=True)

    all_mis_path = os.path.join(out_dir, 'misclassified_records.csv')
    mis.to_csv(all_mis_path, index=False)

    # Aggregate counts per (gt, pred)
    group_cols = ['ground_truth', 'predicted']
    counts = mis.groupby(group_cols).size().reset_index(name='count_total')

    # Per-subskill counts for each (gt,pred)
    per_subskill = (
        mis.groupby(group_cols + ['subskill'])
           .size()
           .reset_index(name='count')
    )

    # Merge summary as a nested dict string for readability
    per_type_dict: Dict[Tuple[str, str], Dict[str, int]] = {}
    for _, r in per_subskill.iterrows():
        key = (r['ground_truth'], r['predicted'])
        per_type_dict.setdefault(key, {})[r['subskill']] = int(r['count'])

    counts['per_subskill'] = counts.apply(
        lambda r: {k: v for k, v in sorted(per_type_dict.get((r['ground_truth'], r['predicted']), {}).items())},
        axis=1
    )

    # Filter by min-count
    counts_f = counts[counts['count_total'] >= args.min_count].sort_values('count_total', ascending=False)

    # Save summary
    summary_path = os.path.join(out_dir, 'summary_by_type.csv')
    counts_f.to_csv(summary_path, index=False)

    # Prepare insights per category
    per_type_examples: Dict[Tuple[str, str], List[str]] = {}
    per_type_keywords: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}

    per_type_dir = os.path.join(out_dir, 'per_type')
    os.makedirs(per_type_dir, exist_ok=True)

    for _, r in counts_f.iterrows():
        gt = r['ground_truth']
        pred = r['predicted']
        key = (gt, pred)
        subset = mis[(mis['ground_truth'] == gt) & (mis['predicted'] == pred)].copy()

        # Save per-type CSV
        safe_gt = re.sub(r"[^a-z0-9]+", "_", gt)
        safe_pred = re.sub(r"[^a-z0-9]+", "_", pred)
        per_type_path = os.path.join(per_type_dir, f"{safe_gt}__to__{safe_pred}.csv")
        subset.to_csv(per_type_path, index=False)

        # Collect example essay IDs
        per_type_examples[key] = list(subset['essay_id'].head(args.examples_per_type))

        # Extract keywords from justifications
        just_texts = [t for t in subset['justification'].tolist() if isinstance(t, str) and t.strip()]
        per_type_keywords[key] = top_keywords(just_texts, top_k=15)

    # Write concise Markdown report
    generate_report(out_dir, counts_f, per_type_examples, per_type_keywords, args.min_count)

    print("\nAnalysis complete.")
    print(f"- Summary: {summary_path}")
    print(f"- All misclassified records: {all_mis_path}")
    print(f"- Per-type CSVs: {per_type_dir}")
    print(f"- Report: {os.path.join(out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
