"""
Harness EDA Advisor — Master Analysis Orchestrator
Observe. Assess. Explain. Recommend. Document.
The Harness NEVER modifies data.
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from .column_classifier import classify_columns
from .quality_checker import run_quality_checks
from .claims_detector import detect_claims
from .target_discoverer import discover_targets
from .feature_scorer import score_features
from .roadmap_generator import generate_roadmap
from .findings_generator import generate_findings
from .meta_extractor import extract_meta

logger = logging.getLogger(__name__)


def load_dataframe(file_path: Path) -> pd.DataFrame:
    """Load any supported format into a DataFrame — read only."""
    ext = file_path.suffix.lower()
    logger.info(f"Loading {ext} file: {file_path}")
    if ext == ".csv":
        # Try common encodings; fall back gracefully
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                return pd.read_csv(file_path, encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode CSV file with common encodings.")
    elif ext in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, engine="openpyxl" if ext == ".xlsx" else "xlrd")
    elif ext == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def safe_val(v):
    """Convert numpy types to Python-native for JSON serialisation."""
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return None if np.isnan(v) else float(v)
    if isinstance(v, (np.ndarray,)): return v.tolist()
    if isinstance(v, pd.Timestamp): return str(v)
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    return v


def jsonify(obj):
    """Recursively make an object JSON-safe."""
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonify(i) for i in obj]
    return safe_val(obj)


def run_analysis(file_path: Path) -> dict:
    """
    Full pipeline. Returns a dict matching the 10-page frontend schema.
    """
    df = load_dataframe(file_path)
    logger.info(f"Loaded dataframe: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── 1. Meta ────────────────────────────────────────────────────────────────
    meta = extract_meta(df, file_path)

    # ── 2. Column Classification ───────────────────────────────────────────────
    anatomy = classify_columns(df)

    # ── 3. Quality Checks ─────────────────────────────────────────────────────
    quality = run_quality_checks(df, anatomy)

    # ── 4. Coverage ───────────────────────────────────────────────────────────
    coverage = build_coverage(df, anatomy)

    # ── 5. Claims Detection ───────────────────────────────────────────────────
    claims = detect_claims(df, anatomy)

    # ── 6. Target Discovery ───────────────────────────────────────────────────
    targets = discover_targets(df, anatomy, claims)

    # ── 7. Feature Scoring ────────────────────────────────────────────────────
    features = score_features(df, anatomy)

    # ── 8. Readiness Score ────────────────────────────────────────────────────
    readiness_score, readiness_breakdown, readiness_summary = compute_readiness(
        df, anatomy, quality, claims, targets
    )

    # ── 9. EDA Roadmap ────────────────────────────────────────────────────────
    roadmap, roadmap_summary = generate_roadmap(df, anatomy, claims, targets, quality)

    # ── 10. Findings ──────────────────────────────────────────────────────────
    findings = generate_findings(df, anatomy, quality, claims, targets, features)

    # ── Interpretation ────────────────────────────────────────────────────────
    interpretation = build_interpretation(df, anatomy, claims, targets)

    result = {
        "meta": meta,
        "interpretation": interpretation,
        "anatomy": anatomy,
        "quality": quality,
        "coverage": coverage,
        "claims": claims,
        "targets": targets,
        "features": features,
        "readiness_score": readiness_score,
        "readiness_breakdown": readiness_breakdown,
        "readiness_summary": readiness_summary,
        "roadmap": roadmap,
        "roadmap_summary": roadmap_summary,
        "findings": findings,
    }

    return jsonify(result)


def build_coverage(df: pd.DataFrame, anatomy: list) -> dict:
    """Build coverage intelligence including heatmaps for categorical columns."""
    coverage_pcts = [c["coverage_pct"] for c in anatomy]
    avg_cov = round(float(np.mean(coverage_pcts)), 1) if coverage_pcts else 0
    fully_populated = sum(1 for x in coverage_pcts if x >= 99)
    sparse = sum(1 for x in coverage_pcts if x < 30)

    heatmaps = []
    # Look for geography, product, plan, breed, province type columns
    cat_targets = ["Geography", "Product", "Pet", "Customer", "Policy"]
    for cat in cat_targets:
        cat_cols = [c for c in anatomy if c["category"] == cat and c["dtype"] == "object"]
        for col_info in cat_cols[:1]:  # one heatmap per category
            col = col_info["name"]
            if col in df.columns:
                vc = df[col].dropna().value_counts(normalize=True) * 100
                cells = [{"label": str(k)[:20], "pct": round(float(v), 1)} for k, v in vc.head(20).items()]
                if cells:
                    heatmaps.append({"title": f"{col} Coverage Distribution", "cells": cells})

    return {
        "avg_coverage": avg_cov,
        "fully_populated": fully_populated,
        "sparse_columns": sparse,
        "heatmaps": heatmaps,
    }


def compute_readiness(df, anatomy, quality, claims, targets) -> tuple:
    """Score EDA readiness 0–100 across 4 dimensions of 25 points each."""
    # Dimension 1: Data Completeness (25)
    avg_cov = np.mean([c["coverage_pct"] for c in anatomy]) if anatomy else 0
    d1 = min(25, round(avg_cov / 4))

    # Dimension 2: Data Quality (25)
    checks = quality.get("checks", [])
    if checks:
        pass_rate = sum(1 for c in checks if c["status"] == "pass") / len(checks)
        d2 = min(25, round(pass_rate * 25))
    else:
        d2 = 12

    # Dimension 3: Target Availability (25)
    if targets and len(targets) >= 2:
        d3 = 25
    elif targets and len(targets) == 1:
        d3 = 15
    else:
        d3 = 5

    # Dimension 4: Structural Richness (25)
    n_categories = len(set(c["category"] for c in anatomy))
    d4 = min(25, n_categories * 3)

    total = d1 + d2 + d3 + d4
    breakdown = {"Completeness": d1, "Quality": d2, "Target Availability": d3, "Structural Richness": d4}

    if total >= 80:
        summary = "Dataset is well-structured and ready for EDA. Most checks passed. Targets identified."
    elif total >= 60:
        summary = "Dataset is usable but has quality gaps. Address warnings before modelling."
    elif total >= 40:
        summary = "Dataset has significant quality issues. Analyst review required before proceeding."
    else:
        summary = "Dataset requires substantial preparation. Multiple critical issues detected."

    return total, breakdown, summary


def build_interpretation(df, anatomy, claims, targets) -> str:
    """Generate a plain-English senior-DS interpretation of the dataset."""
    n_rows, n_cols = df.shape
    cats = list(set(c["category"] for c in anatomy))
    has_claims = claims is not None
    has_targets = bool(targets)

    parts = [
        f"This dataset contains {n_rows:,} records across {n_cols} columns.",
        f"Column analysis identified {len(cats)} distinct field categories: {', '.join(cats)}.",
    ]
    if has_claims:
        parts.append(f"Claims fields were detected — {claims.get('claiming_pct', '?')}% of records carry a claim.")
    if has_targets:
        parts.append(f"{len(targets)} candidate modelling target(s) were identified including: {', '.join(t['name'] for t in targets[:3])}.")
    parts.append(
        "Review the Dataset Anatomy, Quality Assessment, and EDA Advisor sections for a full readiness picture before beginning exploratory analysis."
    )
    return " ".join(parts)
