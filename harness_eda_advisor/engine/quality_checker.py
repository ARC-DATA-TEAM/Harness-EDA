"""
Data Quality Checker — observe and assess only. Never modifies data.
Checks:
  1. Missing Values
  2. Duplicate Rows
  3. Duplicate Columns
  4. Zero Variance Variables
  5. Sparse Variables
  6. Date Integrity
  7. Outliers (IQR method)
  8. Unexpected Null Patterns
  9. Mixed Types
  10. Cardinality Extremes
  11. Negative Values in positive-expected fields
  12. Population Coverage Summary
"""
import re
import pandas as pd
import numpy as np


def run_quality_checks(df: pd.DataFrame, anatomy: list) -> dict:
    checks = []

    # 1. Missing Values
    checks.append(_check_missing(df))

    # 2. Duplicate Rows
    checks.append(_check_dup_rows(df))

    # 3. Duplicate Columns (by content)
    checks.append(_check_dup_cols(df))

    # 4. Zero Variance
    checks.append(_check_zero_variance(df))

    # 5. Sparse Variables
    checks.append(_check_sparse(df))

    # 6. Date Integrity
    checks.append(_check_date_integrity(df, anatomy))

    # 7. Outliers
    checks.append(_check_outliers(df, anatomy))

    # 8. Unexpected Nulls (nulls in numeric claimed-complete columns)
    checks.append(_check_unexpected_nulls(df, anatomy))

    # 9. Mixed Types
    checks.append(_check_mixed_types(df))

    # 10. High Cardinality Categoricals
    checks.append(_check_cardinality(df, anatomy))

    # 11. Negative Values in premium/exposure columns
    checks.append(_check_negatives(df, anatomy))

    # 12. Overall coverage
    checks.append(_check_coverage_summary(df))

    return {"checks": checks}


# ── Individual checks ──────────────────────────────────────────────────────────

def _check_missing(df):
    missing_pct = (df.isna().sum().sum() / df.size) * 100
    n_cols_missing = (df.isna().sum() > 0).sum()
    if missing_pct < 5:
        status = "pass"
    elif missing_pct < 20:
        status = "warn"
    else:
        status = "fail"
    return {
        "check": "Missing Values",
        "status": status,
        "observation": f"{missing_pct:.1f}% of all cells are missing across {n_cols_missing} columns.",
        "why_matters": "High missingness reduces model reliability and may indicate data pipeline failures or collection gaps.",
        "impact": "Biased analysis, reduced sample sizes, misleading averages.",
        "action": f"Investigate the {n_cols_missing} columns with missing values. Prioritise columns with > 30% missingness."
    }


def _check_dup_rows(df):
    dup_count = int(df.duplicated().sum())
    pct = round(dup_count / len(df) * 100, 2) if len(df) > 0 else 0
    status = "pass" if dup_count == 0 else ("warn" if pct < 5 else "fail")
    return {
        "check": "Duplicate Rows",
        "status": status,
        "observation": f"{dup_count:,} duplicate rows detected ({pct}% of dataset).",
        "why_matters": "Duplicates inflate counts, distort frequencies, and can cause data leakage in train/test splits.",
        "impact": "Overstated claims counts, inflated portfolio metrics.",
        "action": "Review duplicate records. Determine if duplicates are intentional (e.g., multiple coverages per policy) or data errors."
    }


def _check_dup_cols(df):
    seen = {}
    dups = []
    for col in df.columns:
        h = tuple(df[col].fillna("__NA__").head(200).astype(str))
        if h in seen:
            dups.append((col, seen[h]))
        else:
            seen[h] = col
    status = "pass" if not dups else "warn"
    obs = f"{len(dups)} potentially duplicate column pairs detected." if dups else "No duplicate columns detected."
    return {
        "check": "Duplicate Columns",
        "status": status,
        "observation": obs,
        "why_matters": "Duplicate columns add noise and can cause multicollinearity in models.",
        "impact": "Redundant features, inflated feature importance scores.",
        "action": "Review: " + "; ".join([f"'{a}' ≈ '{b}'" for a, b in dups[:5]]) if dups else "No action required."
    }


def _check_zero_variance(df):
    zv_cols = [c for c in df.select_dtypes(include=np.number).columns if df[c].nunique() <= 1]
    str_zv = [c for c in df.select_dtypes(include="object").columns if df[c].nunique() <= 1]
    all_zv = zv_cols + str_zv
    status = "pass" if not all_zv else "warn"
    return {
        "check": "Zero Variance Variables",
        "status": status,
        "observation": f"{len(all_zv)} zero/single-value columns detected: {', '.join(all_zv[:5]) or 'none'}.",
        "why_matters": "Constant columns carry no predictive information and waste memory.",
        "impact": "Model clutter, longer training times, potential errors in some algorithms.",
        "action": "Exclude these columns from feature sets: " + (", ".join(all_zv[:10]) if all_zv else "N/A")
    }


def _check_sparse(df):
    sparse = [(c, round(df[c].notna().mean() * 100, 1)) for c in df.columns if df[c].notna().mean() < 0.30]
    status = "pass" if not sparse else ("warn" if len(sparse) < 5 else "fail")
    obs = f"{len(sparse)} columns are sparse (< 30% populated)." if sparse else "No sparse columns detected."
    return {
        "check": "Sparse Variables",
        "status": status,
        "observation": obs,
        "why_matters": "Sparse columns have insufficient data for reliable analysis or modelling.",
        "impact": "Unreliable statistics, biased models, poor segment coverage.",
        "action": ("Review: " + ", ".join([f"{c} ({p}%)" for c, p in sparse[:8]])) if sparse else "No action required."
    }


def _check_date_integrity(df, anatomy):
    date_cols = [c["name"] for c in anatomy if "date" in c["category"].lower() or "date" in c["dtype"].lower()]
    issues = []
    for col in date_cols[:5]:
        if col not in df.columns: continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        null_cnt = parsed.isna().sum()
        if null_cnt > 0:
            issues.append(f"{col}: {null_cnt} unparseable dates")
        # future date check
        try:
            future = (parsed > pd.Timestamp.today()).sum()
            if future > 0:
                issues.append(f"{col}: {future} future dates")
        except Exception:
            pass
    status = "pass" if not issues else "warn"
    return {
        "check": "Date Integrity",
        "status": status,
        "observation": "; ".join(issues) if issues else "Date columns appear structurally consistent.",
        "why_matters": "Bad dates cause incorrect duration calculations, wrong experience periods, and modelling errors.",
        "impact": "Wrong exposure denominators, incorrect trend analysis, invalid date-based features.",
        "action": "Validate date columns: " + ("; ".join(issues[:3]) if issues else "No action required.")
    }


def _check_outliers(df, anatomy):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    outlier_summary = []
    for col in numeric_cols[:20]:
        q1, q3 = df[col].quantile([0.25, 0.75]).values
        iqr = q3 - q1
        if iqr == 0: continue
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_out > 0:
            outlier_summary.append(f"{col}: {n_out} outliers")
    status = "pass" if not outlier_summary else ("warn" if len(outlier_summary) < 5 else "fail")
    return {
        "check": "Outliers (IQR Method)",
        "status": status,
        "observation": (f"{len(outlier_summary)} columns contain outliers (3×IQR): " + "; ".join(outlier_summary[:5])) if outlier_summary else "No significant outliers detected under 3×IQR threshold.",
        "why_matters": "Extreme outliers distort means, inflate variance, and can dominate model loss functions.",
        "impact": "Misleading averages, unstable models, inflated loss ratios.",
        "action": "Investigate outlier records in flagged columns. Determine if values are data errors or genuine extremes."
    }


def _check_unexpected_nulls(df, anatomy):
    # Look for columns expected to be complete (IDs, key dates)
    key_cats = ["Identifiers", "Dates", "Exposure"]
    issues = []
    for c in anatomy:
        if c["category"] in key_cats and c["coverage_pct"] < 95:
            issues.append(f"{c['name']} ({c['coverage_pct']}% coverage)")
    status = "pass" if not issues else "warn"
    return {
        "check": "Unexpected Null Patterns",
        "status": status,
        "observation": (f"{len(issues)} key columns have unexpected nulls: " + "; ".join(issues[:5])) if issues else "Key columns are well-populated.",
        "why_matters": "Nulls in identifier or date columns suggest upstream data quality failures.",
        "impact": "Unable to link records, broken policy history, incorrect exposure calculations.",
        "action": "Trace nulls in: " + ("; ".join(issues[:5]) if issues else "N/A") + " back to source system."
    }


def _check_mixed_types(df):
    issues = []
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(200)
        try:
            numeric_count = pd.to_numeric(sample, errors="coerce").notna().sum()
            if 0 < numeric_count < len(sample) * 0.9:
                issues.append(col)
        except Exception:
            pass
    status = "pass" if not issues else "warn"
    return {
        "check": "Mixed Data Types",
        "status": status,
        "observation": f"{len(issues)} columns appear to contain mixed numeric and text values." if issues else "No mixed-type columns detected.",
        "why_matters": "Mixed types prevent numeric analysis and cause silent errors in aggregations.",
        "impact": "Wrong aggregations, failed type coercions, model input errors.",
        "action": ("Review columns: " + ", ".join(issues[:8])) if issues else "No action required."
    }


def _check_cardinality(df, anatomy):
    cat_cols = [c for c in anatomy if c["dtype"] == "object" and c["category"] not in ("Identifiers",)]
    high_card = [c for c in cat_cols if c["unique_values"] > 100]
    status = "pass" if not high_card else "warn"
    return {
        "check": "High Cardinality Categoricals",
        "status": status,
        "observation": f"{len(high_card)} categorical columns have > 100 unique values." if high_card else "Categorical cardinality is within expected bounds.",
        "why_matters": "High cardinality categoricals are difficult to encode and may be quasi-identifiers.",
        "impact": "Poor one-hot encoding, model memory issues, potential re-identification risk.",
        "action": ("Consider grouping or target-encoding: " + ", ".join([c["name"] for c in high_card[:5]])) if high_card else "No action required."
    }


def _check_negatives(df, anatomy):
    pos_expected = [c for c in anatomy if c["category"] in ("Claims", "Exposure", "Premium")]
    issues = []
    for c in pos_expected:
        col = c["name"]
        if col in df.columns and df[col].dtype in [np.float64, np.int64, float, int]:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                issues.append(f"{col}: {neg_count} negative values")
    status = "pass" if not issues else "warn"
    return {
        "check": "Negative Values in Positive-Expected Fields",
        "status": status,
        "observation": (f"Negative values found in {len(issues)} financial fields: " + "; ".join(issues[:5])) if issues else "No unexpected negatives in financial fields.",
        "why_matters": "Negative claims or premium amounts typically indicate adjustments, cancellations, or data errors.",
        "impact": "Incorrect loss ratios, distorted frequency/severity calculations.",
        "action": ("Verify business rules for: " + "; ".join(issues[:5])) if issues else "No action required."
    }


def _check_coverage_summary(df):
    pcts = [df[c].notna().mean() * 100 for c in df.columns]
    avg = np.mean(pcts)
    below_50 = sum(1 for p in pcts if p < 50)
    status = "pass" if avg >= 85 else ("warn" if avg >= 60 else "fail")
    return {
        "check": "Overall Population Coverage",
        "status": status,
        "observation": f"Average column coverage: {avg:.1f}%. {below_50} columns are less than 50% populated.",
        "why_matters": "Low overall coverage reduces dataset utility and analytical credibility.",
        "impact": "Insufficient sample sizes, unreliable segment-level statistics.",
        "action": "Focus EDA on well-populated segments. Flag sparse columns for analyst review before modelling."
    }
