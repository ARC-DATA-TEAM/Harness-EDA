"""
Column Classifier — auto-categorise every column in the dataset.
Categories: Identifiers, Dates, Claims, Exposure, Premium, Customer,
            Policy, Product, Pet, Geography, Behavioural,
            Potential Targets, Potential Leakage, Other
"""
import re
import pandas as pd
import numpy as np

CATEGORY_PATTERNS = [
    ("Identifiers",      r"_id$|^id$|_no$|_num$|_ref$|_code$|identifier|serial|uuid|sku"),
    ("Dates",            r"date|_dt$|^dt_|year|month|quarter|inception|expiry|renewal|start|end"),
    ("Claims",           r"claim|clm|loss|paid|payment|reimburs|benefit|settlement|indemnity"),
    ("Exposure",         r"exposure|expo|earned|duration|days_at_risk|risk_days|in_force|lapse"),
    ("Premium",          r"premium|prm|gwp|nwp|written|price|rate|tariff|contribution"),
    ("Customer",         r"customer|client|member|insured|policyholder|holder"),
    ("Policy",           r"policy|cover|coverage|plan_type|product_code|scheme"),
    ("Product",          r"product|line_of_business|lob|class|subclass|tier"),
    ("Pet",              r"pet|breed|species|animal|cat\b|dog\b|rabbit"),
    ("Geography",        r"province|state|region|country|postcode|zip|city|area|territory|district"),
    ("Behavioural",      r"tenure|vintage|age|gender|sex|occupation|income|channel|source"),
    ("Potential Leakage",r"future|next_|post_claim|after_claim|outcome"),
    ("Potential Targets", r"target|label|y_|flag|indicator|propensity"),
]

LEAKAGE_SIGNALS = re.compile(r"future|post_claim|outcome|next_period|after_loss", re.I)
LOW_VARIANCE_THRESHOLD = 0.98    # fraction that must be the same value
SPARSE_THRESHOLD = 30            # coverage % below which column is sparse


def classify_columns(df: pd.DataFrame) -> list:
    results = []
    for col in df.columns:
        col_lower = col.lower().replace(" ", "_")
        series = df[col]

        # ── dtype ──────────────────────────────────────────────────────────────
        dtype = infer_dtype(series)

        # ── coverage ───────────────────────────────────────────────────────────
        coverage_pct = round(float(series.notna().mean() * 100), 1)

        # ── unique values ──────────────────────────────────────────────────────
        nunique = int(series.nunique())

        # ── sample values (non-null, up to 5) ─────────────────────────────────
        sample_vals = series.dropna().head(5).astype(str).tolist()

        # ── category ───────────────────────────────────────────────────────────
        category = "Other"
        for cat, pattern in CATEGORY_PATTERNS:
            if re.search(pattern, col_lower, re.I):
                category = cat
                break

        # ── leakage override ───────────────────────────────────────────────────
        if LEAKAGE_SIGNALS.search(col_lower):
            category = "Potential Leakage"

        results.append({
            "name": col,
            "category": category,
            "dtype": dtype,
            "coverage_pct": coverage_pct,
            "unique_values": nunique,
            "sample_values": sample_vals,
        })

    return results


def infer_dtype(series: pd.Series) -> str:
    """Return a human-friendly dtype string."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    # Try to detect date strings
    if series.dtype == object:
        sample = series.dropna().head(20)
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() > len(sample) * 0.7:
                return "datetime (string)"
        except Exception:
            pass
        return "object"
    return str(series.dtype)
