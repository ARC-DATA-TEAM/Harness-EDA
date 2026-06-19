"""
Claims Detector — automatically find and summarise claims fields.
Produces: total_claims, claiming_pct, avg/median/max, histogram, percentiles, skewness.
"""
import re
import pandas as pd
import numpy as np


CLAIM_AMOUNT_PATTERNS = re.compile(
    r"claim.?(amount|paid|value|cost|payment|benefit|settlement|indemnity)|"
    r"paid.?amount|loss.?amount|benefit.?amount|total.?paid|amount.?paid|incurred",
    re.I
)
CLAIM_COUNT_PATTERNS = re.compile(
    r"claim.?(count|cnt|num|number|freq|flag)|n_claims|num_claims|claims_count|clm_cnt",
    re.I
)
CLAIM_FLAG_PATTERNS = re.compile(
    r"has_claim|claim_flag|claimed|is_claim|clm_flag|claim_indicator",
    re.I
)


def detect_claims(df: pd.DataFrame, anatomy: list) -> dict | None:
    """Return claims summary or None if no claims fields detected."""
    # Find amount column
    amount_col = None
    for col in df.columns:
        if CLAIM_AMOUNT_PATTERNS.search(col):
            if pd.api.types.is_numeric_dtype(df[col]):
                amount_col = col
                break

    # Find flag column
    flag_col = None
    for col in df.columns:
        if CLAIM_FLAG_PATTERNS.search(col):
            flag_col = col
            break

    # Find count column
    count_col = None
    for col in df.columns:
        if CLAIM_COUNT_PATTERNS.search(col):
            if pd.api.types.is_numeric_dtype(df[col]):
                count_col = col
                break

    # All detected fields
    detected = list({c for c in [amount_col, flag_col, count_col] if c})
    if not detected:
        return None

    n = len(df)
    result = {"detected_fields": detected}

    # Determine claiming records
    if amount_col:
        has_claim = df[amount_col].notna() & (df[amount_col] > 0)
    elif flag_col:
        has_claim = df[flag_col].astype(str).str.lower().isin(["1", "true", "yes", "y"])
    elif count_col:
        has_claim = df[count_col].notna() & (df[count_col] > 0)
    else:
        has_claim = pd.Series([False] * n)

    total_claims = int(has_claim.sum())
    claiming_pct = round(float(has_claim.mean() * 100), 2)

    result["total_claims"] = total_claims
    result["claiming_pct"] = claiming_pct

    # Amount stats
    if amount_col:
        claim_vals = df.loc[has_claim, amount_col].dropna()
        result["avg_claim"] = float(claim_vals.mean()) if len(claim_vals) else None
        result["median_claim"] = float(claim_vals.median()) if len(claim_vals) else None
        result["max_claim"] = float(claim_vals.max()) if len(claim_vals) else None
        result["total_paid"] = float(claim_vals.sum()) if len(claim_vals) else None

        # Skewness
        if len(claim_vals) > 3:
            result["skewness"] = float(claim_vals.skew())
        else:
            result["skewness"] = None

        # Percentiles
        if len(claim_vals) > 0:
            pcts = claim_vals.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
            result["percentiles"] = {
                "P25": float(pcts[0.25]),
                "P50 (Median)": float(pcts[0.50]),
                "P75": float(pcts[0.75]),
                "P90": float(pcts[0.90]),
                "P95": float(pcts[0.95]),
                "P99": float(pcts[0.99]),
            }

        # Histogram (20 bins, log-friendly)
        if len(claim_vals) > 10:
            pos_vals = claim_vals[claim_vals > 0]
            if len(pos_vals) > 10:
                counts, bin_edges = np.histogram(pos_vals, bins=20)
                labels = [f"{_fmt(bin_edges[i])}–{_fmt(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
                result["histogram_data"] = {
                    "labels": labels,
                    "values": counts.tolist()
                }
    else:
        result["avg_claim"] = None
        result["median_claim"] = None
        result["max_claim"] = None
        result["total_paid"] = None
        result["skewness"] = None
        result["percentiles"] = None
        result["histogram_data"] = None

    return result


def _fmt(v):
    if v >= 1_000_000: return f"${v/1_000_000:.1f}M"
    if v >= 1_000: return f"${v/1_000:.0f}K"
    return f"${v:.0f}"
