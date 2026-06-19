"""Extract high-level metadata from the dataframe."""
import re
from pathlib import Path
import pandas as pd
import numpy as np


def extract_meta(df: pd.DataFrame, file_path: Path) -> dict:
    size_bytes = file_path.stat().st_size
    size_str = _fmt_size(size_bytes)

    # Date range
    date_range = _find_date_range(df)

    # Unique entity counts
    policy_col = _find_col(df, r"policy.?(id|no|num|number|ref)", exact=["policy_id", "policy_no", "policyid"])
    cust_col = _find_col(df, r"cust.?(id|no|num|ref)|customer.?id|client.?id", exact=["customer_id", "client_id"])

    # Claiming entities
    claim_cols = [c for c in df.columns if re.search(r"claim.?(amount|paid|cost|value|cnt|count)", c, re.I)]
    claiming_entities = None
    claim_rate = None
    if claim_cols:
        cc = claim_cols[0]
        if df[cc].dtype in [np.float64, np.int64, float, int]:
            has_claim = df[cc].notna() & (df[cc] > 0)
        else:
            has_claim = df[cc].notna() & (df[cc] != 0) & (df[cc] != "")
        claiming_entities = int(has_claim.sum())
        claim_rate = round(float(has_claim.mean() * 100), 2)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "file_size": size_str,
        "file_type": file_path.suffix.lstrip(".").upper(),
        "date_range": date_range,
        "unique_policies": int(df[policy_col].nunique()) if policy_col else None,
        "unique_customers": int(df[cust_col].nunique()) if cust_col else None,
        "unique_risk_items": None,
        "claiming_entities": claiming_entities,
        "claim_rate": claim_rate,
    }


def _fmt_size(b: int) -> str:
    if b >= 1_073_741_824: return f"{b/1_073_741_824:.1f} GB"
    if b >= 1_048_576: return f"{b/1_048_576:.1f} MB"
    if b >= 1024: return f"{b/1024:.1f} KB"
    return f"{b} B"


def _find_date_range(df: pd.DataFrame) -> str:
    """Find the min–max date across all date-like columns."""
    date_cols = []
    for col in df.columns:
        if re.search(r"date|dt|year|month", col, re.I):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > df.shape[0] * 0.3:
                    date_cols.append(parsed)
            except Exception:
                pass
    if not date_cols:
        return None
    all_dates = pd.concat(date_cols, ignore_index=True).dropna()
    if all_dates.empty:
        return None
    return f"{all_dates.min().strftime('%Y-%m-%d')} → {all_dates.max().strftime('%Y-%m-%d')}"


def _find_col(df: pd.DataFrame, pattern: str, exact: list = None) -> str | None:
    """Find a column by exact name match first, then regex."""
    if exact:
        for name in exact:
            for col in df.columns:
                if col.lower().replace(" ", "_") == name:
                    return col
    for col in df.columns:
        if re.search(pattern, col, re.I):
            return col
    return None
