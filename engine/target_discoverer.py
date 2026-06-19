"""
Target Discoverer — identify candidate modelling targets.
The Harness SUGGESTS. The analyst DECIDES.
"""
import re
import pandas as pd
import numpy as np


def discover_targets(df: pd.DataFrame, anatomy: list, claims: dict | None) -> list:
    targets = []

    # ── 1. Frequency target ────────────────────────────────────────────────────
    freq_target = _find_frequency_target(df, anatomy)
    if freq_target:
        targets.append(freq_target)

    # ── 2. Severity target ─────────────────────────────────────────────────────
    sev_target = _find_severity_target(df, anatomy, claims)
    if sev_target:
        targets.append(sev_target)

    # ── 3. Expected Loss ───────────────────────────────────────────────────────
    el_target = _find_expected_loss_target(df, anatomy)
    if el_target:
        targets.append(el_target)

    # ── 4. Retention / Churn ───────────────────────────────────────────────────
    ret_target = _find_retention_target(df, anatomy)
    if ret_target:
        targets.append(ret_target)

    # ── 5. Explicit label columns ──────────────────────────────────────────────
    for col_info in anatomy:
        if col_info["category"] == "Potential Targets":
            targets.append({
                "name": col_info["name"],
                "type": "Explicit Label",
                "formula": col_info["name"],
                "confidence": 75,
                "reasoning": f"Column '{col_info['name']}' matches typical target naming conventions (target, label, y_, flag, propensity).",
                "source_columns": [col_info["name"]],
            })

    # Deduplicate
    seen = set()
    unique = []
    for t in targets:
        key = t["name"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(t)

    return sorted(unique, key=lambda x: -x["confidence"])


def _find_frequency_target(df, anatomy):
    # Binary claim flag or numeric count → frequency
    flag_pat = re.compile(r"claim.?(flag|ind|indicator|binary)|has.?claim|is.?claim|clm.?flag", re.I)
    cnt_pat = re.compile(r"claim.?(count|cnt|num|freq)|n.?claim|num.?claim", re.I)

    for col_info in anatomy:
        col = col_info["name"]
        if flag_pat.search(col):
            return {
                "name": "Claim Frequency (Binary)",
                "type": "Frequency Target",
                "formula": f"{col} = 1 (claim occurred)",
                "confidence": 92,
                "reasoning": f"'{col}' appears to be a binary claim indicator. Suitable as a frequency/classification target: claim vs. no claim.",
                "source_columns": [col],
            }
        if cnt_pat.search(col) and col in df.columns:
            if df[col].dtype in [float, int] or np.issubdtype(df[col].dtype, np.number):
                return {
                    "name": "Claim Count (Frequency)",
                    "type": "Frequency Target",
                    "formula": f"SUM({col}) / Exposure",
                    "confidence": 88,
                    "reasoning": f"'{col}' contains claim counts. Dividing by exposure period gives claim frequency — a core actuarial target.",
                    "source_columns": [col],
                }
    return None


def _find_severity_target(df, anatomy, claims):
    amt_pat = re.compile(r"claim.?(amount|paid|value|cost)|paid.?amount|loss.?amount|incurred", re.I)
    cnt_pat = re.compile(r"claim.?(count|cnt|num)|n.?claim|num.?claim", re.I)

    amount_col = None
    count_col = None
    for col_info in anatomy:
        col = col_info["name"]
        if not amount_col and amt_pat.search(col):
            amount_col = col
        if not count_col and cnt_pat.search(col):
            count_col = col

    if amount_col:
        formula = (f"{amount_col} / {count_col}" if count_col else f"{amount_col} (where claim > 0)")
        conf = 95 if count_col else 82
        return {
            "name": "Claim Severity",
            "type": "Severity Target",
            "formula": formula,
            "confidence": conf,
            "reasoning": f"'{amount_col}' represents claim amounts. " + (f"Dividing by '{count_col}' (claim count) yields average claim severity — a key actuarial GLM target." if count_col else "Filtering to non-zero claims gives the severity distribution."),
            "source_columns": list({c for c in [amount_col, count_col] if c}),
        }
    return None


def _find_expected_loss_target(df, anatomy):
    sev_col = None
    exp_col = None
    freq_col = None

    for col_info in anatomy:
        col = col_info["name"]
        if re.search(r"claim.?(amount|paid|value|incurred|cost)", col, re.I):
            sev_col = col
        if re.search(r"exposure|expo|earned|duration|days_at_risk", col, re.I):
            exp_col = col
        if re.search(r"claim.?(count|cnt|freq|num)|n.?claim", col, re.I):
            freq_col = col

    if sev_col and exp_col:
        return {
            "name": "Expected Loss",
            "type": "Expected Loss Target",
            "formula": f"({sev_col} / {exp_col})" + (f" × {freq_col}" if freq_col else ""),
            "confidence": 78,
            "reasoning": "Claim amount normalised by exposure yields expected loss per unit of exposure — the standard actuarial pure premium target.",
            "source_columns": list({c for c in [sev_col, exp_col, freq_col] if c}),
        }
    return None


def _find_retention_target(df, anatomy):
    lapse_pat = re.compile(r"lapse|churn|cancel|renew|retain|termination|policy.?end", re.I)
    for col_info in anatomy:
        col = col_info["name"]
        if lapse_pat.search(col):
            return {
                "name": "Retention / Lapse",
                "type": "Retention Target",
                "formula": f"{col} (binary: lapsed vs. retained)",
                "confidence": 72,
                "reasoning": f"'{col}' appears to contain lapse or cancellation information. This could form a retention/churn classification target.",
                "source_columns": [col],
            }
    return None
