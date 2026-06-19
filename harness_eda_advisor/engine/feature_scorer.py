"""
Feature Scorer — classify features as High Value, Modelling Feature,
Suspicious, Potential Leakage, or Low Value.
"""
import re
import pandas as pd
import numpy as np


HIGH_VALUE_PATTERNS = re.compile(
    r"age|pet.?age|animal.?age|gender|sex|breed|province|state|region|plan|product|"
    r"vintage|tenure|channel|income|occupation|sum.?insured|coverage.?amount",
    re.I
)
LEAKAGE_PATTERNS = re.compile(
    r"future|post.?claim|after.?claim|next.?period|outcome|result|paid.?amount|total.?paid|"
    r"benefit.?paid|loss.?ratio|claim.?ratio",
    re.I
)
SUSPICIOUS_PATTERNS = re.compile(
    r"loss.?ratio|combined.?ratio|profit|margin|expense.?ratio|settlement.?ratio",
    re.I
)
LOW_VALUE_THRESHOLD_COVERAGE = 20   # %
LOW_VALUE_THRESHOLD_UNIQUE = 1      # single value


def score_features(df: pd.DataFrame, anatomy: list) -> list:
    scored = []
    for col_info in anatomy:
        col = col_info["name"]
        cat = col_info["category"]

        # Skip pure identifiers and claim amounts (targets, not features)
        if cat in ("Identifiers",):
            scored.append(_make(col, "Low Value", "Primary identifier — not a predictive feature.", 95,
                                "Exclude from modelling features. Use for record linkage only."))
            continue
        if cat in ("Potential Leakage",):
            scored.append(_make(col, "Potential Leakage",
                                "Column name suggests it contains future or post-event information not available at prediction time.",
                                85, "Exclude from model features. Use for target engineering only."))
            continue
        if cat in ("Potential Targets",):
            continue  # handled by target discoverer

        # Low value checks
        cov = col_info["coverage_pct"]
        nuniq = col_info["unique_values"]
        if cov < LOW_VALUE_THRESHOLD_COVERAGE:
            scored.append(_make(col, "Low Value",
                                f"Only {cov}% populated — insufficient data for reliable analysis.",
                                90, "Consider excluding. If retained, document sparseness clearly."))
            continue
        if nuniq <= LOW_VALUE_THRESHOLD_UNIQUE:
            scored.append(_make(col, "Low Value",
                                f"Column has only {nuniq} unique value(s) — no variance.",
                                98, "Exclude from modelling — carries no information."))
            continue

        # Leakage patterns
        if LEAKAGE_PATTERNS.search(col):
            scored.append(_make(col, "Potential Leakage",
                                "Column name pattern suggests it may contain outcome or post-event data.",
                                75, "Validate timing: is this field available at prediction time?"))
            continue

        # Suspicious
        if SUSPICIOUS_PATTERNS.search(col):
            scored.append(_make(col, "Suspicious",
                                "Derived ratio or combined metric — may encode future information or circularity.",
                                70, "Investigate construction. Do not use as a feature without understanding its derivation."))
            continue

        # High value
        if HIGH_VALUE_PATTERNS.search(col) or cat in ("Geography", "Pet", "Behavioural", "Product", "Policy"):
            confidence = min(95, 60 + int(cov / 3))
            scored.append(_make(col, "High Value",
                                f"Strong predictive signal expected. Category: {cat}. Coverage: {cov}%.",
                                confidence, "Include in EDA and initial feature set."))
            continue

        # Modelling feature (default)
        scored.append(_make(col, "Modelling Feature",
                            f"No specific concerns. Category: {cat}. Coverage: {cov}%.",
                            55, "Include in EDA. Assess via univariate analysis."))

    return sorted(scored, key=lambda x: (
        {"Potential Leakage": 0, "Suspicious": 1, "Low Value": 2, "High Value": 3, "Modelling Feature": 4}
        .get(x["classification"], 5)
    ))


def _make(col, cls, reason, conf, usage):
    return {
        "name": col,
        "classification": cls,
        "reason": reason,
        "confidence": conf,
        "suggested_usage": usage,
    }
