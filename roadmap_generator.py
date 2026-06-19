"""
EDA Roadmap Generator — produces a dynamic, dataset-driven EDA path.
Never hardcoded. Derived from what the data actually contains.
"""
import pandas as pd


def generate_roadmap(df, anatomy, claims, targets, quality) -> tuple[list, str]:
    steps = []
    has_claims = claims is not None
    has_exposure = any(c["category"] == "Exposure" for c in anatomy)
    has_premium = any(c["category"] == "Premium" for c in anatomy)
    has_geo = any(c["category"] == "Geography" for c in anatomy)
    has_pet = any(c["category"] == "Pet" for c in anatomy)
    has_dates = any(c["category"] == "Dates" for c in anatomy)
    n_targets = len(targets) if targets else 0
    fail_checks = sum(1 for c in quality.get("checks", []) if c["status"] == "fail")
    warn_checks = sum(1 for c in quality.get("checks", []) if c["status"] == "warn")

    # ── Always first: Portfolio Overview ───────────────────────────────────────
    steps.append({
        "title": "Portfolio Overview",
        "why": "Establish the fundamental shape of the portfolio: size, structure, key segments, and high-level metrics. This orients all subsequent analysis.",
        "priority": "High",
        "effort": "Low",
        "business_value": "High",
        "expected_insight": "Portfolio composition, key segments, relative sizes"
    })

    # ── Quality remediation if issues ──────────────────────────────────────────
    if fail_checks > 0 or warn_checks > 3:
        steps.append({
            "title": "Data Quality Remediation",
            "why": f"{fail_checks} failed and {warn_checks} warning quality checks detected. Address these before proceeding to ensure analytical conclusions are valid.",
            "priority": "High",
            "effort": "Medium",
            "business_value": "High",
            "expected_insight": "Corrected dataset, documented exclusions, known limitations"
        })

    # ── Exposure if detected ───────────────────────────────────────────────────
    if has_exposure:
        steps.append({
            "title": "Exposure Analysis",
            "why": "Exposure is the denominator for all rate-based metrics (frequency, severity, loss rate). Understanding exposure distribution is essential before any claims analysis.",
            "priority": "High",
            "effort": "Medium",
            "business_value": "High",
            "expected_insight": "Exposure distribution, seasonality, segment exposure concentration"
        })

    # ── Claims if detected ─────────────────────────────────────────────────────
    if has_claims:
        steps.append({
            "title": "Claims Validation",
            "why": "Validate that detected claims fields are structurally sound before analysis. Check for zeros, negatives, outliers, and implausible values.",
            "priority": "High",
            "effort": "Low",
            "business_value": "High",
            "expected_insight": "Claims data reliability, anomaly identification"
        })
        steps.append({
            "title": "Claims Overview",
            "why": "Summarise the claims portfolio: total claims, claiming rate, total paid, average claim, distribution shape. This is the anchor for all actuarial analysis.",
            "priority": "High",
            "effort": "Low",
            "business_value": "High",
            "expected_insight": "Claiming rate, total losses, average claim value"
        })
        steps.append({
            "title": "Claim Severity Analysis",
            "why": "Understand the distribution of individual claim amounts. Identify large loss concentration, heavy tails, and severity drivers.",
            "priority": "High",
            "effort": "Medium",
            "business_value": "High",
            "expected_insight": "Severity distribution shape, heavy tail identification, large loss threshold"
        })
        steps.append({
            "title": "Claim Frequency Analysis",
            "why": "Measure how often claims occur across segments. Frequency × severity = pure premium — understanding each component separately is fundamental.",
            "priority": "High",
            "effort": "Medium",
            "business_value": "High",
            "expected_insight": "Frequency by segment, high-risk cohorts, frequency trend"
        })

    # ── Geo analysis ───────────────────────────────────────────────────────────
    if has_geo:
        steps.append({
            "title": "Geographic Analysis",
            "why": "Geography is a primary risk stratifier. Understand how exposure and claims vary across regions before building any segmented models.",
            "priority": "Medium",
            "effort": "Medium",
            "business_value": "High",
            "expected_insight": "Regional risk variation, geographic concentration, territory performance"
        })

    # ── Pet / product specific ─────────────────────────────────────────────────
    if has_pet:
        steps.append({
            "title": "Breed & Species Risk Analysis",
            "why": "In pet insurance, breed is a primary risk factor. Understanding breed-level claim frequency and severity is essential before modelling.",
            "priority": "Medium",
            "effort": "Medium",
            "business_value": "High",
            "expected_insight": "High-risk breeds, species-level variation, breed coverage gaps"
        })

    # ── Premium analysis ───────────────────────────────────────────────────────
    if has_premium:
        steps.append({
            "title": "Premium & Pricing Analysis",
            "why": "Compare earned premium to claims experience to assess loss ratio by segment. Identify underpriced and overpriced segments.",
            "priority": "Medium",
            "effort": "Medium",
            "business_value": "High",
            "expected_insight": "Loss ratio by segment, premium adequacy signals, pricing outliers"
        })

    # ── Date trend ─────────────────────────────────────────────────────────────
    if has_dates:
        steps.append({
            "title": "Experience Trends Over Time",
            "why": "Identify trends in frequency, severity, and loss ratio over time. Detect seasonality, deterioration, or portfolio changes.",
            "priority": "Medium",
            "effort": "Medium",
            "business_value": "Medium",
            "expected_insight": "Claim trend, seasonal patterns, portfolio growth/decline"
        })

    # ── Expected Loss ──────────────────────────────────────────────────────────
    if has_claims and has_exposure:
        steps.append({
            "title": "Expected Loss (Pure Premium) Analysis",
            "why": "Combine frequency and severity to compute pure premium by segment. This is the actuarial pricing basis and the most important pre-modelling diagnostic.",
            "priority": "High",
            "effort": "Medium",
            "business_value": "High",
            "expected_insight": "Pure premium by segment, adequacy signals, modelling targets confirmed"
        })

    # ── Clustering / segmentation ──────────────────────────────────────────────
    steps.append({
        "title": "Risk Segmentation & Clustering",
        "why": "Use multivariate analysis to identify natural risk segments before formal modelling. Understand whether the portfolio has distinct risk strata.",
        "priority": "Medium",
        "effort": "High",
        "business_value": "Medium",
        "expected_insight": "Natural risk groupings, segment homogeneity, clustering structure"
    })

    # ── Target analysis ────────────────────────────────────────────────────────
    if n_targets > 0:
        steps.append({
            "title": f"Target Variable Analysis ({n_targets} candidates)",
            "why": f"{n_targets} candidate target(s) identified. Assess each: distribution, balance, outliers, and suitability for GLM or ML modelling.",
            "priority": "High",
            "effort": "Medium",
            "business_value": "High",
            "expected_insight": "Target distribution, class imbalance, target transformations needed"
        })

    # ── Modelling readiness ────────────────────────────────────────────────────
    steps.append({
        "title": "Modelling Readiness Assessment",
        "why": "Before modelling: confirm feature completeness, target clarity, exposure availability, and absence of leakage. This is the final EDA gate.",
        "priority": "Medium",
        "effort": "Low",
        "business_value": "High",
        "expected_insight": "Go/no-go for modelling, feature set confirmed, targets locked"
    })

    n_high = sum(1 for s in steps if s["priority"] == "High")
    summary = (
        f"A {len(steps)}-step EDA roadmap has been generated from your dataset characteristics. "
        f"{n_high} high-priority steps should be completed before modelling begins. "
        f"{'Claims and exposure fields detected — actuarial analysis path recommended.' if has_claims and has_exposure else ''}"
        f"{'Geographic segmentation included.' if has_geo else ''}"
    ).strip()

    return steps, summary
