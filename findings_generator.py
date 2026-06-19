"""
Findings Generator — synthesise observations into actionable findings.
Categories: opportunities, risks, quality_issues, analytical_risks, business_observations
"""
import numpy as np
import pandas as pd


def generate_findings(df, anatomy, quality, claims, targets, features) -> dict:
    findings = {
        "opportunities": [],
        "risks": [],
        "quality_issues": [],
        "analytical_risks": [],
        "business_observations": [],
    }

    n_rows = df.shape[0]

    # ── Opportunities ──────────────────────────────────────────────────────────
    if targets and len(targets) >= 2:
        findings["opportunities"].append({
            "title": f"{len(targets)} Modelling Targets Available",
            "detail": f"The dataset contains {len(targets)} candidate targets: {', '.join(t['name'] for t in targets[:3])}. Multiple target options allow for frequency/severity two-part modelling.",
            "action": "Confirm target definitions with the business before modelling.",
            "impact": "High",
            "confidence": 90,
            "human_review": True,
        })

    high_val_features = [f for f in features if f["classification"] == "High Value"]
    if len(high_val_features) >= 5:
        findings["opportunities"].append({
            "title": f"{len(high_val_features)} High-Value Features Identified",
            "detail": f"Rich feature set detected including: {', '.join(f['name'] for f in high_val_features[:5])}. These are strong candidates for the initial model feature set.",
            "action": "Assess these features via univariate analysis before modelling.",
            "impact": "High",
            "confidence": 80,
            "human_review": False,
        })

    if n_rows >= 100_000:
        findings["opportunities"].append({
            "title": "Large Dataset — Statistical Power Available",
            "detail": f"Dataset contains {n_rows:,} records, providing strong statistical power for segment-level analysis and machine learning models.",
            "action": "Use this volume to validate results across subgroups with confidence.",
            "impact": "Medium",
            "confidence": 95,
            "human_review": False,
        })

    # ── Risks ──────────────────────────────────────────────────────────────────
    leakage_features = [f for f in features if f["classification"] == "Potential Leakage"]
    if leakage_features:
        findings["risks"].append({
            "title": f"{len(leakage_features)} Potential Leakage Features Detected",
            "detail": f"Columns suspected of containing future or post-event information: {', '.join(f['name'] for f in leakage_features[:5])}. Using these as model features would cause data leakage.",
            "action": "Exclude these columns from feature sets until timing has been validated.",
            "impact": "High",
            "confidence": 75,
            "human_review": True,
        })

    sparse_cols = [c for c in anatomy if c["coverage_pct"] < 30]
    if sparse_cols:
        findings["risks"].append({
            "title": f"{len(sparse_cols)} Sparse Columns — Credibility Risk",
            "detail": f"Columns with < 30% population: {', '.join(c['name'] for c in sparse_cols[:5])}. Segment-level statistics will be unreliable.",
            "action": "Document sparseness. Avoid segment-level conclusions where N is small.",
            "impact": "Medium",
            "confidence": 90,
            "human_review": False,
        })

    if claims and claims.get("skewness") and abs(claims["skewness"]) > 2:
        findings["risks"].append({
            "title": "Highly Skewed Claim Distribution",
            "detail": f"Claims distribution skewness = {claims['skewness']:.2f}. A small number of large claims likely dominate total paid. This is common but requires careful handling.",
            "action": "Apply log-transformation or Tweedie distribution in severity models. Investigate large loss records.",
            "impact": "High",
            "confidence": 85,
            "human_review": True,
        })

    # ── Quality Issues ─────────────────────────────────────────────────────────
    fail_checks = [c for c in quality.get("checks", []) if c["status"] == "fail"]
    for chk in fail_checks[:5]:
        findings["quality_issues"].append({
            "title": f"FAIL: {chk['check']}",
            "detail": chk["observation"],
            "action": chk["action"],
            "impact": "High",
            "confidence": 95,
            "human_review": True,
        })

    warn_checks = [c for c in quality.get("checks", []) if c["status"] == "warn"]
    for chk in warn_checks[:4]:
        findings["quality_issues"].append({
            "title": f"WARNING: {chk['check']}",
            "detail": chk["observation"],
            "action": chk["action"],
            "impact": "Medium",
            "confidence": 85,
            "human_review": False,
        })

    # ── Analytical Risks ───────────────────────────────────────────────────────
    suspicious_features = [f for f in features if f["classification"] == "Suspicious"]
    if suspicious_features:
        findings["analytical_risks"].append({
            "title": f"{len(suspicious_features)} Suspicious Features Require Validation",
            "detail": f"Features flagged for potential circularity or derived ratios: {', '.join(f['name'] for f in suspicious_features[:5])}.",
            "action": "Validate construction of these features before including in models.",
            "impact": "High",
            "confidence": 70,
            "human_review": True,
        })

    if not targets or len(targets) == 0:
        findings["analytical_risks"].append({
            "title": "No Clear Modelling Target Detected",
            "detail": "The Harness could not automatically identify candidate targets. The dataset may lack explicit outcome variables.",
            "action": "Define targets manually in consultation with the business. Consider constructing a target from claims and exposure fields.",
            "impact": "High",
            "confidence": 80,
            "human_review": True,
        })

    # ── Business Observations ──────────────────────────────────────────────────
    if claims:
        findings["business_observations"].append({
            "title": f"Claiming Rate: {claims.get('claiming_pct', '?')}%",
            "detail": f"{claims.get('total_claims', '?'):,} records carry a claim out of {n_rows:,} total. " + ("This is a relatively low claim rate — class imbalance should be addressed in frequency models." if (claims.get("claiming_pct") or 100) < 20 else "Claim rate is significant."),
            "action": "Use appropriate modelling techniques for the observed claim frequency.",
            "impact": "Medium",
            "confidence": 95,
            "human_review": False,
        })

    geo_cols = [c for c in anatomy if c["category"] == "Geography"]
    if geo_cols:
        findings["business_observations"].append({
            "title": "Geographic Segmentation Available",
            "detail": f"Geography fields detected: {', '.join(c['name'] for c in geo_cols[:3])}. Geographic risk stratification is available for pricing and reserving analysis.",
            "action": "Include geographic variables in univariate analysis early in EDA.",
            "impact": "Medium",
            "confidence": 85,
            "human_review": False,
        })

    return findings
