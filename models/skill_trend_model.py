# ============================================================
# Career Obsolescence Intelligence System
# FILE: models/skill_trend_model.py
# PURPOSE: Analyze & score skill trends — rising vs declining
#          Outputs trend classifications + 2026 projections
# RUN: python models/skill_trend_model.py
# ============================================================

import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PROC_DIR  = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    df = pd.read_csv(f"{PROC_DIR}/skills_trend_clean.csv")
    print(f"  📥 Loaded skills_trend_clean.csv — {len(df):,} rows")
    return df


# ============================================================
# MODULE 1 — Per-Skill Linear Trend Model
# Fits a linear regression on trend_score ~ year for each skill
# Extracts: slope, R², intercept, 2026 projection
# ============================================================
def fit_skill_trend_models(df):
    print("\n  🔬 Fitting per-skill Linear Trend Models ...")
    results = []

    for skill, group in df.groupby("skill"):
        group = group.sort_values("year")
        X = group["year"].values.reshape(-1, 1)
        y = group["trend_score"].values

        model = LinearRegression()
        model.fit(X, y)

        slope     = round(model.coef_[0], 5)
        intercept = round(model.intercept_, 4)
        r2        = round(model.score(X, y), 4)

        # 2026 projection
        proj_2026 = round(float(model.predict([[2026]])[0]), 2)
        proj_2026 = np.clip(proj_2026, 0.5, 10.0)

        # Postings growth: slope of job postings over time
        p_model = LinearRegression()
        p_model.fit(X, group["job_postings_count"].values)
        postings_slope = round(p_model.coef_[0], 2)

        results.append({
            "skill":              skill,
            "trend_slope":        slope,
            "trend_intercept":    intercept,
            "trend_r2":           r2,
            "trend_score_2025":   round(group[group["year"] == 2025]["trend_score"].values[0]
                                        if 2025 in group["year"].values
                                        else float(model.predict([[2025]])[0]), 2),
            "trend_score_2026_proj": proj_2026,
            "postings_slope":     postings_slope,
            "category":           group["category"].iloc[0],
            "velocity_label":     group["velocity_label"].iloc[0],
            "avg_trend_score":    round(y.mean(), 3),
            "max_trend_score":    round(y.max(), 3),
            "min_trend_score":    round(y.min(), 3),
        })

    df_results = pd.DataFrame(results)
    print(f"  ✅ Modelled {len(df_results)} skills")
    return df_results


# ============================================================
# MODULE 2 — Composite Trend Score (0–100)
# Combines slope, avg_trend, postings_slope into one score
# ============================================================
def compute_composite_trend_score(df_results):
    print("\n  📊 Computing Composite Trend Score (0–100) ...")

    scaler = MinMaxScaler(feature_range=(0, 100))

    # Features to combine
    features = df_results[["trend_slope", "avg_trend_score", "postings_slope"]].copy()

    # Clip postings_slope to avoid extreme outliers dominating
    features["postings_slope"] = features["postings_slope"].clip(-500, 500)

    scaled = scaler.fit_transform(features)

    # Weighted blend: slope matters most, then avg trend, then postings
    weights = [0.45, 0.35, 0.20]
    df_results["composite_trend_score"] = np.average(scaled, axis=1, weights=weights)
    df_results["composite_trend_score"]  = df_results["composite_trend_score"].round(2)

    # Save scaler for dashboard use
    joblib.dump(scaler, f"{MODEL_DIR}/trend_score_scaler.pkl")

    return df_results


# ============================================================
# MODULE 3 — Classify Skills into 5 Tiers
# ============================================================
def classify_skill_tiers(df_results):
    print("\n  🏷  Classifying skill tiers ...")

    def assign_tier(row):
        score = row["composite_trend_score"]
        slope = row["trend_slope"]

        if score >= 80:
            return "🚀 Skyrocketing"
        elif score >= 60:
            return "📈 Rising"
        elif score >= 40 and abs(slope) < 0.05:
            return "➡️  Stable"
        elif score >= 20:
            return "📉 Declining"
        else:
            return "💀 Obsolete"

    df_results["skill_tier"] = df_results.apply(assign_tier, axis=1)

    tier_counts = df_results["skill_tier"].value_counts()
    print(f"  Tier distribution:")
    for tier, count in tier_counts.items():
        print(f"    {tier}: {count} skills")

    return df_results


# ============================================================
# MODULE 4 — Generate Year-by-Year Projections (2026–2028)
# ============================================================
def generate_projections(df, df_results):
    print("\n  🔮 Generating 2026–2028 projections ...")
    proj_rows = []

    for _, row in df_results.iterrows():
        skill = row["skill"]
        slope = row["trend_slope"]
        intercept = row["trend_intercept"]

        # Historical rows
        hist = df[df["skill"] == skill][["year", "trend_score", "job_postings_count"]].copy()
        hist["type"] = "historical"

        # Projected rows
        for yr in [2026, 2027, 2028]:
            proj_score = np.clip(slope * yr + intercept, 0.5, 10.0)
            # Extrapolate postings using postings_slope
            last_postings = df[df["skill"] == skill]["job_postings_count"].iloc[-1]
            proj_postings = max(0, int(last_postings + row["postings_slope"] * (yr - 2025)))

            proj_rows.append({
                "skill": skill,
                "year": yr,
                "trend_score": round(proj_score, 2),
                "job_postings_count": proj_postings,
                "type": "projected",
                "skill_tier": row["skill_tier"],
            })

    df_proj = pd.DataFrame(proj_rows)

    # Combine historical + projected
    df_hist = df[["skill", "year", "trend_score", "job_postings_count"]].copy()
    df_hist["type"] = "historical"
    df_hist["skill_tier"] = df_hist["skill"].map(
        df_results.set_index("skill")["skill_tier"]
    )

    df_full = pd.concat([df_hist, df_proj], ignore_index=True)
    df_full.sort_values(["skill", "year"], inplace=True)

    df_full.to_csv(f"{PROC_DIR}/skill_projections.csv", index=False)
    print(f"  💾 Saved skill_projections.csv ({len(df_full):,} rows)")
    return df_full


# ============================================================
# MODULE 5 — Top Rising & Declining Skills Summary
# ============================================================
def build_top_skills_summary(df_results):
    print("\n  🏆 Building top skills summary ...")

    top_rising   = df_results.nlargest(10, "composite_trend_score")[
        ["skill", "composite_trend_score", "trend_slope",
         "trend_score_2025", "trend_score_2026_proj", "skill_tier"]
    ].reset_index(drop=True)

    top_declining = df_results.nsmallest(10, "composite_trend_score")[
        ["skill", "composite_trend_score", "trend_slope",
         "trend_score_2025", "trend_score_2026_proj", "skill_tier"]
    ].reset_index(drop=True)

    top_rising.to_csv(f"{PROC_DIR}/top_rising_skills.csv", index=False)
    top_declining.to_csv(f"{PROC_DIR}/top_declining_skills.csv", index=False)

    print("\n  📈 TOP 10 RISING SKILLS:")
    print(top_rising[["skill", "composite_trend_score", "skill_tier"]].to_string(index=False))

    print("\n  📉 TOP 10 DECLINING SKILLS:")
    print(top_declining[["skill", "composite_trend_score", "skill_tier"]].to_string(index=False))

    return top_rising, top_declining


# ============================================================
# MODULE 6 — Save Final Skill Intelligence Table
# ============================================================
def save_skill_intelligence(df_results):
    output_cols = [
        "skill", "skill_tier", "composite_trend_score",
        "trend_slope", "trend_r2", "avg_trend_score",
        "trend_score_2025", "trend_score_2026_proj",
        "postings_slope", "category", "velocity_label",
    ]
    df_out = df_results[output_cols].sort_values(
        "composite_trend_score", ascending=False
    ).reset_index(drop=True)

    df_out.to_csv(f"{PROC_DIR}/skill_intelligence.csv", index=False)

    # Also save as JSON for dashboard lookups
    skill_dict = df_out.set_index("skill").to_dict(orient="index")
    with open(f"{PROC_DIR}/skill_intelligence.json", "w") as f:
        json.dump(skill_dict, f, indent=2)

    print(f"\n  💾 Saved skill_intelligence.csv ({len(df_out)} skills)")
    print(f"  💾 Saved skill_intelligence.json")
    return df_out


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  📡 Skill Trend Analysis Model")
    print("     Career Obsolescence Intelligence System")
    print("=" * 55)

    df          = load_data()
    df_results  = fit_skill_trend_models(df)
    df_results  = compute_composite_trend_score(df_results)
    df_results  = classify_skill_tiers(df_results)
    df_proj     = generate_projections(df, df_results)
    top_r, top_d = build_top_skills_summary(df_results)
    df_final    = save_skill_intelligence(df_results)

    print("\n" + "=" * 55)
    print("  ✅ SKILL TREND MODEL COMPLETE!")
    print("=" * 55)
    print("\n  Outputs:")
    print("  • data/processed/skill_intelligence.csv")
    print("  • data/processed/skill_intelligence.json")
    print("  • data/processed/skill_projections.csv")
    print("  • data/processed/top_rising_skills.csv")
    print("  • data/processed/top_declining_skills.csv")
    print("  • models/trend_score_scaler.pkl")
    print("\n  ➡️  Next: python models/automation_risk_model.py\n")
