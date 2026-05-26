# ============================================================
# Career Obsolescence Intelligence System
# FILE: models/career_survival_score.py
# PURPOSE: Compute Career Survival Score (0–100) per role
#          Formula combines: market demand, salary growth,
#          automation resistance, human dependency
# RUN: python models/career_survival_score.py
# ============================================================

import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PROC_DIR  = "data/processed"
MODEL_DIR = "models"


# ============================================================
# SURVIVAL SCORE FORMULA
# ─────────────────────────────────────────────────────────────
#
#  CSS = w1*DemandScore + w2*SalaryScore + w3*ResistanceScore
#        + w4*HumanDepScore
#
#  Where:
#    DemandScore      = normalized demand_index (market hunger)
#    SalaryScore      = normalized salary_cagr  (earning power)
#    ResistanceScore  = 1 - automation_risk     (AI resistance)
#    HumanDepScore    = avg_skill_trend_score   (skill future-ness)
#
#  Weights (sum to 1.0):
#    w1 = 0.30  Market Demand
#    w2 = 0.25  Salary Growth
#    w3 = 0.30  Automation Resistance
#    w4 = 0.15  Skill Future-Readiness
#
# ============================================================

WEIGHTS = {
    "demand":     0.30,
    "salary":     0.25,
    "resistance": 0.30,
    "skill":      0.15,
}


# ============================================================
# STEP 1 — Load All Input Data
# ============================================================
def load_inputs():
    print("\n  📥 Loading inputs ...")

    df_jobs    = pd.read_csv(f"{PROC_DIR}/job_roles_clean.csv")
    df_risk    = pd.read_csv(f"{PROC_DIR}/role_automation_risk_scores.csv")
    df_sal_traj = pd.read_csv(f"{PROC_DIR}/salary_trajectory_table.csv")
    df_summary = pd.read_csv(f"{PROC_DIR}/role_summary_stats.csv")

    print(f"  ✅ job_roles_clean:        {len(df_jobs):,} rows")
    print(f"  ✅ role_automation_risk:   {len(df_risk)} rows")
    print(f"  ✅ salary_trajectory:      {len(df_sal_traj)} roles")
    print(f"  ✅ role_summary_stats:     {len(df_summary)} roles")

    return df_jobs, df_risk, df_sal_traj, df_summary


# ============================================================
# STEP 2 — Build Role-Level Aggregates
# ============================================================
def build_role_aggregates(df_jobs, df_risk, df_sal_traj, df_summary):
    print("\n  🔧 Building role-level aggregates ...")

    # ── 2a. Latest-year job data (2025 snapshot)
    latest_year = df_jobs["year"].max()
    df_latest = (
        df_jobs[df_jobs["year"] == latest_year]
        .groupby("job_role")
        .agg(
            avg_demand_index    = ("demand_index",        "mean"),
            avg_skill_trend     = ("avg_skill_trend_score","mean"),
            avg_salary          = ("salary_usd",          "mean"),
            pct_future_proof    = ("future_proof",         "mean"),
        )
        .round(4)
        .reset_index()
    )

    # ── 2b. Automation risk — use Senior level as the benchmark
    df_risk_senior = (
        df_risk[df_risk["experience_level"] == "Senior"]
        [["job_role", "predicted_auto_risk", "human_dependency_score"]]
        .copy()
    )

    # ── 2c. Salary CAGR from trajectory table
    sal_cagr = df_summary[["job_role", "avg_salary", "avg_auto_risk"]].copy()
    # Get growth_2025_to_2030_pct from trajectory table if available
    if "growth_2025_to_2030_pct" in df_sal_traj.columns:
        sal_growth = df_sal_traj[["job_role", "growth_2025_to_2030_pct"]].copy()
    else:
        sal_growth = df_latest[["job_role"]].copy()
        sal_growth["growth_2025_to_2030_pct"] = 20.0  # default

    # ── 2d. Merge everything
    df_agg = df_latest.merge(df_risk_senior, on="job_role", how="left")
    df_agg = df_agg.merge(sal_growth,        on="job_role", how="left")

    # Fill any missing values with medians
    for col in ["predicted_auto_risk", "human_dependency_score",
                "growth_2025_to_2030_pct"]:
        df_agg[col].fillna(df_agg[col].median(), inplace=True)

    print(f"  ✅ Aggregated {len(df_agg)} roles")
    return df_agg


# ============================================================
# STEP 3 — Compute Component Scores (each 0–100)
# ============================================================
def compute_component_scores(df_agg):
    print("\n  📐 Computing component scores (0–100 each) ...")

    scaler = MinMaxScaler(feature_range=(0, 100))

    # ── Demand Score: normalize demand_index
    df_agg["demand_score"] = scaler.fit_transform(
        df_agg[["avg_demand_index"]]
    ).round(2)

    # ── Salary Score: normalize salary growth % (clipped to avoid extremes)
    df_agg["salary_growth_clipped"] = df_agg["growth_2025_to_2030_pct"].clip(-10, 60)
    df_agg["salary_score"] = scaler.fit_transform(
        df_agg[["salary_growth_clipped"]]
    ).round(2)

    # ── Resistance Score: inverse of automation risk
    df_agg["resistance_raw"] = 1 - df_agg["predicted_auto_risk"]
    df_agg["resistance_score"] = scaler.fit_transform(
        df_agg[["resistance_raw"]]
    ).round(2)

    # ── Skill Future-Readiness Score: normalize avg_skill_trend
    df_agg["skill_score"] = scaler.fit_transform(
        df_agg[["avg_skill_trend"]]
    ).round(2)

    print("  Component score ranges:")
    for col in ["demand_score", "salary_score", "resistance_score", "skill_score"]:
        print(f"    {col:20s}: {df_agg[col].min():.1f} – {df_agg[col].max():.1f}")

    return df_agg


# ============================================================
# STEP 4 — Compute Final Career Survival Score
# ============================================================
def compute_survival_score(df_agg):
    print("\n  🎯 Computing Career Survival Score ...")

    df_agg["career_survival_score"] = (
        WEIGHTS["demand"]     * df_agg["demand_score"]     +
        WEIGHTS["salary"]     * df_agg["salary_score"]     +
        WEIGHTS["resistance"] * df_agg["resistance_score"] +
        WEIGHTS["skill"]      * df_agg["skill_score"]
    ).round(2)

    # ── Survival Grade
    def survival_grade(score):
        if score >= 80: return "A+ 🏆 Elite"
        if score >= 65: return "A  🟢 Strong"
        if score >= 50: return "B  🟡 Stable"
        if score >= 35: return "C  🟠 Vulnerable"
        if score >= 20: return "D  🔴 At Risk"
        return              "F  💀 Endangered"

    df_agg["survival_grade"] = df_agg["career_survival_score"].apply(survival_grade)

    # ── Rank
    df_agg["survival_rank"] = df_agg["career_survival_score"].rank(
        ascending=False, method="min"
    ).astype(int)

    # ── Urgency flag: needs upskilling NOW
    df_agg["upskill_urgency"] = df_agg.apply(
        lambda r: "🚨 Urgent"   if r["career_survival_score"] < 35  else
                  "⚠️  Soon"    if r["career_survival_score"] < 55  else
                  "✅ On Track", axis=1
    )

    print("\n  🏆 TOP 10 CAREER SURVIVAL SCORES:")
    top10 = df_agg.nlargest(10, "career_survival_score")[
        ["job_role", "career_survival_score", "survival_grade", "upskill_urgency"]
    ]
    print(top10.to_string(index=False))

    print("\n  ⚠️  BOTTOM 10 — MOST ENDANGERED:")
    bot10 = df_agg.nsmallest(10, "career_survival_score")[
        ["job_role", "career_survival_score", "survival_grade", "upskill_urgency"]
    ]
    print(bot10.to_string(index=False))

    return df_agg


# ============================================================
# STEP 5 — Per-Experience Survival Scores
# ============================================================
def compute_experience_survival_scores(df_jobs, df_risk):
    """
    Survival score broken down by experience level.
    Shows how seniority shifts your survival probability.
    """
    print("\n  📊 Computing experience-level survival breakdown ...")

    latest = df_jobs[df_jobs["year"] == df_jobs["year"].max()]

    exp_agg = (
        latest.groupby(["job_role", "experience_level", "experience_ord"])
        .agg(
            demand_index    = ("demand_index",          "mean"),
            skill_trend     = ("avg_skill_trend_score", "mean"),
            salary          = ("salary_usd",            "mean"),
        )
        .reset_index()
    )

    exp_agg = exp_agg.merge(
        df_risk[["job_role", "experience_level",
                 "predicted_auto_risk", "human_dependency_score"]],
        on=["job_role", "experience_level"], how="left"
    )
    exp_agg["predicted_auto_risk"].fillna(0.5, inplace=True)

    # Simple survival formula (normalized inline)
    exp_agg["quick_survival"] = (
        (exp_agg["demand_index"]   / 200 * 30) +
        (exp_agg["skill_trend"]    / 10  * 15) +
        ((1 - exp_agg["predicted_auto_risk"]) * 30) +
        (exp_agg["experience_ord"] / 5   * 25)
    ).clip(0, 100).round(2)

    exp_agg.to_csv(f"{PROC_DIR}/experience_survival_scores.csv", index=False)
    print(f"  💾 Saved experience_survival_scores.csv ({len(exp_agg)} rows)")
    return exp_agg


# ============================================================
# STEP 6 — Year-by-Year Survival Trajectory (2019–2030)
# ============================================================
def compute_survival_trajectory(df_jobs):
    """
    How survival score changes year over year per role.
    Shows which roles are trending safer vs more endangered.
    """
    print("\n  📈 Computing survival trajectory (2019–2030) ...")

    yearly = (
        df_jobs.groupby(["job_role", "year"])
        .agg(
            demand_index  = ("demand_index",          "mean"),
            skill_trend   = ("avg_skill_trend_score", "mean"),
            auto_risk     = ("automation_risk",       "mean"),
        )
        .reset_index()
    )

    yearly["survival_score"] = (
        (yearly["demand_index"] / 200   * 30) +
        (yearly["skill_trend"]  / 10    * 15) +
        ((1 - yearly["auto_risk"])      * 55)
    ).clip(0, 100).round(2)

    # YoY change in survival score
    yearly.sort_values(["job_role", "year"], inplace=True)
    yearly["survival_yoy_change"] = (
        yearly.groupby("job_role")["survival_score"].diff().round(3)
    )

    yearly.to_csv(f"{PROC_DIR}/survival_trajectory.csv", index=False)
    print(f"  💾 Saved survival_trajectory.csv ({len(yearly):,} rows)")
    return yearly


# ============================================================
# STEP 7 — Live Score Function (used by dashboard)
# ============================================================
def calculate_live_survival_score(
    demand_index: float,
    salary_growth_pct: float,
    automation_risk: float,
    avg_skill_trend: float,
    weights: dict = None,
) -> dict:
    """
    Compute career survival score for a single user input.
    Used directly in the Streamlit dashboard.

    Args:
        demand_index:       job posting demand (0–200+)
        salary_growth_pct:  expected 5yr salary growth %
        automation_risk:    0.0 (safe) – 1.0 (fully automatable)
        avg_skill_trend:    average trend score of skills (0–10)
        weights:            optional override of default weights

    Returns:
        dict with score, grade, component breakdown
    """
    w = weights or WEIGHTS

    # Normalize each component to 0–100
    demand_s     = min(100, (demand_index / 200) * 100)
    salary_s     = min(100, max(0, ((salary_growth_pct + 10) / 70) * 100))
    resistance_s = (1 - automation_risk) * 100
    skill_s      = (avg_skill_trend / 10) * 100

    score = (
        w["demand"]     * demand_s     +
        w["salary"]     * salary_s     +
        w["resistance"] * resistance_s +
        w["skill"]      * skill_s
    )
    score = round(np.clip(score, 0, 100), 2)

    grade_map = [
        (80, "A+ 🏆 Elite"),
        (65, "A  🟢 Strong"),
        (50, "B  🟡 Stable"),
        (35, "C  🟠 Vulnerable"),
        (20, "D  🔴 At Risk"),
        (0,  "F  💀 Endangered"),
    ]
    grade = next(g for threshold, g in grade_map if score >= threshold)

    urgency = (
        "🚨 Urgent"  if score < 35 else
        "⚠️  Soon"   if score < 55 else
        "✅ On Track"
    )

    return {
        "career_survival_score": score,
        "survival_grade":        grade,
        "upskill_urgency":       urgency,
        "components": {
            "demand_score":     round(demand_s,     2),
            "salary_score":     round(salary_s,     2),
            "resistance_score": round(resistance_s, 2),
            "skill_score":      round(skill_s,      2),
        },
        "weights": w,
    }


# ============================================================
# STEP 8 — Save Final Outputs
# ============================================================
def save_outputs(df_agg):
    output_cols = [
        "job_role",
        "career_survival_score",
        "survival_grade",
        "survival_rank",
        "upskill_urgency",
        "demand_score",
        "salary_score",
        "resistance_score",
        "skill_score",
        "predicted_auto_risk",
        "human_dependency_score",
        "avg_demand_index",
        "avg_skill_trend",
        "growth_2025_to_2030_pct",
        "pct_future_proof",
    ]
    df_out = df_agg[output_cols].sort_values(
        "career_survival_score", ascending=False
    ).reset_index(drop=True)

    df_out.to_csv(f"{PROC_DIR}/career_survival_scores.csv", index=False)
    print(f"\n  💾 Saved career_survival_scores.csv ({len(df_out)} roles)")

    # JSON for dashboard lookups
    score_dict = df_out.set_index("job_role").to_dict(orient="index")
    with open(f"{PROC_DIR}/career_survival_scores.json", "w") as f:
        json.dump(score_dict, f, indent=2)
    print(f"  💾 Saved career_survival_scores.json")

    # Save weights config
    with open(f"{MODEL_DIR}/survival_score_weights.json", "w") as f:
        json.dump(WEIGHTS, f, indent=2)
    print(f"  💾 Saved survival_score_weights.json")

    return df_out


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🛡️  Career Survival Score Engine")
    print("     Career Obsolescence Intelligence System")
    print("=" * 55)
    print(f"\n  Formula Weights: {WEIGHTS}")

    df_jobs, df_risk, df_sal_traj, df_summary = load_inputs()
    df_agg     = build_role_aggregates(df_jobs, df_risk, df_sal_traj, df_summary)
    df_agg     = compute_component_scores(df_agg)
    df_agg     = compute_survival_score(df_agg)
    exp_scores = compute_experience_survival_scores(df_jobs, df_risk)
    traj       = compute_survival_trajectory(df_jobs)
    df_out     = save_outputs(df_agg)

    # ── Live function sanity check
    print("\n  🧪 Sanity Check — Live Score Function:")
    result = calculate_live_survival_score(
        demand_index=145,
        salary_growth_pct=42,
        automation_risk=0.12,
        avg_skill_trend=8.8,
    )
    print(f"     Score:   {result['career_survival_score']}")
    print(f"     Grade:   {result['survival_grade']}")
    print(f"     Urgency: {result['upskill_urgency']}")
    print(f"     Components: {result['components']}")

    print("\n" + "=" * 55)
    print("  ✅ CAREER SURVIVAL SCORE ENGINE COMPLETE!")
    print("=" * 55)
    print("\n  Outputs:")
    print("  • data/processed/career_survival_scores.csv")
    print("  • data/processed/career_survival_scores.json")
    print("  • data/processed/experience_survival_scores.csv")
    print("  • data/processed/survival_trajectory.csv")
    print("  • models/survival_score_weights.json")
    print("\n  ➡️  Next: python models/recommendation_engine.py\n")
