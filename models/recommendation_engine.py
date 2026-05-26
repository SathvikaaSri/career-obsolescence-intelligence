# ============================================================
# Career Obsolescence Intelligence System
# FILE: models/recommendation_engine.py
# PURPOSE: Recommend future-proof skills based on user role,
#          current skills gap, and survival score boost
# RUN: python models/recommendation_engine.py
# ============================================================

import pandas as pd
import numpy as np
import os
import json

PROC_DIR  = "data/processed"
MODEL_DIR = "models"


# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    df_mapping  = pd.read_csv(f"{PROC_DIR}/role_skill_mapping_clean.csv")
    df_skill_intel = pd.read_csv(f"{PROC_DIR}/skill_intelligence.csv")
    df_survival = pd.read_csv(f"{PROC_DIR}/career_survival_scores.csv")
    return df_mapping, df_skill_intel, df_survival


# ============================================================
# CORE: Get Recommendations for a Role
# ============================================================
def get_recommendations(
    job_role: str,
    current_skills: list,
    df_mapping: pd.DataFrame,
    df_skill_intel: pd.DataFrame,
    df_survival: pd.DataFrame,
    top_n: int = 6,
) -> dict:
    """
    Returns skill recommendations for a given role + skill gap.

    Args:
        job_role:       e.g. "Data Analyst"
        current_skills: list of skills the user already has
        top_n:          how many recommendations to return

    Returns:
        dict with recommendations, gap analysis, survival impact
    """

    # ── 1. Get all skills mapped to this role
    role_skills = df_mapping[df_mapping["job_role"] == job_role].copy()

    if role_skills.empty:
        return {"error": f"Role '{job_role}' not found in mapping."}

    # ── 2. Merge with skill intelligence (trend scores, projections)
    role_skills = role_skills.merge(
        df_skill_intel[["skill", "composite_trend_score",
                        "trend_score_2026_proj", "skill_tier",
                        "trend_slope", "velocity_label"]],
        on="skill", how="left"
    )

    # ── 3. Identify gaps (skills NOT in current_skills)
    current_norm = [s.lower().strip() for s in current_skills]
    role_skills["already_has"] = role_skills["skill"].str.lower().str.strip().isin(current_norm)
    gap_skills   = role_skills[~role_skills["already_has"]].copy()
    owned_skills = role_skills[role_skills["already_has"]].copy()

    # ── 4. Score each gap skill
    # Combined score: priority_score (from mapping) + composite_trend_score
    gap_skills["recommendation_score"] = (
        gap_skills["priority_score"].fillna(0)      * 0.45 +
        gap_skills["composite_trend_score"].fillna(50) * 0.55
    ).round(3)

    # Boost score for "Rising" category
    gap_skills["recommendation_score"] += np.where(
        gap_skills["category"] == "Rising", 5, 0
    )
    # Penalise "Declining" skills
    gap_skills["recommendation_score"] -= np.where(
        gap_skills["category"] == "Declining", 10, 0
    )
    gap_skills["recommendation_score"] = gap_skills["recommendation_score"].clip(0)

    top_recs = gap_skills.nlargest(top_n, "recommendation_score").reset_index(drop=True)

    # ── 5. Survival score impact estimate
    survival_row = df_survival[df_survival["job_role"] == job_role]
    current_survival = float(survival_row["career_survival_score"].values[0]) \
                       if not survival_row.empty else 50.0

    # Each recommended skill adds a small boost based on its trend score
    boost_per_skill = top_recs["composite_trend_score"].fillna(50) / 100 * 3
    projected_survival = min(100, current_survival + boost_per_skill.sum())

    # ── 6. Learning roadmap order (by recommendation_score desc)
    roadmap = []
    for i, row in top_recs.iterrows():
        roadmap.append({
            "rank":                 i + 1,
            "skill":                row["skill"],
            "skill_tier":           row.get("skill_tier", "Unknown"),
            "category":             row["category"],
            "trend_score_2026":     round(row.get("trend_score_2026_proj", 0), 2),
            "salary_boost_usd":     int(row["salary_boost_usd"]),
            "recommendation_score": round(row["recommendation_score"], 2),
            "velocity":             row.get("velocity_label", "Stable"),
            "why": _generate_why(row),
        })

    return {
        "job_role":              job_role,
        "current_skills":        current_skills,
        "skills_owned_count":    len(owned_skills),
        "skills_gap_count":      len(gap_skills),
        "current_survival_score": round(current_survival, 2),
        "projected_survival_score": round(projected_survival, 2),
        "survival_boost":        round(projected_survival - current_survival, 2),
        "recommendations":       roadmap,
        "owned_skills_analysis": _owned_skills_analysis(owned_skills),
    }


def _generate_why(row) -> str:
    """Human-readable reason for recommendation."""
    tier = row.get("skill_tier", "")
    sal  = int(row.get("salary_boost_usd", 0))
    cat  = row.get("category", "")
    vel  = row.get("velocity_label", "")

    reasons = []
    if "Skyrocketing" in str(tier) or "Rising" in str(tier):
        reasons.append("rapidly growing demand")
    if sal >= 15000:
        reasons.append(f"+${sal:,} salary potential")
    if vel == "Rising":
        reasons.append("accelerating trend")
    if cat == "Rising":
        reasons.append("high market demand")
    if not reasons:
        reasons.append("strong role alignment")

    return "Recommended for: " + ", ".join(reasons)


def _owned_skills_analysis(owned: pd.DataFrame) -> list:
    """Analyse skills the user already has."""
    if owned.empty:
        return []
    result = []
    for _, row in owned.iterrows():
        result.append({
            "skill":    row["skill"],
            "category": row["category"],
            "status":   "✅ Strong Keep" if row["category"] == "Rising"
                        else ("⚠️ Complement" if row["category"] == "Stable"
                              else "🔴 At Risk Skill"),
        })
    return result


# ============================================================
# BATCH: Recommendations for ALL Roles
# ============================================================
def generate_all_role_recommendations(df_mapping, df_skill_intel, df_survival):
    """Pre-compute top-3 recommendations for every role (for dashboard)."""
    print("\n  🔄 Generating recommendations for all roles ...")

    all_rows = []
    for role in df_mapping["job_role"].unique():
        result = get_recommendations(
            job_role=role,
            current_skills=[],          # no skills = full gap = best recs
            df_mapping=df_mapping,
            df_skill_intel=df_skill_intel,
            df_survival=df_survival,
            top_n=5,
        )
        if "error" in result:
            continue
        for rec in result["recommendations"]:
            all_rows.append({
                "job_role":           role,
                "recommended_skill":  rec["skill"],
                "skill_tier":         rec["skill_tier"],
                "category":           rec["category"],
                "salary_boost_usd":   rec["salary_boost_usd"],
                "trend_score_2026":   rec["trend_score_2026"],
                "recommendation_score": rec["recommendation_score"],
                "rank":               rec["rank"],
                "why":                rec["why"],
            })

    df_recs = pd.DataFrame(all_rows)
    df_recs.to_csv(f"{PROC_DIR}/all_role_recommendations.csv", index=False)
    print(f"  💾 Saved all_role_recommendations.csv ({len(df_recs)} rows)")
    return df_recs


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🎯 Recommendation Engine")
    print("     Career Obsolescence Intelligence System")
    print("=" * 55)

    df_mapping, df_skill_intel, df_survival = load_data()

    # ── Demo: Data Analyst with some skills
    print("\n  📌 Demo: Data Analyst (has SQL, Excel (Basic))")
    result = get_recommendations(
        job_role="Data Analyst",
        current_skills=["SQL", "Excel (Basic)"],
        df_mapping=df_mapping,
        df_skill_intel=df_skill_intel,
        df_survival=df_survival,
        top_n=5,
    )
    print(f"\n  Current Survival Score:   {result['current_survival_score']}")
    print(f"  Projected After Upskill:  {result['projected_survival_score']}")
    print(f"  Survival Boost:           +{result['survival_boost']}")
    print(f"\n  📋 Top Recommendations:")
    for r in result["recommendations"]:
        print(f"    {r['rank']}. {r['skill']:25s} | {r['skill_tier']:20s} | +${r['salary_boost_usd']:>6,} | {r['why']}")

    print("\n  ✅ Owned Skills Analysis:")
    for s in result["owned_skills_analysis"]:
        print(f"    {s['status']} {s['skill']}")

    # ── Batch generate
    df_all_recs = generate_all_role_recommendations(df_mapping, df_skill_intel, df_survival)

    print("\n" + "=" * 55)
    print("  ✅ RECOMMENDATION ENGINE COMPLETE!")
    print("=" * 55)
    print("  • data/processed/all_role_recommendations.csv")
    print("\n  ➡️  Building Streamlit dashboard next...\n")
