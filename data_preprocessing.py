# ============================================================
# Career Obsolescence Intelligence System
# FILE: data_preprocessing.py
# PURPOSE: Clean, validate, and engineer features from raw CSVs
# RUN: python data_preprocessing.py
# ============================================================

import pandas as pd
import numpy as np
import os
import json

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
os.makedirs(PROC_DIR, exist_ok=True)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def report(df, name):
    """Print a clean summary of a dataframe."""
    print(f"\n  📋 {name}")
    print(f"     Rows: {len(df):,}  |  Cols: {df.shape[1]}")
    nulls = df.isnull().sum().sum()
    print(f"     Nulls: {nulls}  |  Dtypes: {dict(df.dtypes.value_counts())}")


def clip_outliers_iqr(series, factor=2.5):
    """
    Clip outliers beyond factor * IQR.
    Returns cleaned series.
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return series.clip(lower=lower, upper=upper)


# ============================================================
# STEP 1 — Process Job Roles Dataset
# ============================================================

def process_job_roles():
    print("\n" + "─" * 50)
    print("  STEP 1: Processing job_roles_dataset.csv")
    print("─" * 50)

    df = pd.read_csv(f"{RAW_DIR}/job_roles_dataset.csv")
    report(df, "Raw job_roles_dataset")

    # ── 1.1 Drop exact duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  🗑  Removed {before - len(df)} duplicate rows")

    # ── 1.2 Validate numeric ranges
    df["salary_usd"] = clip_outliers_iqr(df["salary_usd"])
    df["salary_usd"] = df["salary_usd"].clip(lower=25000)

    df["automation_risk"] = df["automation_risk"].clip(0.0, 1.0)
    df["demand_index"] = df["demand_index"].clip(lower=10)
    df["avg_skill_trend_score"] = df["avg_skill_trend_score"].clip(0.5, 10.0)

    # ── 1.3 Fill any nulls (shouldn't exist in mock data, but good practice)
    df["industry"].fillna("Unknown", inplace=True)
    df["location"].fillna("Unknown", inplace=True)

    # ── 1.4 Feature Engineering

    # Experience level → ordinal integer
    exp_order = {"Entry": 1, "Mid": 2, "Senior": 3, "Lead": 4, "Principal": 5}
    df["experience_ord"] = df["experience_level"].map(exp_order)

    # Salary tier (Low / Mid / High / Elite)
    df["salary_tier"] = pd.qcut(
        df["salary_usd"], q=4,
        labels=["Low", "Mid", "High", "Elite"]
    )

    # Automation risk label
    def risk_label(r):
        if r < 0.25:  return "Low Risk"
        if r < 0.50:  return "Moderate Risk"
        if r < 0.75:  return "High Risk"
        return "Critical Risk"

    df["risk_label"] = df["automation_risk"].apply(risk_label)

    # Skills list (exploded version saved separately below)
    df["skill_list"] = df["skills"].str.split("|")
    df["skill_count"] = df["skill_list"].apply(len)

    # Is the role "Future-Proof"? (risk < 0.35 AND trend > 7)
    df["future_proof"] = (
        (df["automation_risk"] < 0.35) &
        (df["avg_skill_trend_score"] >= 7.0)
    ).astype(int)

    # Salary growth proxy: salary relative to experience median
    salary_med = df.groupby("experience_level")["salary_usd"].transform("median")
    df["salary_vs_median"] = (df["salary_usd"] / salary_med).round(3)

    # ── 1.5 Encode categoricals for ML
    df["role_encoded"] = df["job_role"].astype("category").cat.codes
    df["exp_encoded"] = df["experience_ord"]

    # ── 1.6 Save role-level encoding map (used by models)
    role_map = dict(zip(df["job_role"], df["role_encoded"]))
    with open(f"{PROC_DIR}/role_encoding_map.json", "w") as f:
        json.dump(role_map, f, indent=2)

    # ── 1.7 Save skill-exploded dataset (one row per skill per job)
    df_skills_exp = df[["id", "year", "job_role", "salary_usd",
                         "automation_risk", "demand_index", "skill_list"]].copy()
    df_skills_exp = df_skills_exp.explode("skill_list").rename(
        columns={"skill_list": "skill"}
    )
    df_skills_exp.dropna(subset=["skill"], inplace=True)
    df_skills_exp.to_csv(f"{PROC_DIR}/jobs_skills_exploded.csv", index=False)
    print(f"  💾 Saved jobs_skills_exploded.csv ({len(df_skills_exp):,} rows)")

    # ── 1.8 Drop helper cols before saving main file
    df.drop(columns=["skill_list"], inplace=True)

    df.to_csv(f"{PROC_DIR}/job_roles_clean.csv", index=False)
    report(df, "Processed job_roles_clean")
    print(f"  💾 Saved job_roles_clean.csv ({len(df):,} rows)")
    return df


# ============================================================
# STEP 2 — Process Skills Trend Dataset
# ============================================================

def process_skills_trend():
    print("\n" + "─" * 50)
    print("  STEP 2: Processing skills_trend_dataset.csv")
    print("─" * 50)

    df = pd.read_csv(f"{RAW_DIR}/skills_trend_dataset.csv")
    report(df, "Raw skills_trend_dataset")

    # ── 2.1 Clip outliers
    df["trend_score"] = df["trend_score"].clip(0.5, 10.0)
    df["job_postings_count"] = df["job_postings_count"].clip(lower=0)

    # ── 2.2 Compute Year-over-Year growth in postings
    df.sort_values(["skill", "year"], inplace=True)
    df["postings_yoy_pct"] = (
        df.groupby("skill")["job_postings_count"]
        .pct_change()
        .round(4)
    )

    # ── 2.3 Compute 3-year rolling avg trend score
    df["trend_score_3yr_avg"] = (
        df.groupby("skill")["trend_score"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean().round(3))
    )

    # ── 2.4 Skill velocity: slope of trend score over all years
    def compute_slope(group):
        if len(group) < 2:
            return 0.0
        x = group["year"].values
        y = group["trend_score"].values
        slope = np.polyfit(x - x.min(), y, 1)[0]
        return round(slope, 4)

    slopes = df.groupby("skill").apply(compute_slope).reset_index()
    slopes.columns = ["skill", "trend_velocity"]
    df = df.merge(slopes, on="skill", how="left")

    # ── 2.5 Tag skills as Rising / Stable / Declining by velocity
    def velocity_label(v):
        if v > 0.05:  return "Rising"
        if v < -0.05: return "Declining"
        return "Stable"

    df["velocity_label"] = df["trend_velocity"].apply(velocity_label)

    # ── 2.6 Normalize postings to 0–100 scale within each year
    df["postings_normalized"] = (
        df.groupby("year")["job_postings_count"]
        .transform(lambda x: (x - x.min()) / (x.max() - x.min()) * 100)
        .round(2)
    )

    df.to_csv(f"{PROC_DIR}/skills_trend_clean.csv", index=False)
    report(df, "Processed skills_trend_clean")
    print(f"  💾 Saved skills_trend_clean.csv ({len(df):,} rows)")
    return df


# ============================================================
# STEP 3 — Process Salary History Dataset
# ============================================================

def process_salary_history():
    print("\n" + "─" * 50)
    print("  STEP 3: Processing salary_history_dataset.csv")
    print("─" * 50)

    df = pd.read_csv(f"{RAW_DIR}/salary_history_dataset.csv")
    report(df, "Raw salary_history_dataset")

    # ── 3.1 Clip
    df["avg_salary_usd"] = clip_outliers_iqr(df["avg_salary_usd"])
    df["avg_salary_usd"] = df["avg_salary_usd"].clip(lower=25000)

    # ── 3.2 Sort
    df.sort_values(["job_role", "year"], inplace=True)

    # ── 3.3 YoY salary change
    df["salary_yoy_pct"] = (
        df.groupby("job_role")["avg_salary_usd"]
        .pct_change()
        .round(4)
    )

    # ── 3.4 Salary index (base = 2019 = 100)
    base_salaries = df[df["year"] == 2019].set_index("job_role")["avg_salary_usd"]
    df["salary_index"] = df.apply(
        lambda row: round((row["avg_salary_usd"] / base_salaries.get(row["job_role"], row["avg_salary_usd"])) * 100, 2),
        axis=1
    )

    # ── 3.5 5-year CAGR per role (2019 → 2024)
    def compute_cagr(group):
        sub = group[group["year"].isin([2019, 2024])]
        if len(sub) < 2:
            return np.nan
        sal_2019 = sub[sub["year"] == 2019]["avg_salary_usd"].values[0]
        sal_2024 = sub[sub["year"] == 2024]["avg_salary_usd"].values[0]
        if sal_2019 <= 0:
            return np.nan
        cagr = ((sal_2024 / sal_2019) ** (1 / 5)) - 1
        return round(cagr, 4)

    cagrs = df.groupby("job_role").apply(compute_cagr).reset_index()
    cagrs.columns = ["job_role", "salary_cagr_5yr"]
    df = df.merge(cagrs, on="job_role", how="left")

    # ── 3.6 Role growth tier
    def growth_tier(cagr):
        if pd.isna(cagr):    return "Unknown"
        if cagr > 0.08:      return "High Growth"
        if cagr > 0.04:      return "Moderate Growth"
        if cagr > 0.01:      return "Slow Growth"
        return "Stagnant/Declining"

    df["growth_tier"] = df["salary_cagr_5yr"].apply(growth_tier)

    # ── 3.7 Lag features for ML (salary 1 and 2 years ago)
    df["salary_lag1"] = df.groupby("job_role")["avg_salary_usd"].shift(1)
    df["salary_lag2"] = df.groupby("job_role")["avg_salary_usd"].shift(2)

    df.to_csv(f"{PROC_DIR}/salary_history_clean.csv", index=False)
    report(df, "Processed salary_history_clean")
    print(f"  💾 Saved salary_history_clean.csv ({len(df):,} rows)")
    return df


# ============================================================
# STEP 4 — Process Role-Skill Mapping
# ============================================================

def process_role_skill_mapping():
    print("\n" + "─" * 50)
    print("  STEP 4: Processing role_skill_mapping.csv")
    print("─" * 50)

    df = pd.read_csv(f"{RAW_DIR}/role_skill_mapping.csv")
    report(df, "Raw role_skill_mapping")

    # ── 4.1 Clip
    df["trend_score"] = df["trend_score"].clip(0.5, 10.0)
    df["salary_boost_usd"] = df["salary_boost_usd"].clip(lower=0)
    df["automation_risk_impact"] = df["automation_risk_impact"].clip(-1.0, 1.0)

    # ── 4.2 Skill priority score for recommendations
    # Higher trend + higher salary boost + lower auto risk = higher priority
    df["priority_score"] = (
        df["trend_score"] * 0.5
        + (df["salary_boost_usd"] / 5000) * 0.3
        + (df["automation_risk_impact"] * -10) * 0.2
    ).round(3)

    # ── 4.3 Rank skills within each role
    df["skill_rank"] = (
        df.groupby("job_role")["priority_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # ── 4.4 Mark top-3 recommended skills per role
    df["is_top_recommendation"] = (df["skill_rank"] <= 3).astype(int)

    df.to_csv(f"{PROC_DIR}/role_skill_mapping_clean.csv", index=False)
    report(df, "Processed role_skill_mapping_clean")
    print(f"  💾 Saved role_skill_mapping_clean.csv ({len(df):,} rows)")
    return df


# ============================================================
# STEP 5 — Build ML Feature Matrix
# ============================================================

def build_ml_feature_matrix(df_jobs, df_salary):
    """
    Merge job-level data with salary CAGR and build
    the final feature matrix for ML models.
    """
    print("\n" + "─" * 50)
    print("  STEP 5: Building ML Feature Matrix")
    print("─" * 50)

    # Get latest salary CAGR per role from salary data
    cagr_lookup = (
        df_salary[["job_role", "salary_cagr_5yr"]]
        .drop_duplicates("job_role")
    )

    df = df_jobs.merge(cagr_lookup, on="job_role", how="left")

    # Select and order feature columns
    feature_cols = [
        "year",
        "experience_ord",
        "role_encoded",
        "num_skills",
        "skill_count",
        "avg_skill_trend_score",
        "demand_index",
        "salary_vs_median",
        "salary_cagr_5yr",
        "future_proof",
    ]

    target_cols = [
        "salary_usd",
        "automation_risk",
    ]

    ml_df = df[feature_cols + target_cols + ["job_role", "experience_level", "risk_label"]].copy()
    ml_df.dropna(subset=feature_cols, inplace=True)
    ml_df.reset_index(drop=True, inplace=True)

    ml_df.to_csv(f"{PROC_DIR}/ml_feature_matrix.csv", index=False)
    report(ml_df, "ML Feature Matrix")
    print(f"  💾 Saved ml_feature_matrix.csv ({len(ml_df):,} rows)")
    print(f"  🎯 Features: {feature_cols}")
    print(f"  🏷  Targets:  {target_cols}")
    return ml_df


# ============================================================
# STEP 6 — Summary Statistics Export
# ============================================================

def export_summary_stats(df_jobs, df_skills):
    """
    Export aggregated stats used by the dashboard's Home page.
    """
    print("\n" + "─" * 50)
    print("  STEP 6: Exporting Summary Statistics")
    print("─" * 50)

    # Per-role summary (2025 snapshot = latest year)
    latest = df_jobs[df_jobs["year"] == df_jobs["year"].max()]
    role_summary = (
        latest.groupby("job_role")
        .agg(
            avg_salary=("salary_usd", "mean"),
            avg_auto_risk=("automation_risk", "mean"),
            avg_demand=("demand_index", "mean"),
            avg_trend=("avg_skill_trend_score", "mean"),
            count=("id", "count"),
        )
        .round(2)
        .reset_index()
    )

    # Career Survival Score preview (computed fully in Part 6)
    # Preview formula: demand * trend * (1 - auto_risk)
    role_summary["survival_preview"] = (
        (role_summary["avg_demand"] / 100) *
        (role_summary["avg_trend"] / 10) *
        (1 - role_summary["avg_auto_risk"])
    ).round(3)

    role_summary.to_csv(f"{PROC_DIR}/role_summary_stats.csv", index=False)
    print(f"  💾 Saved role_summary_stats.csv ({len(role_summary)} roles)")

    # Top rising / declining skills (latest year)
    skill_latest = df_skills[df_skills["year"] == df_skills["year"].max()]
    skill_latest = skill_latest.sort_values("trend_score", ascending=False)
    skill_latest.to_csv(f"{PROC_DIR}/skills_latest_snapshot.csv", index=False)
    print(f"  💾 Saved skills_latest_snapshot.csv ({len(skill_latest)} skills)")

    return role_summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🔧 Career Obsolescence Intelligence System")
    print("     Data Preprocessing Pipeline")
    print("=" * 55)

    df_jobs   = process_job_roles()
    df_skills = process_skills_trend()
    df_salary = process_salary_history()
    df_mapping = process_role_skill_mapping()
    df_ml     = build_ml_feature_matrix(df_jobs, df_salary)
    summary   = export_summary_stats(df_jobs, df_skills)

    print("\n" + "=" * 55)
    print("  ✅ PREPROCESSING COMPLETE!")
    print(f"  📁 Outputs in: data/processed/")
    print("=" * 55)
    print("\n  Files ready:")
    print("  • job_roles_clean.csv")
    print("  • jobs_skills_exploded.csv")
    print("  • skills_trend_clean.csv")
    print("  • salary_history_clean.csv")
    print("  • role_skill_mapping_clean.csv")
    print("  • ml_feature_matrix.csv")
    print("  • role_summary_stats.csv")
    print("  • skills_latest_snapshot.csv")
    print("  • role_encoding_map.json")
    print("\n  ➡️  Next: python models/skill_trend_model.py\n")
