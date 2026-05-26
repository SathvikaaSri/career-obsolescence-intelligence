import pandas as pd
import numpy as np
import os
import random
from datetime import datetime

# Reproducibility
np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────
# OUTPUT PATHS
# ─────────────────────────────────────────────
RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# MASTER DATA DEFINITIONS
# ─────────────────────────────────────────────

JOB_ROLES = [
    "Data Analyst", "Business Analyst", "Data Scientist",
    "Machine Learning Engineer", "Data Engineer", "BI Developer",
    "SQL Developer", "Excel Analyst", "Statistician",
    "AI Engineer", "Cloud Architect", "DevOps Engineer",
    "Software Developer", "QA Engineer", "IT Support Specialist",
    "Digital Marketer", "Content Strategist", "SEO Specialist",
    "Financial Analyst", "Accountant", "HR Analyst",
    "Supply Chain Analyst", "Operations Analyst", "Product Manager",
]

# Each skill tagged with: trend_score (1-10), automation_risk_contribution
SKILLS_CATALOG = {
    # Rising / Future-proof
    "Python":           {"trend": 9.2, "auto_risk": -0.15, "salary_boost": 18000},
    "Machine Learning": {"trend": 9.5, "auto_risk": -0.20, "salary_boost": 25000},
    "Deep Learning":    {"trend": 9.1, "auto_risk": -0.18, "salary_boost": 28000},
    "Cloud (AWS/GCP)":  {"trend": 9.0, "auto_risk": -0.12, "salary_boost": 22000},
    "Spark/PySpark":    {"trend": 8.7, "auto_risk": -0.10, "salary_boost": 20000},
    "LLMs/GenAI":       {"trend": 9.8, "auto_risk": -0.22, "salary_boost": 30000},
    "dbt":              {"trend": 8.5, "auto_risk": -0.08, "salary_boost": 15000},
    "Airflow":          {"trend": 8.3, "auto_risk": -0.07, "salary_boost": 14000},
    "Kubernetes":       {"trend": 8.1, "auto_risk": -0.09, "salary_boost": 19000},
    "Prompt Engineering":{"trend": 9.3, "auto_risk": -0.18, "salary_boost": 21000},
    "MLOps":            {"trend": 9.0, "auto_risk": -0.17, "salary_boost": 24000},
    "Vector Databases": {"trend": 9.1, "auto_risk": -0.16, "salary_boost": 22000},
    "SQL":              {"trend": 7.5, "auto_risk": -0.05, "salary_boost": 8000},
    "Power BI":         {"trend": 7.8, "auto_risk": -0.04, "salary_boost": 9000},
    "Tableau":          {"trend": 7.6, "auto_risk": -0.04, "salary_boost": 9500},
    "Statistics":       {"trend": 7.4, "auto_risk": -0.06, "salary_boost": 11000},
    "A/B Testing":      {"trend": 7.9, "auto_risk": -0.07, "salary_boost": 13000},
    "R":                {"trend": 6.5, "auto_risk": -0.03, "salary_boost": 7000},
    # Declining / Automatable
    "Excel (Basic)":    {"trend": 3.5, "auto_risk": 0.30,  "salary_boost": 2000},
    "Manual Testing":   {"trend": 2.8, "auto_risk": 0.45,  "salary_boost": 1000},
    "Data Entry":       {"trend": 1.5, "auto_risk": 0.75,  "salary_boost": 0},
    "Legacy VBA":       {"trend": 2.0, "auto_risk": 0.60,  "salary_boost": 500},
    "Crystal Reports":  {"trend": 1.8, "auto_risk": 0.55,  "salary_boost": 500},
    "MS Access":        {"trend": 2.2, "auto_risk": 0.50,  "salary_boost": 1000},
    "Cold Calling":     {"trend": 2.0, "auto_risk": 0.65,  "salary_boost": 0},
    "COBOL":            {"trend": 1.2, "auto_risk": 0.40,  "salary_boost": 3000},
    "Basic SEO":        {"trend": 3.0, "auto_risk": 0.55,  "salary_boost": 1500},
    "Copy-Paste Reporting": {"trend": 1.0, "auto_risk": 0.90, "salary_boost": 0},
}

# Base salaries per role (USD, annual)
ROLE_BASE_SALARIES = {
    "Data Analyst": 72000,
    "Business Analyst": 75000,
    "Data Scientist": 110000,
    "Machine Learning Engineer": 135000,
    "Data Engineer": 120000,
    "BI Developer": 85000,
    "SQL Developer": 78000,
    "Excel Analyst": 58000,
    "Statistician": 90000,
    "AI Engineer": 145000,
    "Cloud Architect": 140000,
    "DevOps Engineer": 115000,
    "Software Developer": 105000,
    "QA Engineer": 82000,
    "IT Support Specialist": 55000,
    "Digital Marketer": 65000,
    "Content Strategist": 62000,
    "SEO Specialist": 60000,
    "Financial Analyst": 80000,
    "Accountant": 68000,
    "HR Analyst": 64000,
    "Supply Chain Analyst": 74000,
    "Operations Analyst": 72000,
    "Product Manager": 115000,
}

# Base automation risk per role (0.0 = safe, 1.0 = fully automatable)
ROLE_BASE_AUTO_RISK = {
    "Data Analyst": 0.35,
    "Business Analyst": 0.30,
    "Data Scientist": 0.15,
    "Machine Learning Engineer": 0.10,
    "Data Engineer": 0.18,
    "BI Developer": 0.38,
    "SQL Developer": 0.42,
    "Excel Analyst": 0.68,
    "Statistician": 0.22,
    "AI Engineer": 0.08,
    "Cloud Architect": 0.12,
    "DevOps Engineer": 0.20,
    "Software Developer": 0.25,
    "QA Engineer": 0.55,
    "IT Support Specialist": 0.60,
    "Digital Marketer": 0.45,
    "Content Strategist": 0.40,
    "SEO Specialist": 0.58,
    "Financial Analyst": 0.38,
    "Accountant": 0.62,
    "HR Analyst": 0.44,
    "Supply Chain Analyst": 0.32,
    "Operations Analyst": 0.40,
    "Product Manager": 0.18,
}

YEARS = list(range(2019, 2026))
EXPERIENCE_LEVELS = ["Entry", "Mid", "Senior", "Lead", "Principal"]
INDUSTRIES = ["Tech", "Finance", "Healthcare", "Retail", "Manufacturing",
              "Education", "Government", "Consulting", "Media", "Energy"]
LOCATIONS = ["USA", "UK", "India", "Canada", "Germany", "Australia",
             "Singapore", "UAE", "Netherlands", "Brazil"]


# ─────────────────────────────────────────────
# DATASET 1: Job Roles Dataset
# ─────────────────────────────────────────────
def generate_job_roles_dataset(n=2000):
    """
    Generates a dataset of job postings with:
    - role, skills, salary, demand trend, automation risk, year
    """
    print("⚙️  Generating job_roles_dataset.csv ...")
    rows = []

    for i in range(n):
        year = random.choice(YEARS)
        role = random.choice(JOB_ROLES)
        experience = random.choice(EXPERIENCE_LEVELS)
        industry = random.choice(INDUSTRIES)
        location = random.choice(LOCATIONS)

        # Pick 3–6 skills per role
        num_skills = random.randint(3, 6)
        skill_pool = list(SKILLS_CATALOG.keys())
        chosen_skills = random.sample(skill_pool, num_skills)

        # Compute salary
        base_sal = ROLE_BASE_SALARIES[role]
        exp_multiplier = {"Entry": 0.75, "Mid": 1.0, "Senior": 1.30,
                          "Lead": 1.55, "Principal": 1.80}[experience]
        skill_bonus = sum(SKILLS_CATALOG[s]["salary_boost"] for s in chosen_skills)
        year_growth = (year - 2019) * 0.025  # 2.5% annual growth
        noise = np.random.normal(0, 5000)
        salary = int((base_sal + skill_bonus) * exp_multiplier * (1 + year_growth) + noise)
        salary = max(30000, salary)

        # Compute automation risk
        base_risk = ROLE_BASE_AUTO_RISK[role]
        skill_risk_adj = sum(SKILLS_CATALOG[s]["auto_risk"] for s in chosen_skills)
        year_risk_change = (year - 2019) * 0.015  # risk increases slightly each year
        auto_risk = base_risk + skill_risk_adj + year_risk_change + np.random.normal(0, 0.03)
        auto_risk = round(np.clip(auto_risk, 0.05, 0.98), 3)

        # Demand trend (job postings volume index, 100 = baseline 2019)
        demand_index = 100 + (year - 2019) * random.uniform(2, 12) + np.random.normal(0, 5)
        # Roles losing demand get a penalty
        if role in ["Excel Analyst", "IT Support Specialist", "SQL Developer", "QA Engineer"]:
            demand_index -= (year - 2019) * 5
        demand_index = round(max(20, demand_index), 1)

        # Skill trend scores (average of chosen skills)
        avg_trend = round(np.mean([SKILLS_CATALOG[s]["trend"] for s in chosen_skills]), 2)

        rows.append({
            "id": i + 1,
            "year": year,
            "job_role": role,
            "experience_level": experience,
            "industry": industry,
            "location": location,
            "skills": "|".join(chosen_skills),
            "num_skills": num_skills,
            "salary_usd": salary,
            "automation_risk": auto_risk,
            "demand_index": demand_index,
            "avg_skill_trend_score": avg_trend,
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{RAW_DIR}/job_roles_dataset.csv", index=False)
    print(f"   ✅ Saved {len(df)} rows → {RAW_DIR}/job_roles_dataset.csv")
    return df


# ─────────────────────────────────────────────
# DATASET 2: Skills Trend Dataset
# ─────────────────────────────────────────────
def generate_skills_trend_dataset():
    """
    Yearly trend score per skill from 2019–2025.
    """
    print("⚙️  Generating skills_trend_dataset.csv ...")
    rows = []

    for skill, meta in SKILLS_CATALOG.items():
        base_trend = meta["trend"]
        for year in YEARS:
            delta = (year - 2019) * random.uniform(-0.05, 0.15)
            # Declining skills lose more each year
            if base_trend < 5:
                delta -= (year - 2019) * 0.15
            trend_score = round(np.clip(base_trend + delta + np.random.normal(0, 0.1), 0.5, 10.0), 2)
            # Job postings mentioning this skill
            postings = int(1000 * (trend_score / 10) * (1 + (year - 2019) * 0.08)
                           + np.random.normal(0, 50))
            postings = max(10, postings)

            rows.append({
                "skill": skill,
                "year": year,
                "trend_score": trend_score,
                "job_postings_count": postings,
                "category": "Rising" if base_trend >= 7 else ("Stable" if base_trend >= 5 else "Declining"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{RAW_DIR}/skills_trend_dataset.csv", index=False)
    print(f"   ✅ Saved {len(df)} rows → {RAW_DIR}/skills_trend_dataset.csv")
    return df


# ─────────────────────────────────────────────
# DATASET 3: Salary History Dataset
# ─────────────────────────────────────────────
def generate_salary_history_dataset():
    """
    Average salary per role per year for time-series forecasting.
    """
    print("⚙️  Generating salary_history_dataset.csv ...")
    rows = []

    for role in JOB_ROLES:
        base = ROLE_BASE_SALARIES[role]
        for year in YEARS:
            growth_rate = random.uniform(0.02, 0.07)
            # AI/data roles grow faster
            if role in ["AI Engineer", "Machine Learning Engineer", "Data Scientist", "Data Engineer"]:
                growth_rate = random.uniform(0.07, 0.13)
            # Declining roles shrink or stagnate
            if role in ["Excel Analyst", "IT Support Specialist", "QA Engineer"]:
                growth_rate = random.uniform(-0.01, 0.02)

            salary = int(base * ((1 + growth_rate) ** (year - 2019)) + np.random.normal(0, 3000))
            salary = max(30000, salary)

            rows.append({
                "job_role": role,
                "year": year,
                "avg_salary_usd": salary,
                "growth_rate": round(growth_rate, 4),
                "base_automation_risk": ROLE_BASE_AUTO_RISK[role],
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{RAW_DIR}/salary_history_dataset.csv", index=False)
    print(f"   ✅ Saved {len(df)} rows → {RAW_DIR}/salary_history_dataset.csv")
    return df


# ─────────────────────────────────────────────
# DATASET 4: Role-Skill Mapping Dataset
# ─────────────────────────────────────────────
def generate_role_skill_mapping():
    """
    Maps which skills are commonly required per role.
    Used by the recommendation engine.
    """
    print("⚙️  Generating role_skill_mapping.csv ...")

    # Curated mappings (top skills per role)
    mappings = {
        "Data Analyst":           ["SQL", "Python", "Power BI", "Tableau", "Statistics", "Excel (Basic)"],
        "Business Analyst":       ["SQL", "Power BI", "Statistics", "Excel (Basic)", "A/B Testing"],
        "Data Scientist":         ["Python", "Machine Learning", "Statistics", "SQL", "Deep Learning", "A/B Testing"],
        "Machine Learning Engineer": ["Python", "Machine Learning", "Deep Learning", "MLOps", "Cloud (AWS/GCP)", "Spark/PySpark"],
        "Data Engineer":          ["Python", "SQL", "Spark/PySpark", "Airflow", "dbt", "Cloud (AWS/GCP)"],
        "BI Developer":           ["Power BI", "Tableau", "SQL", "Statistics", "dbt"],
        "SQL Developer":          ["SQL", "MS Access", "Crystal Reports", "Legacy VBA"],
        "Excel Analyst":          ["Excel (Basic)", "Legacy VBA", "MS Access", "Crystal Reports"],
        "Statistician":           ["Statistics", "R", "Python", "A/B Testing"],
        "AI Engineer":            ["Python", "Deep Learning", "LLMs/GenAI", "Machine Learning", "Prompt Engineering", "MLOps", "Vector Databases"],
        "Cloud Architect":        ["Cloud (AWS/GCP)", "Kubernetes", "Python", "DevOps Engineer"],
        "DevOps Engineer":        ["Kubernetes", "Cloud (AWS/GCP)", "Python", "Airflow"],
        "Software Developer":     ["Python", "SQL", "Cloud (AWS/GCP)", "Kubernetes"],
        "QA Engineer":            ["Manual Testing", "Python", "SQL"],
        "IT Support Specialist":  ["MS Access", "Legacy VBA", "Excel (Basic)", "Cold Calling"],
        "Digital Marketer":       ["Basic SEO", "Cold Calling", "A/B Testing", "Statistics"],
        "Content Strategist":     ["Basic SEO", "Cold Calling", "Prompt Engineering"],
        "SEO Specialist":         ["Basic SEO", "Statistics", "A/B Testing"],
        "Financial Analyst":      ["Excel (Basic)", "SQL", "Statistics", "Python", "Power BI"],
        "Accountant":             ["Excel (Basic)", "Legacy VBA", "MS Access", "Copy-Paste Reporting"],
        "HR Analyst":             ["Excel (Basic)", "SQL", "Statistics", "Power BI"],
        "Supply Chain Analyst":   ["SQL", "Python", "Statistics", "Excel (Basic)", "Power BI"],
        "Operations Analyst":     ["SQL", "Excel (Basic)", "Statistics", "Power BI"],
        "Product Manager":        ["SQL", "A/B Testing", "Statistics", "Python", "LLMs/GenAI"],
    }

    rows = []
    for role, skills in mappings.items():
        for skill in skills:
            if skill in SKILLS_CATALOG:
                rows.append({
                    "job_role": role,
                    "skill": skill,
                    "trend_score": SKILLS_CATALOG[skill]["trend"],
                    "salary_boost_usd": SKILLS_CATALOG[skill]["salary_boost"],
                    "automation_risk_impact": SKILLS_CATALOG[skill]["auto_risk"],
                    "category": "Rising" if SKILLS_CATALOG[skill]["trend"] >= 7
                                else ("Stable" if SKILLS_CATALOG[skill]["trend"] >= 5 else "Declining"),
                })

    df = pd.DataFrame(rows)
    df.to_csv(f"{RAW_DIR}/role_skill_mapping.csv", index=False)
    print(f"   ✅ Saved {len(df)} rows → {RAW_DIR}/role_skill_mapping.csv")
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🚀 Career Obsolescence Intelligence System")
    print("     Dataset Generator")
    print("=" * 55 + "\n")

    df1 = generate_job_roles_dataset(n=2000)
    df2 = generate_skills_trend_dataset()
    df3 = generate_salary_history_dataset()
    df4 = generate_role_skill_mapping()

    print("\n" + "=" * 55)
    print("  ✅ ALL DATASETS GENERATED SUCCESSFULLY!")
    print(f"  📁 Location: {RAW_DIR}/")
    print("=" * 55)
    print("\n  Files created:")
    print("  • job_roles_dataset.csv     — 2000 job records")
    print("  • skills_trend_dataset.csv  — Yearly skill trends")
    print("  • salary_history_dataset.csv — Salary over time")
    print("  • role_skill_mapping.csv    — Role→Skill relationships")
    print("\n  ➡️  Next: Run python data_preprocessing.py\n")
