# Career Obsolescence Intelligence System

> **AI-powered system that predicts dying skills, rising careers, automation probability, salary trajectories, and personalised future-proof recommendations.**

---

#  Dashboard Preview

| Page | Description |
|------|-------------|
|  Home | KPI overview, survival leaderboard, skill snapshot |
|  Role Analysis | Deep-dive per role: radar, salary forecast, survival trajectory |
| Skill Trends | Rising vs declining skills, 2026–2028 projections |
|  Automation Risk | Risk heatmap, live calculator, feature importance |
|  Recommendations | Personalised skill roadmap with salary & survival boost |

# Project Architecture

```
career_obsolescence_intelligence/
│
├── generate_dataset.py          # Step 1: Generate all mock CSVs
├── data_preprocessing.py        # Step 2: Clean + engineer features
│
├── data/
│   ├── raw/                     # Raw generated CSVs
│   └── processed/               # Cleaned + ML-ready data
│
├── models/
│   ├── skill_trend_model.py     # Step 3: Trend slope + projection
│   ├── automation_risk_model.py # Step 4: GBR risk predictor
│   ├── salary_prediction_model.py # Step 5: Salary forecaster
│   ├── career_survival_score.py # Step 6: Weighted survival formula
│   └── recommendation_engine.py # Step 7: Gap-based recommender
│
├── dashboard/
│   └── app.py                   # Step 8: Full Streamlit dashboard
│
├── assets/                      # Screenshots, images
├── notebooks/                   # EDA notebooks (optional)
├── requirements.txt
└── README.md

Open your browser at **http://localhost:8501**


