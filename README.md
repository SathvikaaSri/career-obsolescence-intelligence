# 🧠 Career Obsolescence Intelligence System

> **AI-powered system that predicts dying skills, rising careers, automation probability, salary trajectories, and personalised future-proof recommendations.**

---

## 📸 Dashboard Preview

| Page | Description |
|------|-------------|
| 🏠 Home | KPI overview, survival leaderboard, skill snapshot |
| 🔍 Role Analysis | Deep-dive per role: radar, salary forecast, survival trajectory |
| 📈 Skill Trends | Rising vs declining skills, 2026–2028 projections |
| 🤖 Automation Risk | Risk heatmap, live calculator, feature importance |
| 🎯 Recommendations | Personalised skill roadmap with salary & survival boost |

> _Add screenshots to `/assets/` and link them here after first run._

---

## 🏗️ Project Architecture

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
```

---

## ⚙️ Setup Instructions

### 1. Clone / Download the project

```bash
git clone https://github.com/yourusername/career-obsolescence-intelligence.git
cd career-obsolescence-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline (one-time setup)

```bash
# Generate mock datasets
python generate_dataset.py

# Clean and engineer features
python data_preprocessing.py

# Train all models
python models/skill_trend_model.py
python models/automation_risk_model.py
python models/salary_prediction_model.py
python models/career_survival_score.py
python models/recommendation_engine.py
```

### 5. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open your browser at **http://localhost:8501**

---

## 🧪 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| ML | scikit-learn (GradientBoosting, RandomForest, Ridge, LinearRegression) |
| Visualization | Plotly (interactive charts, heatmaps, radar) |
| Dashboard | Streamlit (multi-page, custom CSS) |
| Storage | CSV files (PostgreSQL-ready schema) |
| Serialization | joblib (model pickling), JSON |

---

## 🧠 ML Models Summary

### Automation Risk Model
- **Algorithm**: GradientBoostingRegressor
- **Target**: `automation_risk` (0.0–1.0)
- **R²**: 0.918 | **CV-MAE**: 0.062
- **Top Features**: `future_proof` (65.8%), `avg_skill_trend_score` (25.5%)

### Salary Prediction Model
- **Algorithm**: GradientBoostingRegressor
- **Target**: `salary_usd`
- **R²**: 0.957 | **CV-MAE**: ~$12,384
- **Top Features**: `experience_ord` (52%), `salary_cagr_5yr` (16%)

### Skill Trend Model
- **Algorithm**: Per-skill LinearRegression (28 models)
- **Output**: Composite trend score (0–100), 5-tier classification, 2026–2028 projections

### Career Survival Score
- **Formula**: `0.30 × Demand + 0.25 × SalaryGrowth + 0.30 × AIResistance + 0.15 × SkillReadiness`
- **Output**: Score 0–100, Grade A+–F, Upskill Urgency flag

---

## 📊 Key Findings (from mock data)

| Insight | Value |
|---------|-------|
| Most future-proof role | AI Engineer (81.4) |
| Most endangered role | Excel Analyst (9.6 💀) |
| Fastest rising skill | LLMs/GenAI (score: 9.8) |
| Fastest declining skill | Copy-Paste Reporting (score: 1.0) |
| Highest salary by 2030 | AI Engineer ~$337K |
| Biggest automation risk | SEO Specialist (1.0) |

---

## 📁 Data Dictionary

### `job_roles_dataset.csv`
| Column | Description |
|--------|-------------|
| job_role | Job title |
| skills | Pipe-separated skill list |
| salary_usd | Annual salary (USD) |
| automation_risk | 0.0=safe, 1.0=automatable |
| demand_index | Job postings index (100=baseline 2019) |
| avg_skill_trend_score | Average trend score of listed skills |

### `career_survival_scores.csv`
| Column | Description |
|--------|-------------|
| career_survival_score | 0–100 composite score |
| survival_grade | A+ Elite → F Endangered |
| upskill_urgency | On Track / Soon / Urgent |
| demand_score | Normalized market demand (0–100) |
| resistance_score | Inverse automation risk (0–100) |

---

## 🔧 Customisation

**Change survival score weights** — edit `models/career_survival_score.py`:
```python
WEIGHTS = {
    "demand":     0.30,
    "salary":     0.25,
    "resistance": 0.30,
    "skill":      0.15,
}
```

**Add a real data source** — replace `generate_dataset.py` with a scraper targeting LinkedIn Jobs, Indeed, or Glassdoor API, keeping the same column schema.

**Add PostgreSQL** — swap `pd.read_csv()` calls with `pd.read_sql()` using `psycopg2`.

---

## 📝 Resume Bullet Points

Use these in your portfolio or CV:

```
• Built a Career Obsolescence Intelligence System using Python, scikit-learn,
  and Streamlit that predicts automation risk (R²=0.92) and salary trajectories
  (R²=0.96) across 24 job roles and 28 skill categories.

• Engineered a composite Career Survival Score (0–100) combining market demand,
  salary CAGR, AI-resistance, and skill trend velocity using MinMax-normalized
  weighted formula; surfaced actionable upskilling urgency flags.

• Developed a personalised recommendation engine that identifies skill gaps
  vs role requirements, ranks learning priorities by salary boost (+$30K) and
  survival score impact, and generates 2026–2028 skill projections.

• Designed and deployed a 5-page interactive Streamlit dashboard with custom CSS
  theming, Plotly radar charts, heatmaps, time-series forecasts, and a live
  automation risk calculator with real-time parameter adjustment.

• Trained and compared 3 ML models (Gradient Boosting, Random Forest, Ridge)
  with 5-fold cross-validation; automated full pipeline from raw CSV generation
  through feature engineering to model serialization (joblib).
```

---

## 🤝 Contributing

PRs welcome. Areas for improvement:
- Plug in real job posting APIs (RapidAPI Indeed, LinkedIn)
- Add NLP skill extraction from job descriptions
- Expand to 100+ roles with clustering
- Add user authentication and saved profiles

---

## 📄 License

MIT License — free to use in your portfolio.

---

*Built with Python · scikit-learn · Plotly · Streamlit*
