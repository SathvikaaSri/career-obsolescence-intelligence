# ============================================================
# Career Obsolescence Intelligence System
# FILE: models/salary_prediction_model.py
# PURPOSE: Forecast future salary trajectory per role
#          Models: GBR for point prediction + linear projection
# RUN: python models/salary_prediction_model.py
# ============================================================

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PROC_DIR  = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# STEP 1 — Load Data
# ============================================================
def load_data():
    print("\n  📥 Loading data ...")
    df_ml  = pd.read_csv(f"{PROC_DIR}/ml_feature_matrix.csv")
    df_sal = pd.read_csv(f"{PROC_DIR}/salary_history_clean.csv")

    print(f"  ✅ ML matrix:       {df_ml.shape[0]:,} rows")
    print(f"  ✅ Salary history:  {df_sal.shape[0]:,} rows, "
          f"{df_sal['job_role'].nunique()} roles")
    return df_ml, df_sal


# ============================================================
# STEP 2 — Train Salary Regression Model
# Predicts individual-level salary from job features
# ============================================================
def train_salary_model(df_ml):
    print("\n  🏋️  Training Salary Regression Model ...")

    FEATURE_COLS = [
        "year",
        "experience_ord",
        "role_encoded",
        "num_skills",
        "avg_skill_trend_score",
        "demand_index",
        "automation_risk",       # riskier roles tend to pay less over time
        "salary_cagr_5yr",
        "future_proof",
    ]
    TARGET = "salary_usd"

    df = df_ml.dropna(subset=FEATURE_COLS + [TARGET])
    X  = df[FEATURE_COLS]
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        min_samples_leaf=4,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X, y, cv=5,
                                scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    print(f"\n  ── GradientBoosting Salary Model")
    print(f"     MAE:    ${mae:,.0f}  |  RMSE: ${rmse:,.0f}  |  R²: {r2:.4f}")
    print(f"     CV-MAE: ${cv_mae:,.0f} (5-fold)")

    # Feature importance
    fi = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi["importance_pct"] = (fi["importance"] / fi["importance"].sum() * 100).round(2)
    print("\n  📊 Feature Importance:")
    print(fi[["feature", "importance_pct"]].to_string(index=False))
    fi.to_csv(f"{PROC_DIR}/salary_feature_importance.csv", index=False)

    return model, FEATURE_COLS, {"mae": mae, "rmse": rmse, "r2": r2, "cv_mae": cv_mae}


# ============================================================
# STEP 3 — Per-Role Time-Series Salary Forecast (2026–2030)
# Uses linear regression on historical salary per role
# ============================================================
def forecast_role_salaries(df_sal):
    print("\n  🔮 Forecasting per-role salaries (2026–2030) ...")

    forecast_rows = []
    FORECAST_YEARS = [2026, 2027, 2028, 2029, 2030]

    for role, group in df_sal.groupby("job_role"):
        group = group.sort_values("year")
        X = group["year"].values.reshape(-1, 1)
        y = group["avg_salary_usd"].values

        # Fit linear trend
        lr = LinearRegression()
        lr.fit(X, y)
        slope     = lr.coef_[0]
        intercept = lr.intercept_
        r2        = lr.score(X, y)

        # Historical rows
        for _, row in group.iterrows():
            forecast_rows.append({
                "job_role":       role,
                "year":           int(row["year"]),
                "salary_usd":     round(row["avg_salary_usd"], 0),
                "type":           "historical",
                "growth_tier":    row.get("growth_tier", "Unknown"),
                "salary_cagr":    round(row.get("salary_cagr_5yr", 0), 4),
            })

        # Projected rows
        last_salary = group["avg_salary_usd"].iloc[-1]
        for yr in FORECAST_YEARS:
            proj = max(25000, slope * yr + intercept)
            # Add slight acceleration for high-growth roles
            growth_tier = group["growth_tier"].iloc[-1]
            if growth_tier == "High Growth":
                proj *= (1 + 0.01 * (yr - 2025))  # small extra bump
            elif growth_tier == "Stagnant/Declining":
                proj *= (1 - 0.005 * (yr - 2025))  # small drag

            proj = max(25000, round(proj, 0))
            forecast_rows.append({
                "job_role":    role,
                "year":        yr,
                "salary_usd":  proj,
                "type":        "projected",
                "growth_tier": growth_tier,
                "salary_cagr": round(group.get("salary_cagr_5yr", pd.Series([0])).iloc[-1], 4),
            })

    df_forecast = pd.DataFrame(forecast_rows)
    df_forecast.to_csv(f"{PROC_DIR}/salary_forecast.csv", index=False)
    print(f"  💾 Saved salary_forecast.csv ({len(df_forecast):,} rows, "
          f"years 2019–2030)")
    return df_forecast


# ============================================================
# STEP 4 — Salary Trajectory Summary Table
# Pivot: role × year with projected salaries
# ============================================================
def build_salary_trajectory_table(df_forecast):
    print("\n  📋 Building salary trajectory table ...")

    pivot = df_forecast.pivot_table(
        index="job_role",
        columns="year",
        values="salary_usd",
        aggfunc="mean"
    ).round(0).reset_index()

    # Compute 2025→2030 expected growth %
    if 2025 in pivot.columns and 2030 in pivot.columns:
        pivot["growth_2025_to_2030_pct"] = (
            ((pivot[2030] - pivot[2025]) / pivot[2025]) * 100
        ).round(2)

    # Rank roles by projected 2030 salary
    if 2030 in pivot.columns:
        pivot["rank_by_2030_salary"] = pivot[2030].rank(
            ascending=False, method="min"
        ).astype(int)

    pivot.to_csv(f"{PROC_DIR}/salary_trajectory_table.csv", index=False)
    print(f"  💾 Saved salary_trajectory_table.csv")

    # Show preview
    show_cols = ["job_role"] + [c for c in [2023, 2025, 2028, 2030,
                 "growth_2025_to_2030_pct", "rank_by_2030_salary"]
                 if c in pivot.columns]
    print("\n  💰 Salary Trajectory Preview (Top 8 by 2030):")
    if "rank_by_2030_salary" in pivot.columns:
        top8 = pivot.sort_values("rank_by_2030_salary").head(8)
        print(top8[show_cols].to_string(index=False))

    return pivot


# ============================================================
# STEP 5 — Predict Salary for Custom Input (dashboard use)
# ============================================================
def predict_salary(model, feature_cols,
                   role_encoded, experience_ord,
                   avg_skill_trend_score, num_skills,
                   demand_index, automation_risk,
                   salary_cagr_5yr, future_proof=0,
                   year=2025):
    """
    Single-row salary prediction for the Streamlit dashboard.

    Returns:
        salary (float): predicted annual salary in USD
        band_low (float): lower estimate (–8%)
        band_high (float): upper estimate (+12%)
    """
    row = pd.DataFrame([{
        "year":                   year,
        "experience_ord":         experience_ord,
        "role_encoded":           role_encoded,
        "num_skills":             num_skills,
        "avg_skill_trend_score":  avg_skill_trend_score,
        "demand_index":           demand_index,
        "automation_risk":        automation_risk,
        "salary_cagr_5yr":        salary_cagr_5yr,
        "future_proof":           future_proof,
    }])[feature_cols]

    salary    = float(max(25000, model.predict(row)[0]))
    band_low  = round(salary * 0.92, 0)
    band_high = round(salary * 1.12, 0)
    return round(salary, 0), band_low, band_high


# ============================================================
# STEP 6 — Save Artifacts
# ============================================================
def save_artifacts(model, feature_cols, metrics):
    print("\n  💾 Saving salary model artifacts ...")

    joblib.dump(model, f"{MODEL_DIR}/salary_model.pkl")
    print(f"  ✅ Saved salary_model.pkl")

    meta = {
        "model_type":   "GradientBoostingRegressor",
        "feature_cols": feature_cols,
        "metrics": {
            "mae":    round(metrics["mae"], 2),
            "rmse":   round(metrics["rmse"], 2),
            "r2":     round(metrics["r2"], 4),
            "cv_mae": round(metrics["cv_mae"], 2),
        },
        "target":   "salary_usd",
        "version":  "1.0",
    }
    with open(f"{MODEL_DIR}/salary_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✅ Saved salary_model_meta.json")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  💰 Salary Prediction Model")
    print("     Career Obsolescence Intelligence System")
    print("=" * 55)

    df_ml, df_sal      = load_data()
    model, feat_cols, metrics = train_salary_model(df_ml)
    df_forecast        = forecast_role_salaries(df_sal)
    pivot              = build_salary_trajectory_table(df_forecast)
    save_artifacts(model, feat_cols, metrics)

    # Sanity check — predict salary for ML Engineer (Senior, 2025)
    print("\n  🧪 Sanity Check — ML Engineer (Senior, 2025):")
    with open(f"{PROC_DIR}/role_encoding_map.json") as f:
        role_map = json.load(f)

    sal, low, high = predict_salary(
        model=model,
        feature_cols=feat_cols,
        role_encoded=role_map.get("Machine Learning Engineer", 15),
        experience_ord=3,
        avg_skill_trend_score=9.0,
        num_skills=5,
        demand_index=145,
        automation_risk=0.12,
        salary_cagr_5yr=0.10,
        future_proof=1,
        year=2025,
    )
    print(f"     Predicted Salary:  ${sal:,.0f}")
    print(f"     Salary Band:       ${low:,.0f} – ${high:,.0f}")

    print("\n" + "=" * 55)
    print("  ✅ SALARY PREDICTION MODEL COMPLETE!")
    print("=" * 55)
    print("\n  Outputs:")
    print("  • models/salary_model.pkl")
    print("  • models/salary_model_meta.json")
    print("  • data/processed/salary_forecast.csv")
    print("  • data/processed/salary_trajectory_table.csv")
    print("  • data/processed/salary_feature_importance.csv")
    print("\n  ➡️  Next: python models/career_survival_score.py\n")
