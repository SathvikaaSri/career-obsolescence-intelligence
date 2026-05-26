# ============================================================
# Career Obsolescence Intelligence System
# FILE: models/automation_risk_model.py
# PURPOSE: Train ML model to predict automation risk (0–1)
#          for any job role + experience + skill combination
# RUN: python models/automation_risk_model.py
# ============================================================

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
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
# STEP 1 — Load & Prepare Features
# ============================================================
def load_features():
    print("\n  📥 Loading ML feature matrix ...")
    df = pd.read_csv(f"{PROC_DIR}/ml_feature_matrix.csv")

    # Feature set for automation risk prediction
    FEATURE_COLS = [
        "year",
        "experience_ord",        # Entry=1 … Principal=5
        "role_encoded",          # integer-encoded role
        "num_skills",            # number of skills listed
        "skill_count",
        "avg_skill_trend_score", # how future-proof the skills are
        "demand_index",          # market demand
        "salary_cagr_5yr",       # salary growth rate
        "future_proof",          # binary flag
    ]

    TARGET = "automation_risk"

    df = df.dropna(subset=FEATURE_COLS + [TARGET])
    X = df[FEATURE_COLS]
    y = df[TARGET]

    print(f"  ✅ Dataset: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  🎯 Target range: {y.min():.3f} – {y.max():.3f}  |  mean: {y.mean():.3f}")
    return X, y, df, FEATURE_COLS


# ============================================================
# STEP 2 — Train / Evaluate Multiple Models
# ============================================================
def train_and_evaluate(X, y):
    print("\n  🏋️  Training models ...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            min_samples_leaf=5,
            random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        ),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=1.0)),
        ]),
    }

    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0.0, 1.0)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        # 5-fold CV on full dataset
        cv_scores = cross_val_score(model, X, y, cv=kf,
                                    scoring="neg_mean_absolute_error")
        cv_mae = -cv_scores.mean()

        results[name] = {
            "model":   model,
            "mae":     round(mae, 4),
            "rmse":    round(rmse, 4),
            "r2":      round(r2, 4),
            "cv_mae":  round(cv_mae, 4),
        }

        print(f"\n  ── {name}")
        print(f"     MAE:    {mae:.4f}  |  RMSE: {rmse:.4f}  |  R²: {r2:.4f}")
        print(f"     CV-MAE: {cv_mae:.4f} (5-fold)")

    return results, X_train, X_test, y_train, y_test


# ============================================================
# STEP 3 — Select Best Model
# ============================================================
def select_best_model(results):
    print("\n  🏆 Selecting best model by CV-MAE ...")
    best_name = min(results, key=lambda k: results[k]["cv_mae"])
    best = results[best_name]
    print(f"  ✅ Winner: {best_name}")
    print(f"     CV-MAE={best['cv_mae']}  |  R²={best['r2']}")
    return best_name, best["model"]


# ============================================================
# STEP 4 — Feature Importance
# ============================================================
def compute_feature_importance(model, feature_cols, model_name):
    print("\n  📊 Feature Importance:")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        # Ridge pipeline
        coefs = np.abs(model.named_steps["ridge"].coef_)
        importances = coefs / coefs.sum()
    else:
        print("  ⚠️  Feature importance not available for this model.")
        return None

    fi_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fi_df["importance_pct"] = (fi_df["importance"] / fi_df["importance"].sum() * 100).round(2)

    print(fi_df[["feature", "importance_pct"]].to_string(index=False))
    fi_df.to_csv(f"{PROC_DIR}/automation_risk_feature_importance.csv", index=False)

    return fi_df


# ============================================================
# STEP 5 — Generate Role-Level Risk Scores
# ============================================================
def generate_role_risk_scores(model, df, feature_cols):
    """
    Use the trained model to predict automation risk
    for each unique role × experience combination.
    """
    print("\n  🔮 Generating role-level risk scores ...")

    group_cols = ["job_role", "experience_level", "experience_ord", "role_encoded"]
    # Only aggregate feature cols that are NOT already in the group keys
    agg_cols = [c for c in feature_cols if c not in group_cols]
    role_exp_df = (
        df.groupby(group_cols)[agg_cols]
        .mean()
        .reset_index()
    )

    X_role = role_exp_df[feature_cols]
    role_exp_df["predicted_auto_risk"] = np.clip(model.predict(X_role), 0.0, 1.0).round(4)

    # Risk tier
    def risk_tier(r):
        if r < 0.20: return "🟢 Very Safe"
        if r < 0.40: return "🟡 Low Risk"
        if r < 0.60: return "🟠 Moderate Risk"
        if r < 0.80: return "🔴 High Risk"
        return "💀 Critical Risk"

    role_exp_df["risk_tier"] = role_exp_df["predicted_auto_risk"].apply(risk_tier)

    # Human dependency score = inverse of risk
    role_exp_df["human_dependency_score"] = (
        (1 - role_exp_df["predicted_auto_risk"]) * 100
    ).round(2)

    # Save
    out_cols = ["job_role", "experience_level", "predicted_auto_risk",
                "risk_tier", "human_dependency_score"]
    role_risk = role_exp_df[out_cols].sort_values(
        ["job_role", "experience_level"]
    ).reset_index(drop=True)

    role_risk.to_csv(f"{PROC_DIR}/role_automation_risk_scores.csv", index=False)
    print(f"  💾 Saved role_automation_risk_scores.csv ({len(role_risk)} rows)")

    # Print top safe / risky roles (Senior level for fairness)
    senior = role_risk[role_risk["experience_level"] == "Senior"]

    print("\n  🟢 SAFEST ROLES (Senior):")
    print(senior.nsmallest(5, "predicted_auto_risk")[
        ["job_role", "predicted_auto_risk", "risk_tier"]
    ].to_string(index=False))

    print("\n  🔴 RISKIEST ROLES (Senior):")
    print(senior.nlargest(5, "predicted_auto_risk")[
        ["job_role", "predicted_auto_risk", "risk_tier"]
    ].to_string(index=False))

    return role_risk


# ============================================================
# STEP 6 — Prediction Function (used by dashboard)
# ============================================================
def predict_automation_risk(model, feature_cols,
                             role_encoded, experience_ord,
                             avg_skill_trend_score, num_skills,
                             demand_index, salary_cagr_5yr,
                             future_proof=0, year=2025):
    """
    Single-row prediction — called by the Streamlit dashboard.

    Returns:
        risk (float): automation risk 0.0–1.0
        tier (str):   human-readable risk tier
        human_dep (float): 0–100 human dependency score
    """
    row = pd.DataFrame([{
        "year":                   year,
        "experience_ord":         experience_ord,
        "role_encoded":           role_encoded,
        "num_skills":             num_skills,
        "skill_count":            num_skills,
        "avg_skill_trend_score":  avg_skill_trend_score,
        "demand_index":           demand_index,
        "salary_cagr_5yr":        salary_cagr_5yr,
        "future_proof":           future_proof,
    }])[feature_cols]

    risk = float(np.clip(model.predict(row)[0], 0.0, 1.0))

    tier_map = [
        (0.20, "🟢 Very Safe"),
        (0.40, "🟡 Low Risk"),
        (0.60, "🟠 Moderate Risk"),
        (0.80, "🔴 High Risk"),
        (1.01, "💀 Critical Risk"),
    ]
    tier = next(t for threshold, t in tier_map if risk < threshold)
    human_dep = round((1 - risk) * 100, 1)

    return risk, tier, human_dep


# ============================================================
# STEP 7 — Save Model Artifacts
# ============================================================
def save_artifacts(model, model_name, feature_cols, results):
    print("\n  💾 Saving model artifacts ...")

    joblib.dump(model, f"{MODEL_DIR}/automation_risk_model.pkl")
    print(f"  ✅ Saved automation_risk_model.pkl")

    # Save metadata
    meta = {
        "model_type":   model_name,
        "feature_cols": feature_cols,
        "metrics": {
            k: {mk: mv for mk, mv in v.items() if mk != "model"}
            for k, v in results.items()
        },
        "best_model":   model_name,
        "version":      "1.0",
    }
    with open(f"{MODEL_DIR}/automation_risk_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✅ Saved automation_risk_model_meta.json")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🤖 Automation Risk Scoring Model")
    print("     Career Obsolescence Intelligence System")
    print("=" * 55)

    X, y, df, feature_cols = load_features()
    results, X_train, X_test, y_train, y_test = train_and_evaluate(X, y)
    best_name, best_model = select_best_model(results)
    fi_df       = compute_feature_importance(best_model, feature_cols, best_name)
    role_risk   = generate_role_risk_scores(best_model, df, feature_cols)
    save_artifacts(best_model, best_name, feature_cols, results)

    # Quick sanity check — predict for a Data Scientist (Senior)
    print("\n  🧪 Sanity Check — Data Scientist (Senior):")
    with open(f"{PROC_DIR}/role_encoding_map.json") as f:
        role_map = json.load(f)

    risk, tier, hdep = predict_automation_risk(
        model=best_model,
        feature_cols=feature_cols,
        role_encoded=role_map.get("Data Scientist", 5),
        experience_ord=3,        # Senior
        avg_skill_trend_score=8.5,
        num_skills=5,
        demand_index=130,
        salary_cagr_5yr=0.09,
        future_proof=1,
    )
    print(f"     Automation Risk:      {risk:.3f}")
    print(f"     Risk Tier:            {tier}")
    print(f"     Human Dependency:     {hdep}%")

    print("\n" + "=" * 55)
    print("  ✅ AUTOMATION RISK MODEL COMPLETE!")
    print("=" * 55)
    print("\n  Outputs:")
    print("  • models/automation_risk_model.pkl")
    print("  • models/automation_risk_model_meta.json")
    print("  • data/processed/role_automation_risk_scores.csv")
    print("  • data/processed/automation_risk_feature_importance.csv")
    print("\n  ➡️  Next: python models/salary_prediction_model.py\n")
