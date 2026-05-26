# ============================================================
# Career Obsolescence Intelligence System
# FILE: dashboard/app.py
# PURPOSE: Full Streamlit multi-page dashboard
# RUN: streamlit run dashboard/app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.career_survival_score import calculate_live_survival_score
from models.recommendation_engine import get_recommendations, load_data as load_rec_data

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Career Obsolescence Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME / CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.main { background: #0a0e1a; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #12172b 0%, #1a2240 100%);
    border: 1px solid #2a3560;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #6b7db3;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #e8edff;
    line-height: 1;
}
.metric-sub {
    font-size: 0.78rem;
    color: #4ade80;
    margin-top: 0.3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1224;
    border-right: 1px solid #1e2a4a;
}
section[data-testid="stSidebar"] .stRadio > label {
    color: #8899cc !important;
}

/* Score gauge */
.score-ring {
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
}

/* Skill badge */
.skill-badge {
    display: inline-block;
    background: #1e2a4a;
    border: 1px solid #2a3d6e;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #a0b4e8;
    margin: 3px;
    font-family: 'DM Mono', monospace;
}
.skill-badge.rising { border-color: #4ade80; color: #4ade80; background: #0d2e1a; }
.skill-badge.declining { border-color: #f87171; color: #f87171; background: #2e0d0d; }
.skill-badge.owned { border-color: #60a5fa; color: #60a5fa; background: #0d1e3a; }

/* Section headers */
.section-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #e8edff;
    margin-bottom: 0.2rem;
    border-left: 4px solid #6366f1;
    padding-left: 12px;
}
.section-sub {
    color: #6b7db3;
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
    padding-left: 16px;
    font-family: 'DM Mono', monospace;
}

/* Alert boxes */
.alert-urgent { background:#2e1010; border-left:4px solid #f87171; padding:12px 16px; border-radius:8px; color:#fca5a5; }
.alert-soon   { background:#2e2010; border-left:4px solid #fb923c; padding:12px 16px; border-radius:8px; color:#fdba74; }
.alert-ok     { background:#0d2e1a; border-left:4px solid #4ade80; padding:12px 16px; border-radius:8px; color:#86efac; }

/* Plotly dark fix */
.js-plotly-plot .plotly .modebar { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────
COLORS = {
    "bg":        "#0a0e1a",
    "card":      "#12172b",
    "border":    "#2a3560",
    "accent1":   "#6366f1",
    "accent2":   "#22d3ee",
    "green":     "#4ade80",
    "orange":    "#fb923c",
    "red":       "#f87171",
    "text":      "#e8edff",
    "muted":     "#6b7db3",
    "rising":    "#4ade80",
    "stable":    "#facc15",
    "declining": "#f87171",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Syne, sans-serif", color=COLORS["text"]),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="#1e2a4a", linecolor="#2a3560"),
    yaxis=dict(gridcolor="#1e2a4a", linecolor="#2a3560"),
)

# ─────────────────────────────────────────────
# DATA LOADERS (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_all_data():
    base = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

    def rp(f): return os.path.join(base, f)

    d = {}
    d["survival"]     = pd.read_csv(rp("career_survival_scores.csv"))
    d["jobs"]         = pd.read_csv(rp("job_roles_clean.csv"))
    d["skills"]       = pd.read_csv(rp("skills_trend_clean.csv"))
    d["skill_intel"]  = pd.read_csv(rp("skill_intelligence.csv"))
    d["skill_proj"]   = pd.read_csv(rp("skill_projections.csv"))
    d["salary_fc"]    = pd.read_csv(rp("salary_forecast.csv"))
    d["sal_traj"]     = pd.read_csv(rp("salary_trajectory_table.csv"))
    d["risk_scores"]  = pd.read_csv(rp("role_automation_risk_scores.csv"))
    d["risk_fi"]      = pd.read_csv(rp("automation_risk_feature_importance.csv"))
    d["top_rising"]   = pd.read_csv(rp("top_rising_skills.csv"))
    d["top_decline"]  = pd.read_csv(rp("top_declining_skills.csv"))
    d["survival_traj"]= pd.read_csv(rp("survival_trajectory.csv"))
    d["exp_survival"] = pd.read_csv(rp("experience_survival_scores.csv"))
    d["all_recs"]     = pd.read_csv(rp("all_role_recommendations.csv"))
    d["role_map"]     = json.load(open(rp("role_encoding_map.json")))
    d["mapping"]      = pd.read_csv(rp("role_skill_mapping_clean.csv"))
    return d


@st.cache_resource
def load_models():
    import joblib
    base = os.path.join(os.path.dirname(__file__), "..", "models")
    risk_model   = joblib.load(os.path.join(base, "automation_risk_model.pkl"))
    salary_model = joblib.load(os.path.join(base, "salary_model.pkl"))
    risk_meta    = json.load(open(os.path.join(base, "automation_risk_model_meta.json")))
    sal_meta     = json.load(open(os.path.join(base, "salary_model_meta.json")))
    return risk_model, salary_model, risk_meta, sal_meta


# ─────────────────────────────────────────────
# SIDEBAR NAV
# ─────────────────────────────────────────────
def sidebar(data):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0 1.5rem;'>
            <div style='font-size:2.2rem;'>🧠</div>
            <div style='font-size:1.1rem; font-weight:800; color:#e8edff;'>Career<br>Intelligence</div>
            <div style='font-size:0.7rem; color:#4b5a80; font-family:DM Mono; margin-top:4px;'>OBSOLESCENCE SYSTEM v1.0</div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            ["🏠  Home", "🔍  Role Analysis", "📈  Skill Trends",
             "🤖  Automation Risk", "🎯  Career Recommendations"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.72rem; color:#3a4a70; font-family:DM Mono;'>"
            "Data: 2019–2025 mock dataset<br>Models: GradientBoosting<br>Skills: 28 | Roles: 24"
            "</div>", unsafe_allow_html=True
        )
    return page


# ─────────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────────
def page_home(data):
    st.markdown("""
    <div style='padding: 2rem 0 1rem;'>
        <div style='font-size:0.8rem; font-family:DM Mono; color:#6366f1; letter-spacing:0.2em;'>INTELLIGENCE DASHBOARD</div>
        <h1 style='font-size:3rem; font-weight:800; color:#e8edff; margin:0.3rem 0; line-height:1.1;'>
            Career Obsolescence<br><span style='color:#6366f1;'>Intelligence System</span>
        </h1>
        <p style='color:#6b7db3; font-size:1rem; max-width:600px;'>
            AI-powered predictions for dying skills, rising careers, automation risk,
            salary trajectories, and personalised future-proof recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    survival = data["survival"]
    jobs     = data["jobs"]
    skills   = data["skill_intel"]

    # ── KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        (col1, "Roles Tracked",   f"{survival['job_role'].nunique()}",       "24 professions"),
        (col2, "Skills Analysed", f"{skills['skill'].nunique()}",             "28 skill signals"),
        (col3, "At Risk Roles",   f"{(survival['career_survival_score']<40).sum()}", "score < 40"),
        (col4, "Rising Skills",   f"{(skills['skill_tier'].str.contains('Rising|Skyrocket')).sum()}", "trend ↑"),
        (col5, "Data Points",     f"{len(jobs):,}",                           "job records"),
    ]
    for col, label, val, sub in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Survival Score Leaderboard + Donut
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-title">Career Survival Leaderboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">All 24 roles ranked by survival score</div>', unsafe_allow_html=True)

        df_sorted = survival.sort_values("career_survival_score", ascending=True)

        color_map = {
            "A+ 🏆 Elite":     "#4ade80",
            "A  🟢 Strong":    "#86efac",
            "B  🟡 Stable":    "#facc15",
            "C  🟠 Vulnerable":"#fb923c",
            "D  🔴 At Risk":   "#f87171",
            "F  💀 Endangered":"#ef4444",
        }
        bar_colors = df_sorted["survival_grade"].map(
            lambda g: next((v for k, v in color_map.items() if k[:2] == g[:2]), "#6366f1")
        )

        fig = go.Figure(go.Bar(
            y=df_sorted["job_role"],
            x=df_sorted["career_survival_score"],
            orientation="h",
            marker_color=bar_colors,
            text=df_sorted["career_survival_score"].round(1),
            textposition="outside",
            textfont=dict(size=11, color=COLORS["text"]),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=640,
                          xaxis_range=[0, 105],
                          xaxis_title="Career Survival Score (0–100)")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Grade Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">How careers are distributed</div>', unsafe_allow_html=True)

        grade_counts = survival["survival_grade"].apply(
            lambda g: g[:2].strip()
        ).value_counts().reset_index()
        grade_counts.columns = ["grade", "count"]

        grade_colors = ["#4ade80", "#86efac", "#facc15", "#fb923c", "#f87171", "#ef4444"]
        fig2 = go.Figure(go.Pie(
            labels=grade_counts["grade"],
            values=grade_counts["count"],
            hole=0.55,
            marker_colors=grade_colors[:len(grade_counts)],
            textinfo="label+percent",
            textfont_size=13,
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=300,
                           showlegend=False,
                           annotations=[dict(text="Grades", x=0.5, y=0.5,
                                            font_size=16, showarrow=False,
                                            font_color=COLORS["text"])])
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title" style="margin-top:1rem;">🔥 Most Endangered</div>', unsafe_allow_html=True)
        bottom5 = survival.nsmallest(5, "career_survival_score")[
            ["job_role", "career_survival_score", "upskill_urgency"]
        ]
        for _, row in bottom5.iterrows():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
                f"border-bottom:1px solid #1e2a4a;font-size:0.85rem;'>"
                f"<span style='color:#e8edff;'>{row['job_role']}</span>"
                f"<span style='color:#f87171;font-family:DM Mono;'>"
                f"{row['career_survival_score']:.1f} {row['upskill_urgency']}</span></div>",
                unsafe_allow_html=True,
            )

    # ── Skill Trend Snapshot
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📡 Skill Intelligence Snapshot</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Rising vs Declining skills by composite trend score</div>', unsafe_allow_html=True)

    top10 = data["top_rising"].head(10)
    bot10 = data["top_decline"].head(10)
    snap = pd.concat([
        top10.assign(direction="Rising"),
        bot10.assign(direction="Declining"),
    ])

    fig3 = px.bar(
        snap, x="composite_trend_score", y="skill",
        color="direction", orientation="h",
        color_discrete_map={"Rising": COLORS["green"], "Declining": COLORS["red"]},
        text="composite_trend_score",
    )
    fig3.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig3.update_layout(**PLOTLY_LAYOUT, height=460,
                       xaxis_title="Composite Trend Score (0–100)",
                       yaxis_title="", showlegend=True,
                       legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 2 — ROLE ANALYSIS
# ─────────────────────────────────────────────
def page_role_analysis(data):
    st.markdown('<h2 style="color:#e8edff; font-weight:800;">🔍 Role Deep-Dive Analysis</h2>', unsafe_allow_html=True)

    survival  = data["survival"]
    jobs      = data["jobs"]
    sal_fc    = data["salary_fc"]
    surv_traj = data["survival_trajectory"]
    exp_surv  = data["exp_survival"]

    roles = sorted(survival["job_role"].unique())
    selected_role = st.selectbox("Select a Job Role", roles, index=roles.index("Data Scientist") if "Data Scientist" in roles else 0)

    row = survival[survival["job_role"] == selected_role].iloc[0]
    score = row["career_survival_score"]

    # Score colour
    score_color = ("#4ade80" if score >= 65 else "#facc15" if score >= 50
                   else "#fb923c" if score >= 35 else "#f87171")

    # ── Hero metrics
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    hero_data = [
        (c1, "Survival Score",    f"{score:.1f}",          row["survival_grade"],          score_color),
        (c2, "Automation Risk",   f"{row['predicted_auto_risk']:.0%}", row["upskill_urgency"], COLORS["red"]),
        (c3, "Human Dependency",  f"{row['human_dependency_score']:.0f}%", "human-centric work", COLORS["accent2"]),
        (c4, "Demand Index",      f"{row['avg_demand_index']:.0f}",   "job postings index",   COLORS["accent1"]),
        (c5, "Salary Growth",     f"{row['growth_2025_to_2030_pct']:.1f}%", "2025→2030",         COLORS["green"]),
    ]
    for col, lbl, val, sub, clr in hero_data:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{lbl}</div>
                <div class="metric-value" style="color:{clr};">{val}</div>
                <div class="metric-sub" style="color:{clr}80;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Radar chart of component scores
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Component Score Radar</div>', unsafe_allow_html=True)
        categories = ["Demand", "Salary Growth", "AI Resistance", "Skill Readiness"]
        values     = [row["demand_score"], row["salary_score"],
                      row["resistance_score"], row["skill_score"]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=f"{COLORS['accent1']}33",
            line_color=COLORS["accent1"],
            name=selected_role,
        ))
        fig_radar.update_layout(
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ["xaxis", "yaxis"]},
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor="#2a3560", linecolor="#2a3560",
                                tickfont=dict(color=COLORS["muted"])),
                angularaxis=dict(gridcolor="#2a3560", linecolor="#2a3560",
                                 tickfont=dict(color=COLORS["text"])),
            ),
            height=360, showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Survival by Experience Level</div>', unsafe_allow_html=True)
        exp_role = exp_surv[exp_surv["job_role"] == selected_role].sort_values("experience_ord")

        if not exp_role.empty:
            fig_exp = go.Figure(go.Bar(
                x=exp_role["experience_level"],
                y=exp_role["quick_survival"],
                marker_color=[COLORS["accent1"], COLORS["accent2"],
                              COLORS["green"], COLORS["orange"], COLORS["red"]][:len(exp_role)],
                text=exp_role["quick_survival"].round(1),
                textposition="outside",
                textfont=dict(color=COLORS["text"]),
            ))
            fig_exp.update_layout(**PLOTLY_LAYOUT, height=360,
                                  yaxis_title="Survival Score",
                                  xaxis_title="Experience Level",
                                  yaxis_range=[0, 105])
            st.plotly_chart(fig_exp, use_container_width=True)

    # ── Salary Forecast
    st.markdown('<div class="section-title">💰 Salary Forecast 2019–2030</div>', unsafe_allow_html=True)
    sal_role = data["salary_fc"][data["salary_fc"]["job_role"] == selected_role]

    hist = sal_role[sal_role["type"] == "historical"]
    proj = sal_role[sal_role["type"] == "projected"]

    fig_sal = go.Figure()
    fig_sal.add_trace(go.Scatter(
        x=hist["year"], y=hist["salary_usd"],
        mode="lines+markers", name="Historical",
        line=dict(color=COLORS["accent2"], width=3),
        marker=dict(size=8),
    ))
    fig_sal.add_trace(go.Scatter(
        x=pd.concat([hist.tail(1), proj])["year"],
        y=pd.concat([hist.tail(1), proj])["salary_usd"],
        mode="lines+markers", name="Projected",
        line=dict(color=COLORS["green"], width=3, dash="dash"),
        marker=dict(size=8, symbol="diamond"),
    ))
    fig_sal.update_layout(**PLOTLY_LAYOUT, height=320,
                          yaxis_title="Avg Salary (USD)",
                          xaxis_title="Year",
                          legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig_sal, use_container_width=True)

    # ── Survival Trajectory
    st.markdown('<div class="section-title">🛡️ Survival Score Trajectory</div>', unsafe_allow_html=True)
    straj = surv_traj[surv_traj["job_role"] == selected_role]

    fig_straj = go.Figure(go.Scatter(
        x=straj["year"], y=straj["survival_score"],
        mode="lines+markers+text",
        text=straj["survival_score"].round(1),
        textposition="top center",
        line=dict(color=score_color, width=3),
        fill="tozeroy",
        fillcolor=f"{score_color}22",
        marker=dict(size=9),
    ))
    fig_straj.add_hline(y=50, line_dash="dot", line_color="#6b7db3",
                        annotation_text="Stable threshold")
    fig_straj.update_layout(**PLOTLY_LAYOUT, height=280,
                             yaxis_title="Survival Score", yaxis_range=[0, 105])
    st.plotly_chart(fig_straj, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 3 — SKILL TRENDS
# ─────────────────────────────────────────────
def page_skill_trends(data):
    st.markdown('<h2 style="color:#e8edff; font-weight:800;">📈 Skill Trend Intelligence</h2>', unsafe_allow_html=True)

    skill_intel = data["skill_intel"]
    skill_proj  = data["skill_proj"]
    skills_ts   = data["skills"]

    # ── Tier overview
    tier_order = ["🚀 Skyrocketing", "📈 Rising", "➡️  Stable", "📉 Declining", "💀 Obsolete"]
    tier_colors = {
        "🚀 Skyrocketing": "#4ade80",
        "📈 Rising":        "#86efac",
        "➡️  Stable":       "#facc15",
        "📉 Declining":     "#fb923c",
        "💀 Obsolete":      "#f87171",
    }

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown('<div class="section-title">Skill Tier Summary</div>', unsafe_allow_html=True)
        tier_df = skill_intel["skill_tier"].value_counts().reset_index()
        tier_df.columns = ["tier", "count"]
        tier_df["color"] = tier_df["tier"].map(tier_colors)

        fig_tier = go.Figure(go.Bar(
            x=tier_df["tier"], y=tier_df["count"],
            marker_color=tier_df["color"],
            text=tier_df["count"],
            textposition="outside",
            textfont=dict(color=COLORS["text"]),
        ))
        fig_tier.update_layout(**PLOTLY_LAYOUT, height=320,
                               xaxis_tickangle=-20, yaxis_title="Number of Skills")
        st.plotly_chart(fig_tier, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Composite Trend Score — All Skills</div>', unsafe_allow_html=True)
        si_sorted = skill_intel.sort_values("composite_trend_score", ascending=False)
        si_sorted["bar_color"] = si_sorted["skill_tier"].map(tier_colors).fillna(COLORS["accent1"])

        fig_all = go.Figure(go.Bar(
            x=si_sorted["composite_trend_score"],
            y=si_sorted["skill"],
            orientation="h",
            marker_color=si_sorted["bar_color"],
            text=si_sorted["composite_trend_score"].round(1),
            textposition="outside",
            textfont=dict(size=10, color=COLORS["text"]),
        ))
        fig_all.update_layout(**PLOTLY_LAYOUT, height=720,
                              xaxis_range=[0, 115],
                              xaxis_title="Composite Trend Score")
        st.plotly_chart(fig_all, use_container_width=True)

    # ── Individual skill time series
    st.markdown('<div class="section-title">📊 Skill Trend Over Time</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Historical (solid) + Projected 2026–2028 (dashed)</div>', unsafe_allow_html=True)

    all_skills = sorted(skill_intel["skill"].unique())
    default_skills = ["Python", "Machine Learning", "LLMs/GenAI", "Excel (Basic)", "COBOL"]
    selected_skills = st.multiselect(
        "Select skills to compare",
        all_skills,
        default=[s for s in default_skills if s in all_skills],
    )

    if selected_skills:
        fig_ts = go.Figure()
        palette = [COLORS["accent1"], COLORS["green"], COLORS["accent2"],
                   COLORS["orange"], COLORS["red"], "#a78bfa", "#f472b6", "#38bdf8"]

        for i, skill in enumerate(selected_skills):
            clr = palette[i % len(palette)]
            sub = skill_proj[skill_proj["skill"] == skill]
            hist = sub[sub["type"] == "historical"]
            proj = sub[sub["type"] == "projected"]

            fig_ts.add_trace(go.Scatter(
                x=hist["year"], y=hist["trend_score"],
                mode="lines+markers", name=skill,
                line=dict(color=clr, width=3),
                marker=dict(size=7),
            ))
            if not proj.empty:
                bridge = pd.concat([hist.tail(1), proj])
                fig_ts.add_trace(go.Scatter(
                    x=bridge["year"], y=bridge["trend_score"],
                    mode="lines", name=f"{skill} (proj)",
                    line=dict(color=clr, width=2, dash="dot"),
                    showlegend=False,
                ))

        fig_ts.add_vrect(x0=2025.5, x1=2028.5, fillcolor="#6366f111",
                         line_width=0, annotation_text="Projected",
                         annotation_position="top left",
                         annotation_font_color=COLORS["muted"])
        fig_ts.update_layout(**PLOTLY_LAYOUT, height=420,
                             yaxis_title="Trend Score (0–10)",
                             xaxis_title="Year",
                             legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig_ts, use_container_width=True)

    # ── Rising vs Declining table
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">🚀 Top Rising Skills</div>', unsafe_allow_html=True)
        top_r = data["top_rising"][["skill", "composite_trend_score", "skill_tier"]].head(10)
        top_r["composite_trend_score"] = top_r["composite_trend_score"].round(1)
        st.dataframe(top_r, use_container_width=True, hide_index=True,
                     column_config={"composite_trend_score": st.column_config.ProgressColumn(
                         "Trend Score", min_value=0, max_value=100)})

    with col_b:
        st.markdown('<div class="section-title">💀 Top Declining Skills</div>', unsafe_allow_html=True)
        top_d = data["top_decline"][["skill", "composite_trend_score", "skill_tier"]].head(10)
        top_d["composite_trend_score"] = top_d["composite_trend_score"].round(1)
        st.dataframe(top_d, use_container_width=True, hide_index=True,
                     column_config={"composite_trend_score": st.column_config.ProgressColumn(
                         "Trend Score", min_value=0, max_value=100)})


# ─────────────────────────────────────────────
# PAGE 4 — AUTOMATION RISK
# ─────────────────────────────────────────────
def page_automation_risk(data):
    st.markdown('<h2 style="color:#e8edff; font-weight:800;">🤖 Automation Risk Analysis</h2>', unsafe_allow_html=True)

    risk   = data["risk_scores"]
    risk_fi = data["risk_fi"]

    # ── Heatmap: Role × Experience
    pivot = risk.pivot_table(
        index="job_role", columns="experience_level",
        values="predicted_auto_risk", aggfunc="mean"
    )
    exp_order = ["Entry", "Mid", "Senior", "Lead", "Principal"]
    pivot = pivot.reindex(columns=[c for c in exp_order if c in pivot.columns])
    pivot = pivot.sort_values("Senior", ascending=False)

    st.markdown('<div class="section-title">Automation Risk Heatmap: Role × Experience</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Higher = more likely to be automated (0.0–1.0)</div>', unsafe_allow_html=True)

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#0d2e1a"], [0.3, "#facc15"], [0.6, "#fb923c"], [1, "#ef4444"]],
        text=np.round(pivot.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="Risk", tickfont=dict(color=COLORS["text"]),
                      titlefont=dict(color=COLORS["text"])),
        hovertemplate="Role: %{y}<br>Experience: %{x}<br>Risk: %{z:.2f}<extra></extra>",
    ))
    fig_heat.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items()
                               if k not in ["xaxis", "yaxis"]},
                           height=700,
                           xaxis=dict(side="top", tickfont=dict(color=COLORS["text"])),
                           yaxis=dict(tickfont=dict(color=COLORS["text"])))
    st.plotly_chart(fig_heat, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        fig_fi = go.Figure(go.Bar(
            x=risk_fi["importance_pct"],
            y=risk_fi["feature"],
            orientation="h",
            marker_color=COLORS["accent1"],
            text=risk_fi["importance_pct"].map(lambda x: f"{x:.1f}%"),
            textposition="outside",
            textfont=dict(color=COLORS["text"]),
        ))
        fig_fi.update_layout(**PLOTLY_LAYOUT, height=380,
                             xaxis_title="Importance (%)", yaxis_title="")
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Risk Distribution by Tier</div>', unsafe_allow_html=True)
        tier_bins = pd.cut(
            risk["predicted_auto_risk"],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["🟢 Very Safe", "🟡 Low", "🟠 Moderate", "🔴 High", "💀 Critical"],
        ).value_counts().reset_index()
        tier_bins.columns = ["tier", "count"]

        fig_dist = go.Figure(go.Bar(
            x=tier_bins["tier"],
            y=tier_bins["count"],
            marker_color=[COLORS["green"], "#86efac", COLORS["orange"],
                          COLORS["red"], "#7f1d1d"],
            text=tier_bins["count"],
            textposition="outside",
            textfont=dict(color=COLORS["text"]),
        ))
        fig_dist.update_layout(**PLOTLY_LAYOUT, height=380,
                               yaxis_title="Number of Records", xaxis_title="Risk Tier")
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Live Risk Calculator
    st.markdown('<div class="section-title">⚡ Live Automation Risk Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Adjust parameters to calculate your role\'s risk in real-time</div>', unsafe_allow_html=True)

    with st.container():
        calc_col1, calc_col2, calc_col3 = st.columns(3)
        with calc_col1:
            c_demand = st.slider("Market Demand Index", 20, 200, 120)
            c_trend  = st.slider("Avg Skill Trend Score", 1.0, 10.0, 7.0, 0.1)
        with calc_col2:
            c_exp    = st.slider("Experience Level (1=Entry → 5=Principal)", 1, 5, 3)
            c_skills = st.slider("Number of Skills", 1, 10, 5)
        with calc_col3:
            c_cagr   = st.slider("Salary CAGR 5yr", -0.05, 0.20, 0.06, 0.01, format="%.2f")
            c_fp     = st.selectbox("Future-Proof Skills?", ["No", "Yes"])

        # Simple formula-based live calculation (no model pickle needed for instant feedback)
        fp_val = 1 if c_fp == "Yes" else 0
        # Mirror the model's key drivers
        live_risk = max(0.05, min(0.98,
            0.85
            - (c_trend / 10) * 0.40
            - fp_val * 0.25
            - (c_exp / 5) * 0.05
            - (c_cagr * 2)
            - (c_skills / 10) * 0.03
            + (1 - c_demand / 200) * 0.10
            + np.random.normal(0, 0.01)  # tiny noise for realism
        ))

        risk_clr = (COLORS["green"] if live_risk < 0.3 else
                    COLORS["orange"] if live_risk < 0.6 else COLORS["red"])

        tier_lbl = ("🟢 Very Safe"    if live_risk < 0.20 else
                    "🟡 Low Risk"     if live_risk < 0.40 else
                    "🟠 Moderate"     if live_risk < 0.60 else
                    "🔴 High Risk"    if live_risk < 0.80 else "💀 Critical")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Automation Risk</div>
                <div class="metric-value" style="color:{risk_clr};">{live_risk:.0%}</div>
                <div class="metric-sub" style="color:{risk_clr}80;">{tier_lbl}</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Human Dependency</div>
                <div class="metric-value" style="color:{COLORS['accent2']};">{(1-live_risk):.0%}</div>
                <div class="metric-sub">work AI can't replicate</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            verdict = ("✅ Future-proof with current skills" if live_risk < 0.35 else
                       "⚠️ Upskill recommended soon" if live_risk < 0.60 else
                       "🚨 Immediate upskilling critical")
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Verdict</div>
                <div class="metric-value" style="font-size:1rem; color:{risk_clr};">{verdict}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 5 — CAREER RECOMMENDATIONS
# ─────────────────────────────────────────────
def page_recommendations(data):
    st.markdown('<h2 style="color:#e8edff; font-weight:800;">🎯 Personalised Career Recommendations</h2>', unsafe_allow_html=True)

    df_mapping, df_skill_intel, df_survival = load_rec_data()
    survival = data["survival"]
    all_recs = data["all_recs"]

    roles = sorted(df_mapping["job_role"].unique())
    all_skills_list = sorted(df_mapping["skill"].unique())

    col_inp1, col_inp2 = st.columns([1, 2])
    with col_inp1:
        sel_role = st.selectbox("Your Current Role", roles,
                                index=roles.index("Data Analyst") if "Data Analyst" in roles else 0)
    with col_inp2:
        owned = st.multiselect(
            "Skills You Already Have",
            all_skills_list,
            default=["SQL", "Excel (Basic)"] if all(s in all_skills_list
                                                     for s in ["SQL", "Excel (Basic)"]) else [],
            help="Select skills from your current toolkit"
        )

    result = get_recommendations(
        job_role=sel_role,
        current_skills=owned,
        df_mapping=df_mapping,
        df_skill_intel=df_skill_intel,
        df_survival=df_survival,
        top_n=6,
    )

    if "error" in result:
        st.error(result["error"])
        return

    # ── Survival score delta
    st.markdown("<br>", unsafe_allow_html=True)
    curr_s = result["current_survival_score"]
    proj_s = result["projected_survival_score"]
    boost  = result["survival_boost"]

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        clr = "#4ade80" if curr_s >= 65 else "#facc15" if curr_s >= 50 else "#f87171"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Current Survival Score</div>
            <div class="metric-value" style="color:{clr};">{curr_s:.1f}</div>
        </div>""", unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">After Upskilling</div>
            <div class="metric-value" style="color:#4ade80;">{proj_s:.1f}</div>
        </div>""", unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Score Boost</div>
            <div class="metric-value" style="color:#22d3ee;">+{boost:.1f}</div>
        </div>""", unsafe_allow_html=True)
    with sc4:
        total_sal_boost = sum(r["salary_boost_usd"] for r in result["recommendations"])
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Salary Potential</div>
            <div class="metric-value" style="color:#a78bfa;">+${total_sal_boost:,}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Recommendations cards
    st.markdown('<div class="section-title">🎓 Your Learning Roadmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Ranked by impact on your career survival score</div>', unsafe_allow_html=True)

    recs = result["recommendations"]
    cols = st.columns(min(3, len(recs)))
    tier_colors_map = {
        "🚀 Skyrocketing": "#4ade80",
        "📈 Rising":        "#86efac",
        "➡️  Stable":       "#facc15",
        "📉 Declining":     "#fb923c",
        "💀 Obsolete":      "#f87171",
    }

    for i, rec in enumerate(recs):
        col = cols[i % 3]
        tier_clr = next((v for k, v in tier_colors_map.items()
                         if k[:2] == rec["skill_tier"][:2]), COLORS["accent1"])
        with col:
            st.markdown(f"""
            <div style="background:#12172b; border:1px solid {tier_clr}44;
                        border-top:3px solid {tier_clr}; border-radius:12px;
                        padding:1.2rem; margin-bottom:1rem;">
                <div style="font-size:0.7rem; font-family:DM Mono; color:{tier_clr};
                            letter-spacing:0.1em; margin-bottom:4px;">
                    RANK #{rec['rank']} · {rec['skill_tier']}
                </div>
                <div style="font-size:1.3rem; font-weight:800; color:#e8edff;
                            margin-bottom:8px;">{rec['skill']}</div>
                <div style="font-size:0.8rem; color:#6b7db3; margin-bottom:12px;">
                    {rec['why']}
                </div>
                <div style="display:flex; gap:8px; flex-wrap:wrap;">
                    <span style="background:#1e2a4a; border-radius:6px; padding:3px 10px;
                                 font-size:0.75rem; color:#a0b4e8; font-family:DM Mono;">
                        📈 2026: {rec['trend_score_2026']}
                    </span>
                    <span style="background:#0d2e1a; border-radius:6px; padding:3px 10px;
                                 font-size:0.75rem; color:#4ade80; font-family:DM Mono;">
                        💰 +${rec['salary_boost_usd']:,}
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Owned skills analysis
    if result["owned_skills_analysis"]:
        st.markdown('<div class="section-title">🔍 Your Current Skills Analysis</div>', unsafe_allow_html=True)
        owned_data = result["owned_skills_analysis"]
        o_cols = st.columns(min(4, len(owned_data)))
        for i, s in enumerate(owned_data):
            with o_cols[i % 4]:
                bg = "#0d2e1a" if "Strong" in s["status"] else "#2e2010" if "Complement" in s["status"] else "#2e0d0d"
                bdr = "#4ade80" if "Strong" in s["status"] else "#fb923c" if "Complement" in s["status"] else "#f87171"
                st.markdown(f"""
                <div style="background:{bg}; border-left:3px solid {bdr}; border-radius:8px;
                            padding:10px 14px; margin-bottom:8px;">
                    <div style="font-size:0.85rem; font-weight:700; color:#e8edff;">{s['skill']}</div>
                    <div style="font-size:0.75rem; color:{bdr}; font-family:DM Mono;">{s['status']}</div>
                </div>""", unsafe_allow_html=True)

    # ── Recommendation score bar chart
    st.markdown('<div class="section-title">📊 Recommendation Priority Scores</div>', unsafe_allow_html=True)
    rec_df = pd.DataFrame(recs)
    fig_rec = go.Figure(go.Bar(
        x=rec_df["recommendation_score"],
        y=rec_df["skill"],
        orientation="h",
        marker_color=[tier_colors_map.get(t[:2] + t[2:], COLORS["accent1"])
                      for t in rec_df["skill_tier"]],
        text=rec_df["recommendation_score"].map(lambda x: f"{x:.1f}"),
        textposition="outside",
        textfont=dict(color=COLORS["text"]),
    ))
    fig_rec.update_layout(**PLOTLY_LAYOUT, height=320,
                          xaxis_title="Recommendation Score",
                          xaxis_range=[0, rec_df["recommendation_score"].max() * 1.2])
    st.plotly_chart(fig_rec, use_container_width=True)

    # ── Industry-wide top recommendations table
    st.markdown('<div class="section-title">🌐 Top Recommendations Across All Roles</div>', unsafe_allow_html=True)
    top_all = (
        all_recs[all_recs["rank"] == 1]
        .sort_values("recommendation_score", ascending=False)
        [["job_role", "recommended_skill", "skill_tier",
          "salary_boost_usd", "trend_score_2026", "recommendation_score"]]
        .reset_index(drop=True)
    )
    st.dataframe(top_all, use_container_width=True, hide_index=True,
                 column_config={
                     "recommendation_score": st.column_config.ProgressColumn(
                         "Priority Score", min_value=0, max_value=100),
                     "salary_boost_usd": st.column_config.NumberColumn(
                         "Salary Boost", format="$%d"),
                 })


# ─────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────
def main():
    data = load_all_data()
    page = sidebar(data)

    if   "Home"            in page: page_home(data)
    elif "Role Analysis"   in page: page_role_analysis(data)
    elif "Skill Trends"    in page: page_skill_trends(data)
    elif "Automation Risk" in page: page_automation_risk(data)
    elif "Recommendations" in page: page_recommendations(data)


if __name__ == "__main__":
    main()
