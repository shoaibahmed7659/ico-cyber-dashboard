import pathlib
import warnings

# Targeted suppression: hide library deprecation/version noise from logs,
# but do NOT blanket-ignore every warning category.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICO Data Security Incident Explorer",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

TPL        = "plotly_white"
C_CYBER    = "#dc2626"
C_NONCYBER = "#2563eb"
C_NEUTRAL  = "#0f766e"
C_WARN     = "#f59e0b"
C_PURPLE   = "#7c3aed"

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.kpi-card{background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);border:1px solid #334155;border-radius:12px;padding:18px 20px 14px;text-align:center;height:110px;display:flex;flex-direction:column;justify-content:center;}
.kpi-label{font-size:0.70rem;text-transform:uppercase;letter-spacing:.08em;color:#94a3b8;margin-bottom:6px;}
.kpi-value{font-size:1.65rem;font-weight:800;color:#f1f5f9;line-height:1.1;}
.kpi-sub{font-size:0.68rem;color:#64748b;margin-top:4px;}
.hero{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#0f172a 100%);border:1px solid #1e40af;border-radius:16px;padding:28px 32px;margin-bottom:24px;}
.hero h1{color:#f1f5f9;font-size:1.9rem;font-weight:800;margin:0 0 8px;}
.hero p{color:#94a3b8;font-size:0.95rem;margin:0;line-height:1.6;}
.section-tag{display:inline-block;background:#1e3a5f;color:#93c5fd;border-radius:6px;padding:2px 10px;font-size:0.72rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px;}
button[data-baseweb="tab"]{font-size:0.82rem;}
section[data-testid="stSidebar"]{background:#0f172a;}
section[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"]{background:#1e40af !important;}
</style>
""", unsafe_allow_html=True)

# ── UI HELPERS ─────────────────────────────────────────────────────────────────
def note(title, body):
    with st.expander("ℹ️  " + title, expanded=False):
        st.markdown(body)

def kpi(label, value, sub=""):
    return ('<div class="kpi-card"><div class="kpi-label">'+label+'</div>'
            '<div class="kpi-value">'+str(value)+'</div>'
            '<div class="kpi-sub">'+sub+'</div></div>')

# ── SHARED FEATURE-DERIVATION LOGIC (single source of truth) ────────────────────
# These constants/functions are used BOTH when engineering training features and
# when deriving features for a single prediction, so the two can never diverge.
BANDS         = ["1 to 9", "10 to 99", "100 to 1k", "1k to 10k", "10k to 100k", "Over 100k"]
BAND_SCORE    = {b: i + 1 for i, b in enumerate(BANDS)}            # 1..6
SC_KEYWORDS   = ["health", "racial", "ethnic", "biometric", "genetic",
                 "sexual", "religion", "political", "criminal"]
W72_TOKENS    = ["0 to 24", "24 to 48", "48 to 72", "within 72", "<72"]
HIGH_IMPACT   = ["1k to 10k", "10k to 100k", "Over 100k"]

def is_special_category(text) -> int:
    """1 if the data-type text references UK GDPR Art.9 special category data."""
    t = str(text).lower()
    return int(any(k in t for k in SC_KEYWORDS))

def is_within_72(text) -> int:
    """1 if the report was made inside the UK GDPR Art.33 72-hour window."""
    t = str(text).lower()
    return int(any(tok in t for tok in W72_TOKENS))

def impact_score_from_band(band) -> int:
    """Ordinal 1..6 encoding of the people-affected band (0 if unknown)."""
    return BAND_SCORE.get(str(band), 0)

# ── DATA LOAD + FEATURE ENGINEERING ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pure transform: raw ICO frame -> analysis-ready frame with engineered cols."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "BI Reference": "BI_Reference", "Year": "Year", "Quarter": "Quarter",
        "Data Subject Type": "Data_Subject_Type", "Data Type": "Data_Type",
        "Decision Taken": "Decision_Taken", "Incident Category": "Incident_Category",
        "Incident Type": "Incident_Type", "No. Data Subjects Affected": "No_Data_Subjects_Affected",
        "Sector": "Sector", "Time Taken to Report": "Time_Taken_to_Report",
    })
    # Robustness: if optional columns are absent, create safe defaults so the
    # rest of the app never KeyErrors on a slightly different export.
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"

    df["No_Data_Subjects_Affected"] = pd.Categorical(
        df["No_Data_Subjects_Affected"], categories=BANDS, ordered=True)

    q_map = {"Qtr 1": 2, "Qtr 2": 5, "Qtr 3": 8, "Qtr 4": 11}  # mid-quarter months
    df["Month"]   = df["Quarter"].map(q_map).fillna(2).astype(int)
    df["Date"]    = pd.to_datetime(df["Year"].astype(str) + "-" +
                                   df["Month"].astype(str).str.zfill(2) + "-01")
    df["YearQtr"] = df["Year"].astype(str) + " " + df["Quarter"].astype(str)

    band_str = df["No_Data_Subjects_Affected"].astype(str)   # strip Categorical before mapping
    df["Is_Cyber"]            = (df["Incident_Category"] == "Cyber").astype(int)
    df["Is_High_Impact"]      = band_str.isin(HIGH_IMPACT).astype(int)
    df["Impact_Score"]        = band_str.map(BAND_SCORE).fillna(0).astype(int)
    df["Is_Special_Category"] = df["Data_Type"].apply(is_special_category)
    if "Time_Taken_to_Report" in df.columns:
        df["Within_72hrs"]    = df["Time_Taken_to_Report"].apply(is_within_72)
    else:
        df["Within_72hrs"]    = 0
    df["Severity_Score"]      = df["Is_Cyber"] * 3 + df["Impact_Score"] + df["Is_Special_Category"] * 2
    return df

def compute_sector_tiers(df: pd.DataFrame, k: int = 20) -> pd.Series:
    """
    Volume-aware risk tiers. Raw cyber rate is unstable for tiny sectors
    (1 of 2 reports = 50%), so we shrink each sector's rate toward the global
    mean using a pseudocount k (empirical-Bayes style) before taking tertiles.
    """
    grp = (df.groupby("Sector")["Is_Cyber"]
             .agg(["sum", "count"])
             .rename(columns={"sum": "n_cyber", "count": "n_total"}))
    global_rate = df["Is_Cyber"].mean()
    grp["rate"] = (grp["n_cyber"] + global_rate * k) / (grp["n_total"] + k)
    q33, q66 = grp["rate"].quantile(0.33), grp["rate"].quantile(0.66)

    def tier(r):
        return "High" if r >= q66 else ("Medium" if r >= q33 else "Low")

    tier_map = grp["rate"].apply(tier)
    return df["Sector"].map(tier_map).fillna("Medium")

@st.cache_data(show_spinner=False)
def load_data():
    path = pathlib.Path("ico_raw.csv")
    if not path.exists():
        return None
    df = engineer_features(pd.read_csv(path))
    df["Sector_Risk_Tier"] = compute_sector_tiers(df)   # computed once, inside the cache
    return df

df_full = load_data()
if df_full is None or df_full.empty:
    st.error("Could not find **ico_raw.csv** in the app folder. "
             "Add the dataset to the repository root and reload.")
    st.stop()

# ── MODEL TRAINING ───────────────────────────────────────────────────────────────
def train_models(df: pd.DataFrame):
    """
    Trains 3 classifiers to predict Cyber vs Non-Cyber.

    IMPORTANT — target leakage avoidance:
    'Incident_Type' is deliberately EXCLUDED. The ICO assigns the Cyber/Non-Cyber
    *category* directly from the incident *type* (ransomware -> Cyber, mis-sent
    email -> Non-Cyber), so using it would let the model re-read the label and
    inflate AUC toward ~1.0 without learning anything generalisable. The model is
    therefore trained on genuinely independent signals only.
    """
    data = df[df["Incident_Category"].isin(["Cyber", "Non Cyber"])].copy()
    data["No_Data_Subjects_Affected"] = data["No_Data_Subjects_Affected"].astype(str)
    y = data["Is_Cyber"]

    cat_candidates = ["Sector", "Data_Subject_Type", "Data_Type",
                      "No_Data_Subjects_Affected", "Time_Taken_to_Report"]
    num_candidates = ["Year", "Is_Special_Category", "Impact_Score", "Within_72hrs"]
    # Keep only columns that exist AND vary (drop constant/empty features).
    cats = [c for c in cat_candidates if c in data.columns and data[c].nunique(dropna=True) > 1]
    nums = [c for c in num_candidates if c in data.columns and data[c].nunique(dropna=True) > 1]
    feats = cats + nums
    X = data[feats].copy()

    base_rate = float(y.mean())
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preproc = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)]), cats),
        ("num", SimpleImputer(strategy="median"), nums),
    ])

    # class_weight='balanced' addresses the Non-Cyber >> Cyber imbalance.
    # (GradientBoosting has no class_weight parameter, so it is left as-is.)
    clfs = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ("Random Forest",       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                        class_weight="balanced", random_state=42, n_jobs=-1)),
        ("Gradient Boosting",   GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)),
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, clf in clfs:
        pipe = Pipeline([("preproc", preproc), ("clf", clf)])
        cv_auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(Xtr, ytr)
        yp    = pipe.predict(Xte)
        yprob = pipe.predict_proba(Xte)[:, 1]
        rep   = classification_report(yte, yp, output_dict=True, zero_division=0)
        auc   = roc_auc_score(yte, yprob)
        fpr, tpr, _ = roc_curve(yte, yprob)
        cm    = confusion_matrix(yte, yp)

        # Model-agnostic permutation importance, reported per real feature
        # (works for all three models, not just tree-based ones).
        try:
            perm = permutation_importance(pipe, Xte, yte, n_repeats=5,
                                          random_state=42, scoring="roc_auc", n_jobs=-1)
            pi = (pd.DataFrame({"Feature": feats, "Importance": perm.importances_mean})
                    .sort_values("Importance", ascending=False))
        except Exception:
            pi = None

        results[name] = {
            "pipe": pipe, "report": rep, "auc": auc,
            "cv_auc_mean": float(cv_auc.mean()), "cv_auc_std": float(cv_auc.std()),
            "fpr": fpr, "tpr": tpr, "cm": cm,
            "perm_importance": pi, "feats": feats, "base_rate": base_rate,
        }
    return results

@st.cache_resource(show_spinner=False)
def get_model_results():
    return train_models(df_full)

model_results = get_model_results()
# Pick the best model by cross-validated AUC (more robust than a single split).
best_name = max(model_results, key=lambda k: model_results[k]["cv_auc_mean"])

# ── AUTO-INSIGHTS ────────────────────────────────────────────────────────────────
def insights(df):
    if df.empty:
        return ["No data matches the current filters."]
    out = []
    cyber_mask = df["Is_Cyber"] == 1
    if cyber_mask.any():
        cyber_sectors = df.loc[cyber_mask, "Sector"].value_counts()
        if not cyber_sectors.empty:
            ts, tc = cyber_sectors.idxmax(), int(cyber_sectors.iloc[0])
            out.append("🔴 **" + str(ts) + "** recorded the most cyber breaches (" + f"{tc:,}" +
                       "). The ICO defines cyber breaches as those involving malicious third-party "
                       "actors such as ransomware or phishing.")
    pct  = round(cyber_mask.mean() * 100, 1)
    base = round(df_full["Is_Cyber"].mean() * 100, 1)
    flag = ("**above** the full-dataset average of " + str(base) + "%") if pct > base else "**within** the typical range"
    out.append("📊 Cyber breaches represent **" + str(pct) + "%** of reports — " + flag + ".")
    hi = round(df["Is_High_Impact"].mean() * 100, 1)
    if hi > 10:
        out.append("⚠️ **" + str(hi) + "%** of breaches affected 1,000+ people — above the 10% threshold. "
                   "The ICO requires organisations to assess and document the risk of harm to affected individuals.")
    sc_rate = round(df["Is_Special_Category"].mean() * 100, 1)
    if sc_rate > 0:
        out.append("🏥 **" + str(sc_rate) + "%** involved **special category data** (health, biometric, "
                   "racial/ethnic, etc.). Under UK GDPR Article 9, this carries the strictest protection obligations.")
    avg_sev = round(df["Severity_Score"].mean(), 2)
    out.append("📈 Average **Severity Score**: **" + str(avg_sev) +
               " / 11** (cyber=3pts, impact band=1-6pts, special category=2pts).")
    type_counts = df["Incident_Type"].value_counts()
    if not type_counts.empty:
        tt, ttc = type_counts.idxmax(), int(type_counts.iloc[0])
        out.append("📌 Most frequent breach type: **" + str(tt) + "** (" + f"{ttc:,}" + " reports).")
    tier_by_sector = df.groupby("Sector")["Sector_Risk_Tier"].first()   # computed once
    high_sectors = tier_by_sector[tier_by_sector == "High"]
    if len(high_sectors) > 0:
        top3 = ", ".join(list(high_sectors.index)[:3])
        out.append("🎯 **" + str(len(high_sectors)) + " sectors** are in the High Cyber Risk Tier — including " + top3 + ".")
    return out

# ── SIDEBAR ──────────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    '<div style="text-align:center;padding:12px 0 8px;">'
    '<img src="https://ico.org.uk/media/about-the-ico/images/ico-logo-2019.png" '
    'width="160" style="max-width:100%;border-radius:4px;" onerror="this.style.display=\'none\'" />'
    '<p style="color:#94a3b8;font-size:0.75rem;margin-top:6px;">Information Commissioner\'s Office</p>'
    '</div>', unsafe_allow_html=True)
st.sidebar.markdown("## Filters")
st.sidebar.caption("Filters apply to all tabs.")
years      = sorted(df_full["Year"].dropna().unique())
sectors    = sorted(df_full["Sector"].dropna().unique())
categories = sorted(df_full["Incident_Category"].dropna().unique())
year_sel   = st.sidebar.multiselect("Year", years, default=years)
sector_sel = st.sidebar.multiselect("Sector", sectors, default=sectors)
cat_sel    = st.sidebar.multiselect("Breach category", categories, default=categories)
filtered   = df_full[df_full["Year"].isin(year_sel) &
                     df_full["Sector"].isin(sector_sel) &
                     df_full["Incident_Category"].isin(cat_sel)].copy()
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="background:#1e293b;border-radius:8px;padding:12px;">'
    '<div style="font-size:0.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;">Records in view</div>'
    '<div style="font-size:1.4rem;font-weight:700;color:#f1f5f9;">' + f"{len(filtered):,}" + '</div>'
    '</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Source:** [ICO Breach Trends](https://ico.org.uk/action-weve-taken/complaints-and-concerns-data-sets/data-security-incident-trends/)  \nData: 2019 – Q4 2025")

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero"><h1>🔐 ICO Data Security Incident Explorer</h1>'
    '<p>Interactive analysis of personal data breaches self-reported to the UK '
    '<strong>Information Commissioner\'s Office (ICO)</strong> — 2019 to Q4 2025. '
    'The ICO publishes this data to help organisations understand breach trends and meet their UK GDPR obligations. '
    'Use the sidebar filters to focus on a specific time period, sector, or breach category.</p></div>',
    unsafe_allow_html=True)

tabs = st.tabs(["📊 Overview", "📈 Trends", "🏢 Sector Analysis", "⚠️ Impact & Severity",
                "🧪 Feature Insights", "🔍 Data Quality", "🤖 Predictive Model",
                "🔮 Risk Predictor", "💡 Key Insights", "📋 About the Data"])

# --- TAB 0 — OVERVIEW ----------------------------------------------------------
with tabs[0]:
    if filtered.empty:
        st.warning("No data matches. Adjust sidebar filters.")
    else:
        total     = len(filtered)
        pct_cyber = round(filtered["Is_Cyber"].mean() * 100, 1)
        pct_hi    = round(filtered["Is_High_Impact"].mean() * 100, 1)
        yr_range  = (str(min(year_sel)) + " – " + str(max(year_sel))) if year_sel else "N/A"
        top_s     = filtered["Sector"].value_counts().idxmax()
        n_secs    = filtered["Sector"].nunique()
        avg_sev   = round(filtered["Severity_Score"].mean(), 1)
        sc_pct    = round(filtered["Is_Special_Category"].mean() * 100, 1)
        hi_risk   = (filtered.groupby("Sector")["Sector_Risk_Tier"].first() == "High").sum()

        st.markdown('<div class="section-tag">Headline Figures</div>', unsafe_allow_html=True)
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.markdown(kpi("Total Breach Reports", f"{total:,}", "All self-reported to ICO"), unsafe_allow_html=True)
        r1c2.markdown(kpi("Cyber Breaches", str(pct_cyber) + "%", "Malicious/technical origin"), unsafe_allow_html=True)
        r1c3.markdown(kpi("High-Impact Breaches", str(pct_hi) + "%", "1,000+ people affected"), unsafe_allow_html=True)
        r1c4.markdown(kpi("Period Covered", yr_range, "Financial years"), unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        r2c1.markdown(kpi("Avg. Severity Score", str(avg_sev) + " / 11", "Cyber+impact+special data"), unsafe_allow_html=True)
        r2c2.markdown(kpi("Special Category Data", str(sc_pct) + "%", "Art.9 UK GDPR — heightened risk"), unsafe_allow_html=True)
        r2c3.markdown(kpi("High-Risk Sectors", str(hi_risk), "Elevated cyber breach rate"), unsafe_allow_html=True)
        r2c4.markdown(kpi("Sectors in Selection", str(n_secs), "Most reported: " + str(top_s)[:18]), unsafe_allow_html=True)
        note("What do these figures mean?",
             "- **Total Breach Reports** — all personal data breach notifications after filters.\n"
             "- **Cyber Breaches** — share classified by ICO as cyber (malicious third-party origin).\n"
             "- **High-Impact** — breaches affecting 1,000+ people.\n"
             "- **Avg. Severity Score** — 0–11: 3pts for cyber, 1–6pts for impact band, 2pts for special category data.\n"
             "- **Special Category Data** — health, biometric, racial/ethnic, religious, criminal (UK GDPR Article 9).\n"
             "- **High-Risk Sectors** — top third by shrinkage-adjusted cyber breach rate across all sectors.")
        st.markdown("---")
        st.markdown('<div class="section-tag">What the data tells us</div>', unsafe_allow_html=True)
        for ins in insights(filtered):
            st.info(ins)
        st.markdown("---")
        st.markdown('<div class="section-tag">Breach Trends at a Glance</div>', unsafe_allow_html=True)
        ch1, ch2 = st.columns([3, 2])
        with ch1:
            st.markdown("#### Cyber vs Non-Cyber breach reports over time")
            tdf = (filtered.groupby(["Date", "Incident_Category"]).size()
                   .reset_index(name="Reports").sort_values("Date"))
            fig_t = px.line(tdf, x="Date", y="Reports", color="Incident_Category", markers=True,
                            color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER}, template=TPL, height=320)
            fig_t.update_layout(legend_title="Breach Category", margin=dict(t=10, b=10))
            st.plotly_chart(fig_t, width="stretch", key="chart_1")
            note("How to read this chart", "Red = cyber (malicious). Blue = non-cyber (human error, physical loss). A rising red line indicates growing cyber threat activity or improved reporting.")
        with ch2:
            st.markdown("#### Breach category breakdown")
            cat_bar = filtered["Incident_Category"].value_counts().reset_index()
            cat_bar.columns = ["Category", "Count"]
            cat_bar["Share (%)"] = round(cat_bar["Count"] / cat_bar["Count"].sum() * 100, 1)
            fig_cb = px.bar(cat_bar, x="Category", y="Count", color="Category", text="Share (%)",
                            color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER}, template=TPL, height=320)
            fig_cb.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_cb.update_layout(showlegend=False, margin=dict(t=10, b=10), yaxis_title="Breach Reports")
            st.plotly_chart(fig_cb, width="stretch", key="chart_2")
            note("How to read this chart", "Bar height = absolute volume. Percentage label = proportional split. Allows direct comparison of both volume and proportion.")
        st.markdown("---")
        st.markdown('<div class="section-tag">Year-on-Year Cyber Trend</div>', unsafe_allow_html=True)
        st.markdown("#### Annual cyber breach rate")
        yoy = filtered.groupby("Year").agg(Total=("Is_Cyber", "count"), Cyber=("Is_Cyber", "sum")).reset_index()
        yoy["Cyber Rate (%)"] = round(yoy["Cyber"] / yoy["Total"] * 100, 1)
        fig_yoy = make_subplots(specs=[[{"secondary_y": True}]])
        fig_yoy.add_trace(go.Bar(x=yoy["Year"], y=yoy["Cyber"], name="Cyber Breach Count",
                                 marker_color=C_CYBER, opacity=0.75), secondary_y=False)
        fig_yoy.add_trace(go.Scatter(x=yoy["Year"], y=yoy["Cyber Rate (%)"], name="Cyber Rate (%)",
                                     mode="lines+markers", line=dict(color=C_WARN, width=2.5),
                                     marker=dict(size=8)), secondary_y=True)
        fig_yoy.update_layout(template=TPL, height=360, legend=dict(orientation="h", y=-0.15),
                              yaxis_title="Cyber Breaches", yaxis2_title="Cyber Breach Rate (%)", hovermode="x unified")
        st.plotly_chart(fig_yoy, width="stretch", key="chart_3")
        note("How to read this dual-axis chart", "Bars (left axis) = absolute cyber breach count. Line (right axis) = cyber rate as % of all breaches that year. If both rise together, cyber is growing in both volume and share.")

# --- TAB 1 — TRENDS ------------------------------------------------------------
with tabs[1]:
    st.markdown("### Breach Report Trends")
    st.info("This version uses full Q1–Q4 2025 data. All years are directly comparable.")
    if not filtered.empty:
        gran = st.selectbox("Group by", ["Year", "Quarter (Year + Qtr)"], key="gran")
        work = filtered.copy()
        work["TimeBucket"] = (work["Year"].astype(str) + " " + work["Quarter"].astype(str)
                              if gran == "Quarter (Year + Qtr)" else work["Year"].astype(str))
        trend = work.groupby(["TimeBucket", "Incident_Category"]).size().reset_index(name="Reports")
        fig_tr = px.bar(trend, x="TimeBucket", y="Reports", color="Incident_Category", barmode="group",
                        color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER}, template=TPL, height=400)
        fig_tr.update_layout(xaxis_title="Period", legend_title="Breach Category")
        st.plotly_chart(fig_tr, width="stretch", key="chart_4")
        note("How to read this chart", "Each pair of bars = one time period. Red = cyber, Blue = non-cyber. Comparing across periods shows whether cyber reporting is growing.")
        st.markdown("---")
        st.markdown("#### Cyber share as proportion of all breaches")
        share = work.groupby(["TimeBucket", "Incident_Category"]).size().reset_index(name="Count")
        fig_a = px.area(share, x="TimeBucket", y="Count", color="Incident_Category", groupnorm="fraction",
                        color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER}, template=TPL, height=340)
        fig_a.update_layout(yaxis_title="Proportion", yaxis_tickformat=".0%", legend_title="Breach Category")
        st.plotly_chart(fig_a, width="stretch", key="chart_5")
        note("How to read this chart", "Growing red area = cyber breaches rising as a share of all reported breaches.")
        st.markdown("---")
        st.markdown("#### Severity Score trend over time")
        sev_t = filtered.groupby("Year")["Severity_Score"].mean().round(2).reset_index()
        sev_t.columns = ["Year", "Avg Severity Score"]
        fig_sv = px.line(sev_t, x="Year", y="Avg Severity Score", markers=True,
                         color_discrete_sequence=[C_PURPLE], template=TPL, height=320)
        st.plotly_chart(fig_sv, width="stretch", key="chart_6")
        note("How to read this chart", "Rising trend = breaches becoming more severe on average — more cyber-classified, affecting more people, or involving more sensitive data.")
        st.markdown("---")
        n_t = st.slider("Top breach types to show", 3, 8, 5, key="n_types")
        top_types = filtered["Incident_Type"].value_counts().head(n_t).index.tolist()
        type_df = (filtered[filtered["Incident_Type"].isin(top_types)]
                   .groupby(["Date", "Incident_Type"]).size().reset_index(name="Reports").sort_values("Date"))
        if not type_df.empty:
            fig_tt = px.line(type_df, x="Date", y="Reports", color="Incident_Type", markers=True, template=TPL, height=380)
            st.plotly_chart(fig_tt, width="stretch", key="chart_7")
            note("How to read this chart", "Each line = one breach type. Rising lines indicate that type is being reported more frequently over time.")

# --- TAB 2 — SECTOR ANALYSIS ---------------------------------------------------
with tabs[2]:
    st.markdown("### Breach Reports by Sector")
    if not filtered.empty:
        top_n = st.slider("Sectors to display", 5, 20, 10, key="topn_s")
        col_a, col_b = st.columns(2)
        with col_a:
            sc = filtered.groupby(["Sector", "Incident_Category"]).size().reset_index(name="Reports")
            tops = sc.groupby("Sector")["Reports"].sum().nlargest(top_n).index
            fig_s = px.bar(sc[sc["Sector"].isin(tops)], y="Sector", x="Reports", color="Incident_Category",
                           barmode="stack", orientation="h",
                           color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER}, template=TPL, height=520)
            fig_s.update_layout(yaxis=dict(categoryorder="total ascending"), legend_title="Breach Category")
            st.plotly_chart(fig_s, width="stretch", key="chart_8")
            note("How to read this chart", "Bar length = total reports. Red = cyber. Long red = predominantly cyber-driven sector.")
        with col_b:
            st.markdown("#### Cyber breach rate by sector")
            cr = filtered.groupby("Sector")["Is_Cyber"].mean().mul(100).round(1).reset_index()
            cr.columns = ["Sector", "Cyber Breach Rate (%)"]
            cr = cr.sort_values("Cyber Breach Rate (%)", ascending=False).head(20)
            tier_map = filtered.groupby("Sector")["Sector_Risk_Tier"].first().to_dict()
            cr["Risk Tier"] = cr["Sector"].map(tier_map).fillna("Medium")
            tier_colours = {"High": C_CYBER, "Medium": C_WARN, "Low": C_NEUTRAL}
            fig_cr = px.bar(cr, x="Cyber Breach Rate (%)", y="Sector", orientation="h", color="Risk Tier",
                            color_discrete_map=tier_colours, template=TPL, height=520)
            fig_cr.update_layout(yaxis=dict(categoryorder="total ascending"),
                                 xaxis_title="% of sector reports classified as cyber")
            st.plotly_chart(fig_cr, width="stretch", key="chart_9")
            note("How to read this chart", "% of each sector's reports classified as cyber. Tier colour comes from the shrinkage-adjusted rate: Red = High (top third), Amber = Medium, Green = Low. Shrinkage prevents tiny sectors with one cyber report from being mislabelled High.")
        st.markdown("---")
        s_pick = st.selectbox("Drill into a sector", ["All sectors"] + sorted(filtered["Sector"].dropna().unique().tolist()), key="sector_dd")
        sec_df = filtered if s_pick == "All sectors" else filtered[filtered["Sector"] == s_pick]
        d1, d2 = st.columns(2)
        with d1:
            it = sec_df["Incident_Type"].value_counts().head(10).reset_index(); it.columns = ["Breach Type", "Reports"]
            fig_it = px.bar(it, x="Reports", y="Breach Type", orientation="h", template=TPL, height=360,
                            title="Most common breach types", color_discrete_sequence=[C_CYBER])
            fig_it.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig_it, width="stretch", key="chart_10")
        with d2:
            dt = sec_df["Data_Type"].value_counts().head(10).reset_index(); dt.columns = ["Data Category", "Reports"]
            fig_dt = px.bar(dt, x="Reports", y="Data Category", orientation="h", template=TPL, height=360,
                            title="Data categories affected", color_discrete_sequence=[C_NEUTRAL])
            fig_dt.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig_dt, width="stretch", key="chart_11")
            note("How to read this chart", "Shows which personal data types are most frequently involved. Special category data (health, biometric etc.) carries UK GDPR Article 9 obligations.")

# --- TAB 3 — IMPACT & SEVERITY -------------------------------------------------
with tabs[3]:
    st.markdown("### Breach Impact & Severity Analysis")
    if not filtered.empty:
        imp = filtered.groupby(["No_Data_Subjects_Affected", "Incident_Category"]).size().reset_index(name="Reports")
        fig_imp = px.bar(imp, x="No_Data_Subjects_Affected", y="Reports", color="Incident_Category", barmode="group",
                         color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER},
                         category_orders={"No_Data_Subjects_Affected": BANDS}, template=TPL, height=380)
        fig_imp.update_layout(xaxis_title="People affected (band)", legend_title="Breach Category")
        st.plotly_chart(fig_imp, width="stretch", key="chart_12")
        note("How to read this chart", "X-axis = ordinal bands of people affected. Cyber breaches (red) tend to appear more in larger impact bands as they often target databases.")
        st.markdown("---")
        top12 = filtered["Sector"].value_counts().head(12).index.tolist()
        fig_sv_box = px.box(filtered[filtered["Sector"].isin(top12)], x="Sector", y="Severity_Score", color="Incident_Category",
                            color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER}, template=TPL, height=420,
                            labels={"Severity_Score": "Severity Score (0-11)"})
        fig_sv_box.update_layout(xaxis_tickangle=-35, legend_title="Breach Category")
        st.plotly_chart(fig_sv_box, width="stretch", key="chart_13")
        note("How to read this box plot", "Box = interquartile range (25th–75th percentile). Line = median. Dots beyond whiskers = outlier breaches. Wider/higher boxes = more variable, more severe breaches.")
        st.markdown("---")
        all_yrs = sorted(df_full["Year"].dropna().unique().tolist())
        midpt = all_yrs[len(all_yrs) // 2]
        p1, p2 = st.columns(2)
        with p1:
            period_a = st.multiselect("Period A", all_yrs, default=[y for y in all_yrs if y < midpt], key="pa")
        with p2:
            period_b = st.multiselect("Period B", all_yrs, default=[y for y in all_yrs if y >= midpt], key="pb")
        if period_a and period_b:
            cmp_col = st.selectbox("Compare by", ["Incident_Category", "Sector", "Incident_Type", "Data_Type"],
                                   format_func=lambda x: {"Incident_Category": "Breach Category", "Sector": "Sector",
                                                          "Incident_Type": "Breach Type", "Data_Type": "Data Category"}.get(x, x),
                                   key="cmp_metric")
            da = df_full[df_full["Year"].isin(period_a)]; db = df_full[df_full["Year"].isin(period_b)]
            ca = da[cmp_col].value_counts().reset_index(); ca.columns = [cmp_col, "Period A"]
            cb = db[cmp_col].value_counts().reset_index(); cb.columns = [cmp_col, "Period B"]
            mg = ca.merge(cb, on=cmp_col, how="outer").fillna(0).sort_values("Period A", ascending=False).head(15)
            melt = mg.melt(id_vars=cmp_col, var_name="Period", value_name="Reports")
            fig_c = px.bar(melt, x="Reports", y=cmp_col, color="Period", barmode="group", orientation="h", template=TPL, height=420)
            fig_c.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig_c, width="stretch", key="chart_14")
            note("How to read this chart", "Longer Period B bars = increase. Helps identify growing or declining breach types between periods.")

# --- TAB 4 — FEATURE INSIGHTS --------------------------------------------------
with tabs[4]:
    st.markdown("### Feature Engineering Insights")
    st.markdown("Patterns derived from engineered features — metrics constructed to reveal deeper analytical insights.")
    if not filtered.empty:
        st.markdown("#### Correlation between engineered numeric features")
        corr_cols = [c for c in ["Is_Cyber", "Is_High_Impact", "Impact_Score", "Is_Special_Category",
                                 "Within_72hrs", "Severity_Score"] if c in filtered.columns]
        corr = filtered[corr_cols].corr().round(2)
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                             aspect="auto", template=TPL, height=420)
        st.plotly_chart(fig_corr, width="stretch", key="chart_corr")
        note("How to read this correlation matrix", "Values near +1 = features rise together; near −1 = move oppositely; near 0 = little linear relationship. Severity Score correlates with its own components by construction. This is exploratory EDA and shows association, not causation.")
        st.markdown("---")
        top10s = filtered["Sector"].value_counts().head(10).index.tolist()
        hm_data = filtered[filtered["Sector"].isin(top10s)].groupby(["Sector", "Year"])["Severity_Score"].mean().round(2).reset_index()
        hm_pivot = hm_data.pivot(index="Sector", columns="Year", values="Severity_Score")
        fig_hm = px.imshow(hm_pivot, color_continuous_scale="RdYlGn_r", labels=dict(color="Avg Severity"),
                           aspect="auto", template=TPL, height=420, title="Avg Severity Score by Sector & Year")
        st.plotly_chart(fig_hm, width="stretch", key="chart_15")
        note("How to read this heatmap", "Darker red = more severe on average. Tracks which sectors are getting worse over time.")
        st.markdown("---")
        dual = filtered[(filtered["Is_Cyber"] == 1) & (filtered["Is_Special_Category"] == 1)]
        if len(dual) > 0:
            dh = dual["Sector"].value_counts().head(12).reset_index(); dh.columns = ["Sector", "High-Risk Breaches"]
            fig_dh = px.bar(dh, x="High-Risk Breaches", y="Sector", orientation="h", color_discrete_sequence=[C_PURPLE],
                            template=TPL, height=400, title="Cyber breaches involving special category data")
            fig_dh.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig_dh, width="stretch", key="chart_16")
            note("How to read this chart", "Sectors with the highest count of breaches that are both cyber-classified AND involve special category data — the highest regulatory exposure combination.")
        st.markdown("---")
        fig_is = px.histogram(filtered, x="Impact_Score", color="Incident_Category", barmode="overlay",
                              color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER}, opacity=0.7,
                              template=TPL, height=340, nbins=6,
                              labels={"Impact_Score": "Impact Score (1=1-9 people, 6=Over 100k)"})
        fig_is.update_layout(legend_title="Breach Category")
        st.plotly_chart(fig_is, width="stretch", key="chart_17")
        note("How to read this histogram", "Score 1 = smallest breaches. Score 6 = largest (100k+). Cyber breaches tend to concentrate at higher scores.")
        st.markdown("---")
        dec_sev = (filtered.groupby("Decision_Taken")["Severity_Score"].mean().round(2).reset_index()
                   .sort_values("Severity_Score", ascending=False))
        fig_ds = px.bar(dec_sev, x="Severity_Score", y="Decision_Taken", orientation="h", color="Severity_Score",
                        color_continuous_scale="RdYlGn_r", template=TPL, height=360,
                        labels={"Severity_Score": "Avg Severity Score", "Decision_Taken": "Regulatory Decision"})
        fig_ds.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_ds, width="stretch", key="chart_18")
        note("How to read this chart", "If investigation-type decisions have higher severity scores, it confirms the ICO's enforcement activity correlates with breach severity.")
        st.markdown("---")
        top8s = filtered["Sector"].value_counts().head(8).index.tolist()
        tm_df = filtered[filtered["Sector"].isin(top8s)].groupby(["Sector", "Incident_Category", "Decision_Taken"]).size().reset_index(name="Reports")
        fig_tm = px.treemap(tm_df, path=["Sector", "Incident_Category", "Decision_Taken"], values="Reports",
                            color="Reports", color_continuous_scale="Blues", template=TPL, height=480)
        fig_tm.update_layout(margin=dict(t=30, b=10))
        st.plotly_chart(fig_tm, width="stretch", key="chart_19")
        note("How to read this treemap", "Area = volume of reports. Click a sector to drill down into its cyber/non-cyber split and regulatory outcomes.")

# --- TAB 5 — DATA QUALITY ------------------------------------------------------
with tabs[5]:
    st.markdown("### Data Quality & Completeness")
    miss = df_full.isnull().sum().reset_index(); miss.columns = ["Field", "Missing Values"]
    miss["Missing (%)"] = (miss["Missing Values"] / len(df_full) * 100).round(1)
    st.dataframe(miss.sort_values("Missing Values", ascending=False), width="stretch")
    note("Why missing values matter", "Missing fields can distort sector/type-level charts. Records without a Sector label are excluded from sector charts, which may undercount certain industries.")
    if not filtered.empty:
        yr_cnt = df_full["Year"].value_counts().sort_index().reset_index(); yr_cnt.columns = ["Year", "Breach Reports"]
        fig_yr = px.bar(yr_cnt, x="Year", y="Breach Reports", template=TPL, color_discrete_sequence=[C_NEUTRAL],
                        title="Total breach reports received by the ICO — by year")
        st.plotly_chart(fig_yr, width="stretch", key="chart_20")
        note("About the 2025 data", "This version uses full Q1–Q4 2025 data. All years are fully comparable.")
        pivot = filtered.groupby(["Decision_Taken", "Incident_Category"]).size().reset_index(name="Reports")
        fig_pv = px.density_heatmap(pivot, x="Incident_Category", y="Decision_Taken", z="Reports",
                                    color_continuous_scale="Blues", template=TPL, height=420)
        st.plotly_chart(fig_pv, width="stretch", key="chart_21")
        note("How to read this heatmap", "Darker = more reports in that combination. Cyber breaches in investigation cells suggests they attract more regulatory scrutiny.")

# --- TAB 6 — PREDICTIVE MODEL --------------------------------------------------
with tabs[6]:
    st.markdown("### Predicting Breach Category: Cyber vs Non-Cyber")
    st.info("**Why isn't 'Incident Type' used as a feature?** The ICO assigns the Cyber / Non-Cyber "
            "*category* directly from the *incident type* (e.g. ransomware → Cyber, a mis-sent email → "
            "Non-Cyber). Feeding incident type to the model would let it re-read the label and push accuracy "
            "to near 100% without learning anything generalisable — a classic case of **target leakage**. "
            "It is therefore excluded, so the model must predict from genuinely independent signals "
            "(sector, data type, scale, timing). The resulting scores are lower but honest and defensible.")
    m_ch = st.selectbox("Select model", list(model_results.keys()), key="mc")
    res = model_results[m_ch]; rep = res["report"]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy", str(round(rep["accuracy"], 3)))
    mc2.metric("Precision — Cyber", str(round(rep.get("1", {}).get("precision", 0), 3)))
    mc3.metric("Recall — Cyber", str(round(rep.get("1", {}).get("recall", 0), 3)))
    mc4.metric("Test ROC-AUC", str(round(res["auc"], 3)))
    st.caption("5-fold cross-validated ROC-AUC: **{:.3f} ± {:.3f}**  •  Cyber base rate in data: **{:.1%}**  "
               "(accuracy must beat this base rate to be meaningful).".format(
                   res["cv_auc_mean"], res["cv_auc_std"], res["base_rate"]))
    note("What these metrics mean", "**Accuracy** = overall correctness. **Precision** = of predicted cyber, how many truly are. **Recall** = of real cyber breaches, how many the model caught. **ROC-AUC** above 0.8 is strong. The cross-validated AUC (mean ± std over 5 folds) is more trustworthy than a single split, and the best model is chosen on it.")
    st.markdown("---")
    rc1, rc2 = st.columns(2)
    with rc1:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=res["fpr"], y=res["tpr"], mode="lines",
                                     name=m_ch + " (AUC=" + str(round(res["auc"], 3)) + ")",
                                     line=dict(color=C_CYBER, width=2)))
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template=TPL, height=360)
        st.plotly_chart(fig_roc, width="stretch", key="chart_22")
        note("How to read the ROC Curve", "Curve hugging top-left = better model. Dashed diagonal = random guessing baseline.")
    with rc2:
        fig_cm = px.imshow(res["cm"], labels=dict(x="Predicted", y="Actual", color="Reports"),
                           x=["Non-Cyber", "Cyber"], y=["Non-Cyber", "Cyber"], text_auto=True,
                           color_continuous_scale="Blues", template=TPL, height=360)
        st.plotly_chart(fig_cm, width="stretch", key="chart_23")
        note("How to read the Confusion Matrix", "Top-left = correct Non-Cyber. Bottom-right = correct Cyber. Off-diagonal = errors.")
    if res.get("perm_importance") is not None:
        pi = res["perm_importance"].sort_values("Importance")
        fig_fi = px.bar(pi, x="Importance", y="Feature", orientation="h", template=TPL, height=420,
                        color="Importance", color_continuous_scale="Blues")
        fig_fi.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"),
                             xaxis_title="Drop in ROC-AUC when this feature is shuffled")
        st.plotly_chart(fig_fi, width="stretch", key="chart_24")
        note("How to read Permutation Importance", "Each bar shows how much the model's ROC-AUC falls when that feature's values are randomly shuffled. A bigger drop means the feature matters more. This method is model-agnostic (works for all three models) and reports importance per real feature rather than per one-hot column. Engineered features (Impact_Score, Is_Special_Category) appearing high confirms they add predictive value.")
    st.markdown("---")
    comp = []
    for k, v in model_results.items():
        comp.append({
            "Model": k,
            "Accuracy": round(v["report"]["accuracy"], 3),
            "Precision (Cyber)": round(v["report"].get("1", {}).get("precision", 0), 3),
            "Recall (Cyber)": round(v["report"].get("1", {}).get("recall", 0), 3),
            "F1 (Cyber)": round(v["report"].get("1", {}).get("f1-score", 0), 3),
            "Test ROC-AUC": round(v["auc"], 3),
            "CV ROC-AUC (mean)": round(v["cv_auc_mean"], 3),
            "CV ROC-AUC (std)": round(v["cv_auc_std"], 3),
            "Best?": "✅" if k == best_name else "",
        })
    st.dataframe(pd.DataFrame(comp), width="stretch", hide_index=True)
    with st.expander("📇  Model card (assumptions, intended use, limitations)", expanded=False):
        st.markdown(
            "**Task:** binary classification — Cyber vs Non-Cyber breach.\n\n"
            "**Intended use:** exploratory pattern-finding and teaching. **Not** for operational breach "
            "triage, compliance decisions, or any automated action affecting individuals.\n\n"
            "**Features:** sector, data subject type, data category, people-affected band, time-to-report, "
            "year, and engineered flags (special-category, impact score, within-72h). Incident type is "
            "excluded to avoid target leakage.\n\n"
            "**Validation:** stratified 80/20 hold-out + 5-fold cross-validation; best model selected on "
            "mean CV ROC-AUC. Class imbalance handled with `class_weight='balanced'` where supported.\n\n"
            "**Key limitations:** self-reported data only; sector labels inconsistent across years; the "
            "target itself is partly definitional; predictions are probabilities, not facts.")

# --- TAB 7 — RISK PREDICTOR ----------------------------------------------------
with tabs[7]:
    st.markdown("### Cyber Breach Risk Estimator")
    st.markdown("Model: **" + best_name + "** (CV ROC-AUC = " + str(round(model_results[best_name]["cv_auc_mean"], 3)) + ")")
    st.warning("**Important:** For exploratory use only. Not for operational breach reporting. "
               "Visit [ico.org.uk/report-a-breach](https://ico.org.uk/for-organisations/report-a-breach/) for guidance.")
    feats_used = model_results[best_name]["feats"]
    time_opts = (sorted(df_full["Time_Taken_to_Report"].dropna().unique().tolist())
                 if "Time_Taken_to_Report" in df_full.columns else ["Unknown"])
    pA, pB, pC = st.columns(3)
    with pA:
        p_s = st.selectbox("Sector", sorted(df_full["Sector"].dropna().unique().tolist()), key="p_s")
        p_dst = st.selectbox("Who was affected", sorted(df_full["Data_Subject_Type"].dropna().unique().tolist()), key="p_dst")
    with pB:
        p_dt = st.selectbox("Data category", sorted(df_full["Data_Type"].dropna().unique().tolist()), key="p_dt")
        p_band = st.selectbox("People affected", BANDS, key="p_band")
    with pC:
        p_time = st.selectbox("Time to report", time_opts, key="p_time")
        p_year = st.selectbox("Year", sorted(df_full["Year"].dropna().unique().tolist()), key="p_year")
    st.caption("Note: incident type is intentionally **not** an input — see the leakage note on the Predictive Model tab.")
    if st.button("Estimate cyber breach probability", width="stretch"):
        is_sc = is_special_category(p_dt)
        imp_s = impact_score_from_band(p_band)
        w72   = is_within_72(p_time)
        candidate = {"Sector": p_s, "Data_Subject_Type": p_dst, "Data_Type": p_dt,
                     "No_Data_Subjects_Affected": p_band, "Time_Taken_to_Report": p_time,
                     "Year": p_year, "Is_Special_Category": is_sc, "Impact_Score": imp_s, "Within_72hrs": w72}
        X_new = pd.DataFrame([{k: candidate[k] for k in feats_used}])   # only the columns the model was trained on
        best_pipe = model_results[best_name]["pipe"]
        proba = best_pipe.predict_proba(X_new)[0, 1]
        label = "Cyber" if proba >= 0.5 else "Non-Cyber"
        sev_s = 3 * int(proba >= 0.5) + imp_s + is_sc * 2
        pct_s = str(round(proba * 100, 1)) + "%"
        g_col, t_col = st.columns([1, 2])
        with g_col:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=round(proba * 100, 1), number={"suffix": "%"},
                title={"text": "Cyber Probability"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": C_CYBER if proba >= 0.5 else C_NONCYBER},
                       "steps": [{"range": [0, 40], "color": "#dcfce7"},
                                 {"range": [40, 70], "color": "#fef9c3"},
                                 {"range": [70, 100], "color": "#fee2e2"}],
                       "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
            fig_g.update_layout(height=300, margin=dict(t=60, b=20))
            st.plotly_chart(fig_g, width="stretch", key="chart_25")
        with t_col:
            msg = ("Predicted: **" + label + "**  \nCyber probability: **" + pct_s + "**  \n"
                   "Severity Score: **" + str(sev_s) + " / 11**  \nSpecial category data: **" +
                   ("Yes" if is_sc else "No") + "**  \nModel: " + best_name)
            (st.error if label == "Cyber" else st.success)(msg)

# --- TAB 8 — KEY INSIGHTS ------------------------------------------------------
with tabs[8]:
    st.markdown("### Key Findings")
    if filtered.empty:
        st.warning("No data available.")
    else:
        for ins in insights(filtered):
            st.info(ins)
        st.markdown("---")
        dec = filtered.groupby(["Decision_Taken", "Incident_Category"]).size().reset_index(name="Reports")
        fig_dec = px.bar(dec, x="Decision_Taken", y="Reports", color="Incident_Category", barmode="group",
                         color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER}, template=TPL, height=380)
        st.plotly_chart(fig_dec, width="stretch", key="chart_26")
        note("How to read this chart", "If cyber breaches more frequently result in 'Investigation Pursued', this aligns with ICO guidance that cyber breaches carry higher risk of harm to individuals.")
        st.markdown("---")
        st.markdown("#### Download pre-built filtered dataset")
        csv_path = pathlib.Path("ico_breach_data_filtered.csv")  # static file stored in the repo (kept lightweight on purpose)
        if csv_path.exists():
            with open(csv_path, "rb") as f:
                csv_bytes = f.read()
            st.download_button(
                "⬇️  Download filtered CSV",
                csv_bytes,
                file_name=csv_path.name,
                mime="text/csv",
                width="stretch",
                help="Static filtered CSV generated offline and stored in the repo (keeps the free-tier server light).",
            )
        else:
            st.info("Filtered CSV file not found in app folder. Please upload ico_breach_data_filtered.csv.")

# --- TAB 9 — ABOUT THE DATA ----------------------------------------------------
with tabs[9]:
    st.markdown("### About This Dashboard & the Underlying Data")
    st.markdown("""
#### Data Source
Built on the **ICO Data Security Incident Trends** dataset published by the [UK Information Commissioner's Office (ICO)](https://ico.org.uk).

> *"We publish this information to help organisations understand what to look out for and take appropriate action."* — ICO

---

#### What the data contains
Each row = one personal data breach self-reported to the ICO.

> *"A breach of security leading to the accidental or unlawful destruction, loss, alteration, unauthorised disclosure of, or access to, personal data."* — UK GDPR Article 4(12)

| Field | ICO Definition |
|---|---|
| **Breach Category** | Cyber (malicious, third-party) or Non-Cyber (human error, physical loss) |
| **Breach Type** | How the breach occurred |
| **Data Subject Type** | Who was affected: customers, employees, patients, etc. |
| **Data Category** | Type of personal data, including special category data |
| **Regulatory Decision** | ICO response: No Further Action, Investigation Pursued, etc. |
| **People Affected** | Estimated band of individuals impacted |
| **Sector** | Organisation type |
| **Time to Report** | Hours from discovery to notification |

---

#### Engineered Features

| Feature | Construction | Range |
|---|---|---|
| **Severity Score** | 3pts (cyber) + 1–6pts (impact) + 2pts (special category) | 0–11 |
| **Is Special Category** | 1 if Data Category contains Art.9 keywords | 0/1 |
| **Impact Score** | Ordinal encoding of people-affected band | 1–6 |
| **Sector Risk Tier** | High/Medium/Low based on *shrinkage-adjusted* cyber rate tertiles | 3 levels |
| **Within 72hrs** | 1 if reported within Art.33 timeframe | 0/1 |

---

#### Modelling note — why incident type is excluded
The Cyber / Non-Cyber **category** is assigned by the ICO directly from the **incident type**, so using incident
type as a model feature would be **target leakage** (the model would simply re-read the label). It is excluded so the
classifier must learn from independent signals. Validation uses a stratified hold-out plus 5-fold cross-validation,
and the best model is chosen on mean cross-validated ROC-AUC.

---

#### Limitations
- Covers **self-reported breaches only** — unreported incidents are not captured.
- **Sector labels** are not always consistent in historic records.
- People-affected figures are **estimates at time of notification**.
- Engineered features are **analytical constructs** — not ICO classifications.
- The Cyber/Non-Cyber target is **partly definitional**; the model is interpretive, not a compliance tool.

---

#### Further reading
- [ICO Data Security Incident Trends](https://ico.org.uk/action-weve-taken/complaints-and-concerns-data-sets/data-security-incident-trends/)
- [ICO Glossary of Terms](https://ico.org.uk/action-weve-taken/complaints-and-concerns-data-sets/data-security-incident-trends/glossary-of-terms/)
- [UK GDPR Article 33 — Breach Notification](https://ico.org.uk/for-organisations/report-a-breach/personal-data-breach/)
- [ICO: Responding to a Cybersecurity Incident](https://ico.org.uk/media2/migrated/2614816/responding-to-a-cybersecurity-incident.pdf)
""")

st.markdown("---")
st.caption("Data: UK Information Commissioner's Office — ico.org.uk | Built with Streamlit and Plotly")
