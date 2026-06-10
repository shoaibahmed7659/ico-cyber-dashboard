import pathlib
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
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

# ════════════════════════════════════════════════════════════════════════════════
#  PAGE + DESIGN SYSTEM
# ════════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ICO Data Breach Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Palette: institutional, accessible, red reserved for "cyber / threat" --------
INK        = "#0f172a"
ACCENT     = "#4f46e5"   # indigo — primary brand accent
C_CYBER    = "#d1495b"   # muted crimson — malicious / cyber
C_NONCYBER = "#1d6fb8"   # institutional blue — human error / physical
C_TEAL     = "#2a9d8f"   # low risk / positive
C_AMBER    = "#e09f3e"   # medium risk / caution
C_PURPLE   = "#7c5cbf"
GRID       = "#eef1f6"

# --- Register one Plotly template so every chart is styled consistently (and the
#     per-chart layout code stays light, which keeps rendering efficient). --------
pio.templates["ico"] = go.layout.Template(layout=dict(
    font=dict(family="ui-sans-serif, system-ui, 'Segoe UI', sans-serif", size=13, color=INK),
    colorway=[ACCENT, C_CYBER, C_NONCYBER, C_TEAL, C_AMBER, C_PURPLE],
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, ticks="outside", tickcolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    margin=dict(t=46, b=40, l=10, r=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=""),
    title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
    hoverlabel=dict(font_size=12),
))
TPL = "plotly_white+ico"
CAT_COLOURS  = {"Cyber": C_CYBER, "Non Cyber": C_NONCYBER}
TIER_COLOURS = {"High": C_CYBER, "Medium": C_AMBER, "Low": C_TEAL}

# --- Minimal, self-contained CSS (cards + headings only — no fragile selectors) ---
st.markdown("""
<style>
:root{ --ink:#0f172a; --muted:#64748b; --line:#e6e8ee; --accent:#4f46e5; }
.block-container{ padding-top:2.2rem; max-width:1280px; }
.eyebrow{ font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
          color:var(--accent); margin:0 0 .35rem; }
.h-section{ font-size:1.35rem; font-weight:750; color:var(--ink); margin:.1rem 0 .2rem; letter-spacing:-.01em; }
.sub{ color:var(--muted); font-size:.92rem; margin:0 0 .4rem; }
.hero{ border:1px solid var(--line); border-radius:18px; padding:26px 30px;
       background:linear-gradient(120deg,#eef2ff 0%,#f8fafc 55%,#ffffff 100%);
       box-shadow:0 1px 2px rgba(16,24,40,.04); margin-bottom:18px; }
.hero h1{ font-size:1.7rem; font-weight:800; color:var(--ink); margin:0 0 6px; letter-spacing:-.02em; }
.hero p{ color:#475569; font-size:.95rem; line-height:1.55; margin:0; max-width:70ch; }
.kpi{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:15px 17px 13px;
      border-top:3px solid var(--accent); height:108px; display:flex; flex-direction:column; justify-content:center;
      box-shadow:0 1px 2px rgba(16,24,40,.04); }
.kpi.red{border-top-color:#d1495b;} .kpi.blue{border-top-color:#1d6fb8;}
.kpi.teal{border-top-color:#2a9d8f;} .kpi.amber{border-top-color:#e09f3e;}
.kpi .l{ font-size:.68rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); }
.kpi .v{ font-size:1.6rem; font-weight:800; color:var(--ink); line-height:1.15; margin-top:5px; letter-spacing:-.02em; }
.kpi .s{ font-size:.7rem; color:#94a3b8; margin-top:3px; }
.finding{ background:#fff; border:1px solid var(--line); border-left:3px solid var(--accent);
          border-radius:10px; padding:11px 14px; margin-bottom:9px; font-size:.92rem; color:#1e293b; line-height:1.5; }
.note{ color:var(--muted); font-size:.83rem; line-height:1.5; margin:-.2rem 0 .4rem; }
hr{ border:none; border-top:1px solid var(--line); margin:1.1rem 0; }
.stTabs [data-baseweb="tab"]{ font-weight:600; }
</style>
""", unsafe_allow_html=True)

def eyebrow(tag, title, sub=""):
    html = '<div class="eyebrow">' + tag + '</div><div class="h-section">' + title + '</div>'
    if sub:
        html += '<div class="sub">' + sub + '</div>'
    st.markdown(html, unsafe_allow_html=True)

def cap(text):
    """Concise inline reading-note under a chart (reading + why it matters)."""
    st.markdown('<div class="note">' + text + '</div>', unsafe_allow_html=True)

def kpi(label, value, sub="", tone="accent"):
    cls = "kpi" if tone == "accent" else "kpi " + tone
    return ('<div class="' + cls + '"><div class="l">' + label + '</div><div class="v">' +
            str(value) + '</div><div class="s">' + sub + '</div></div>')

# One chart renderer for the whole app: consistent width, no Plotly toolbar (cleaner + lighter).
PCFG = {"displayModeBar": False, "responsive": True}
def pchart(fig, key):
    fig.update_layout(legend_title_text="")   # px otherwise prints the raw column name ("Incident_Category")
    st.plotly_chart(fig, width="stretch", key=key, config=PCFG)

def delta_html(curr, prev, pts=False, bad_up=True):
    """Compact ▲/▼ indicator: latest period vs the one before. Red = worse when bad_up."""
    if prev is None or prev == 0:
        return ""
    d = (curr - prev) if pts else (curr - prev) / prev * 100
    if abs(d) < 0.05:
        return '<span style="color:#94a3b8">— level vs prior yr</span>'
    up = d > 0
    col = ("#d1495b" if up else "#2a9d8f") if bad_up else "#64748b"
    val = (f"{abs(d):.1f} pts" if pts else f"{abs(d):.0f}%")
    return '<span style="color:' + col + '">' + ("▲" if up else "▼") + " " + val + " vs prior yr</span>"

# ════════════════════════════════════════════════════════════════════════════════
#  SHARED FEATURE LOGIC (single source of truth for training AND prediction)
# ════════════════════════════════════════════════════════════════════════════════
BANDS       = ["1 to 9", "10 to 99", "100 to 1k", "1k to 10k", "10k to 100k", "Over 100k"]
BAND_SCORE  = {b: i + 1 for i, b in enumerate(BANDS)}
SC_KEYWORDS = ["health", "racial", "ethnic", "biometric", "genetic", "sexual",
               "religion", "political", "criminal"]
W72_TOKENS  = ["0 to 24", "24 to 48", "48 to 72", "within 72", "<72"]
HIGH_IMPACT = ["1k to 10k", "10k to 100k", "Over 100k"]

def is_special_category(text) -> int:
    t = str(text).lower()
    return int(any(k in t for k in SC_KEYWORDS))

def is_within_72(text) -> int:
    t = str(text).lower()
    return int(any(tok in t for tok in W72_TOKENS))

def impact_score_from_band(band) -> int:
    return BAND_SCORE.get(str(band), 0)

# ════════════════════════════════════════════════════════════════════════════════
#  DATA
# ════════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "BI Reference": "BI_Reference", "Data Subject Type": "Data_Subject_Type",
        "Data Type": "Data_Type", "Decision Taken": "Decision_Taken",
        "Incident Category": "Incident_Category", "Incident Type": "Incident_Type",
        "No. Data Subjects Affected": "No_Data_Subjects_Affected",
        "Time Taken to Report": "Time_Taken_to_Report",
    })
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"

    df["No_Data_Subjects_Affected"] = pd.Categorical(
        df["No_Data_Subjects_Affected"], categories=BANDS, ordered=True)
    q_map = {"Qtr 1": 2, "Qtr 2": 5, "Qtr 3": 8, "Qtr 4": 11}
    df["Month"]   = df["Quarter"].map(q_map).fillna(2).astype(int)
    df["Date"]    = pd.to_datetime(df["Year"].astype(str) + "-" +
                                   df["Month"].astype(str).str.zfill(2) + "-01")
    df["YearQtr"] = df["Year"].astype(str) + " " + df["Quarter"].astype(str)

    band_str = df["No_Data_Subjects_Affected"].astype(str)
    df["Is_Cyber"]            = (df["Incident_Category"] == "Cyber").astype(int)
    df["Is_High_Impact"]      = band_str.isin(HIGH_IMPACT).astype(int)
    df["Impact_Score"]        = band_str.map(BAND_SCORE).fillna(0).astype(int)
    df["Is_Special_Category"] = df["Data_Type"].apply(is_special_category)
    df["Within_72hrs"]        = (df["Time_Taken_to_Report"].apply(is_within_72)
                                 if "Time_Taken_to_Report" in df.columns else 0)
    df["Severity_Score"]      = df["Is_Cyber"] * 3 + df["Impact_Score"] + df["Is_Special_Category"] * 2
    return df

def compute_sector_tiers(df: pd.DataFrame, k: int = 20) -> pd.Series:
    """Volume-aware tiers: shrink each sector's cyber rate toward the global mean
    (pseudocount k) before taking tertiles, so tiny sectors can't be spuriously High."""
    grp = (df.groupby("Sector")["Is_Cyber"].agg(["sum", "count"])
             .rename(columns={"sum": "n_cyber", "count": "n_total"}))
    global_rate = df["Is_Cyber"].mean()
    grp["rate"] = (grp["n_cyber"] + global_rate * k) / (grp["n_total"] + k)
    q33, q66 = grp["rate"].quantile(0.33), grp["rate"].quantile(0.66)
    tier = lambda r: "High" if r >= q66 else ("Medium" if r >= q33 else "Low")
    return df["Sector"].map(grp["rate"].apply(tier)).fillna("Medium")

@st.cache_data(show_spinner=False)
def load_data():
    path = pathlib.Path("ico_raw.csv")
    if not path.exists():
        return None
    df = engineer_features(pd.read_csv(path))
    df["Sector_Risk_Tier"] = compute_sector_tiers(df)
    return df

df_full = load_data()
if df_full is None or df_full.empty:
    st.error("**ico_raw.csv** was not found in the app folder. "
             "Add the dataset to the repository root and reload the app.")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
#  MODEL  (Incident_Type excluded — see leakage note in Modelling tab)
# ════════════════════════════════════════════════════════════════════════════════
def train_models(df: pd.DataFrame):
    data = df[df["Incident_Category"].isin(["Cyber", "Non Cyber"])].copy()
    data["No_Data_Subjects_Affected"] = data["No_Data_Subjects_Affected"].astype(str)
    y = data["Is_Cyber"]

    cat_candidates = ["Sector", "Data_Subject_Type", "Data_Type",
                      "No_Data_Subjects_Affected", "Time_Taken_to_Report"]
    num_candidates = ["Year", "Is_Special_Category", "Impact_Score", "Within_72hrs"]
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
    clfs = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ("Random Forest",       RandomForestClassifier(n_estimators=100, max_depth=10,
                                                        class_weight="balanced", random_state=42, n_jobs=-1)),
        ("Gradient Boosting",   GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=42)),
    ]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = {}
    for name, clf in clfs:
        pipe = Pipeline([("preproc", preproc), ("clf", clf)])
        cv_auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(Xtr, ytr)
        yp, yprob = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
        rep = classification_report(yte, yp, output_dict=True, zero_division=0)
        fpr, tpr, _ = roc_curve(yte, yprob)
        try:
            perm = permutation_importance(pipe, Xte, yte, n_repeats=2,
                                          random_state=42, scoring="roc_auc", n_jobs=-1)
            pi = (pd.DataFrame({"Feature": feats, "Importance": perm.importances_mean})
                    .sort_values("Importance", ascending=False))
        except Exception:
            pi = None
        results[name] = {
            "pipe": pipe, "report": rep, "auc": roc_auc_score(yte, yprob),
            "cv_auc_mean": float(cv_auc.mean()), "cv_auc_std": float(cv_auc.std()),
            "fpr": fpr, "tpr": tpr, "cm": confusion_matrix(yte, yp),
            "perm_importance": pi, "feats": feats, "base_rate": base_rate,
        }
    return results

@st.cache_resource(show_spinner="Training models (first load only)…")
def get_model_results():
    return train_models(df_full)

def models_and_best():
    """Lazy accessor — models train on first use (Modelling/Predictor tab), not at startup."""
    mr = get_model_results()
    return mr, max(mr, key=lambda k: mr[k]["cv_auc_mean"])

# ════════════════════════════════════════════════════════════════════════════════
#  AUTO FINDINGS
# ════════════════════════════════════════════════════════════════════════════════
def insights(df):
    if df.empty:
        return ["No reports match the current filters — widen the selection in the sidebar."]
    out = []
    cyber_mask = df["Is_Cyber"] == 1
    if cyber_mask.any():
        cs = df.loc[cyber_mask, "Sector"].value_counts()
        if not cs.empty:
            out.append("<b>" + str(cs.idxmax()) + "</b> reported the most cyber breaches (" +
                       f"{int(cs.iloc[0]):,}" + ") — the ICO classes these as involving a malicious "
                       "third party, such as ransomware or phishing.")
    pct, base = round(cyber_mask.mean() * 100, 1), round(df_full["Is_Cyber"].mean() * 100, 1)
    flag = ("<b>above</b> the dataset average of " + str(base) + "%") if pct > base else "in line with the dataset average"
    out.append("Cyber accounts for <b>" + str(pct) + "%</b> of reports in view — " + flag + ".")
    hi = round(df["Is_High_Impact"].mean() * 100, 1)
    if hi > 10:
        out.append("<b>" + str(hi) + "%</b> of breaches affected 1,000+ people, the band where the ICO "
                   "expects a documented assessment of harm to individuals.")
    sc = round(df["Is_Special_Category"].mean() * 100, 1)
    if sc > 0:
        out.append("<b>" + str(sc) + "%</b> involved special-category data (health, biometric, ethnicity), "
                   "which carries the strictest duties under UK GDPR Article 9.")
    out.append("Average severity is <b>" + str(round(df["Severity_Score"].mean(), 2)) +
               " / 11</b> (cyber = 3, people-affected band = 1–6, special-category = 2).")
    tc = df["Incident_Type"].value_counts()
    if not tc.empty:
        out.append("The single most common breach type is <b>" + str(tc.idxmax()) + "</b> (" +
                   f"{int(tc.iloc[0]):,}" + " reports).")
    tbs = df.groupby("Sector")["Sector_Risk_Tier"].first()
    high = tbs[tbs == "High"]
    if len(high) > 0:
        out.append("<b>" + str(len(high)) + " sectors</b> sit in the High cyber-risk tier, including " +
                   ", ".join(list(high.index)[:3]) + ".")
    return out

# ════════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("### 🛡️ ICO Breach Intelligence")
st.sidebar.caption("UK data security incident trends · 2019 – Q4 2025")
st.sidebar.divider()
st.sidebar.markdown("**Filters** — applied across every tab")
years      = sorted(df_full["Year"].dropna().unique())
sectors    = sorted(df_full["Sector"].dropna().unique())
categories = sorted(df_full["Incident_Category"].dropna().unique())
year_sel   = st.sidebar.multiselect("Year", years, default=years)
sector_sel = st.sidebar.multiselect("Sector", sectors, default=sectors)
cat_sel    = st.sidebar.multiselect("Breach category", categories, default=categories)
filtered   = df_full[df_full["Year"].isin(year_sel) &
                     df_full["Sector"].isin(sector_sel) &
                     df_full["Incident_Category"].isin(cat_sel)].copy()
st.sidebar.divider()
st.sidebar.metric("Reports in view", f"{len(filtered):,}", help="Rows matching the filters above")
st.sidebar.caption("Source: [ICO Data Security Incident Trends]"
                   "(https://ico.org.uk/action-weve-taken/complaints-and-concerns-data-sets/"
                   "data-security-incident-trends/)")

# ════════════════════════════════════════════════════════════════════════════════
#  HEADER + TABS
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="hero"><h1>🛡️ ICO Data Breach Intelligence</h1>'
    '<p>An interactive read on personal-data breaches self-reported to the UK '
    '<b>Information Commissioner&rsquo;s Office</b> between 2019 and Q4 2025 — what is breached, '
    'who is affected, which sectors carry the most cyber risk, and how well that risk can be '
    'predicted from breach characteristics alone. Use the sidebar to focus on a period, sector or category.</p></div>',
    unsafe_allow_html=True)

T_OVERVIEW, T_TRENDS, T_SECTORS, T_IMPACT, T_FEATURE, T_MODEL, T_PREDICT, T_DATA = st.tabs(
    ["Overview", "Trends", "Sectors", "Impact & Risk", "Feature Insights",
     "Modelling", "Risk Predictor", "Data & Method"])

# ─────────────────────────────────────────────────────────────────────────────────
#  1 · OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────────
with T_OVERVIEW:
    if filtered.empty:
        st.warning("No reports match the current filters. Widen the selection in the sidebar.")
    else:
        total   = len(filtered)
        p_cyber = round(filtered["Is_Cyber"].mean() * 100, 1)
        p_hi    = round(filtered["Is_High_Impact"].mean() * 100, 1)
        p_sc    = round(filtered["Is_Special_Category"].mean() * 100, 1)
        avg_sev = round(filtered["Severity_Score"].mean(), 1)
        n_sec   = filtered["Sector"].nunique()
        top_sec = filtered["Sector"].value_counts().idxmax()
        n_high  = int((filtered.groupby("Sector")["Sector_Risk_Tier"].first() == "High").sum())
        yr_rng  = (str(min(year_sel)) + "–" + str(max(year_sel))) if year_sel else "—"

        # Latest-year vs prior-year deltas (informational on the headline cards)
        yrs = sorted(filtered["Year"].dropna().unique())
        d_rep = d_cyb = ""
        if len(yrs) >= 2:
            last, prev = yrs[-1], yrs[-2]
            d_rep = delta_html(int((filtered["Year"] == last).sum()),
                               int((filtered["Year"] == prev).sum()), pts=False, bad_up=False)
            d_cyb = delta_html(filtered.loc[filtered["Year"] == last, "Is_Cyber"].mean() * 100,
                               filtered.loc[filtered["Year"] == prev, "Is_Cyber"].mean() * 100, pts=True, bad_up=True)

        eyebrow("At a glance", "Headline figures")
        r1 = st.columns(4)
        r1[0].markdown(kpi("Breach reports", f"{total:,}", d_rep or "Self-reported to the ICO"), unsafe_allow_html=True)
        r1[1].markdown(kpi("Cyber share", str(p_cyber) + "%", d_cyb or "Malicious / technical origin", "red"), unsafe_allow_html=True)
        r1[2].markdown(kpi("High-impact", str(p_hi) + "%", "1,000+ people affected", "amber"), unsafe_allow_html=True)
        r1[3].markdown(kpi("Period", yr_rng, "Reporting years", "blue"), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        r2 = st.columns(4)
        r2[0].markdown(kpi("Avg. severity", str(avg_sev) + " / 11", "Cyber + impact + sensitivity"), unsafe_allow_html=True)
        r2[1].markdown(kpi("Special-category", str(p_sc) + "%", "Article 9 data involved", "red"), unsafe_allow_html=True)
        r2[2].markdown(kpi("High-risk sectors", str(n_high), "Top tier by adjusted rate", "amber"), unsafe_allow_html=True)
        r2[3].markdown(kpi("Sectors in view", str(n_sec), "Most active: " + str(top_sec)[:20], "teal"), unsafe_allow_html=True)

        st.divider()
        eyebrow("What the data says", "Key findings", "Generated live from the filtered selection.")
        for line in insights(filtered):
            st.markdown('<div class="finding">' + line + '</div>', unsafe_allow_html=True)

        st.divider()
        eyebrow("Trend", "Cyber vs non-cyber over time")
        a, b = st.columns([3, 2])
        with a:
            tdf = (filtered.groupby(["Date", "Incident_Category"]).size()
                   .reset_index(name="Reports").sort_values("Date"))
            fig = px.line(tdf, x="Date", y="Reports", color="Incident_Category", markers=True,
                          color_discrete_map=CAT_COLOURS, template=TPL, height=320)
            pchart(fig, key="ov_line")
            cap("Red tracks malicious (cyber) breaches; blue tracks human error and physical loss. "
                "A rising red line points to growing attacker activity — or improved detection and reporting.")
        with b:
            cb = filtered["Incident_Category"].value_counts().reset_index()
            cb.columns = ["Category", "Count"]
            cb["Share"] = round(cb["Count"] / cb["Count"].sum() * 100, 1)
            fig = px.bar(cb, x="Category", y="Count", color="Category", text="Share",
                         color_discrete_map=CAT_COLOURS, template=TPL, height=320)
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(showlegend=False)
            pchart(fig, key="ov_bar")
            cap("Most reported breaches are still non-cyber, but cyber&rsquo;s share is what regulators watch most closely.")

        eyebrow("Trajectory", "Annual cyber rate")
        yoy = filtered.groupby("Year").agg(Total=("Is_Cyber", "count"), Cyber=("Is_Cyber", "sum")).reset_index()
        yoy["Rate"] = round(yoy["Cyber"] / yoy["Total"] * 100, 1)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=yoy["Year"], y=yoy["Cyber"], name="Cyber breaches",
                             marker_color=C_CYBER, opacity=0.8), secondary_y=False)
        fig.add_trace(go.Scatter(x=yoy["Year"], y=yoy["Rate"], name="Cyber rate (%)",
                             mode="lines+markers", line=dict(color=ACCENT, width=2.5)), secondary_y=True)
        fig.update_layout(template=TPL, height=330, hovermode="x unified",
                          yaxis_title="Cyber breaches", yaxis2_title="Cyber rate (%)")
        pchart(fig, key="ov_yoy")
        cap("Bars count cyber breaches each year; the line is their share of all breaches. "
            "Both climbing together signals cyber risk growing in volume <i>and</i> proportion.")

# ─────────────────────────────────────────────────────────────────────────────────
#  2 · TRENDS  (fragment: in-tab controls won't rerun the whole app)
# ─────────────────────────────────────────────────────────────────────────────────
@st.fragment
def render_trends():
    if filtered.empty:
        st.warning("No reports match the current filters."); return
    eyebrow("Over time", "Reporting trends")
    gran = st.selectbox("Group by", ["Year", "Year + quarter"], key="tr_gran")
    w = filtered.copy()
    w["Bucket"] = (w["Year"].astype(str) + " " + w["Quarter"].astype(str)
                   if gran == "Year + quarter" else w["Year"].astype(str))
    g = w.groupby(["Bucket", "Incident_Category"]).size().reset_index(name="Reports")
    fig = px.bar(g, x="Bucket", y="Reports", color="Incident_Category", barmode="group",
                 color_discrete_map=CAT_COLOURS, template=TPL, height=380)
    fig.update_layout(xaxis_title="")
    pchart(fig, key="tr_bar")
    cap("Compare reporting volumes across periods. Widening gaps show where breach reporting is accelerating.")

    c1, c2 = st.columns(2)
    with c1:
        sh = w.groupby(["Bucket", "Incident_Category"]).size().reset_index(name="Count")
        fig = px.area(sh, x="Bucket", y="Count", color="Incident_Category", groupnorm="fraction",
                      color_discrete_map=CAT_COLOURS, template=TPL, height=340)
        fig.update_layout(yaxis_title="Share", yaxis_tickformat=".0%", xaxis_title="")
        pchart(fig, key="tr_area")
        cap("The red band is cyber as a fraction of the total. A thickening band means cyber is crowding out other breach types.")
    with c2:
        sev = filtered.groupby("Year")["Severity_Score"].mean().round(2).reset_index()
        fig = px.line(sev, x="Year", y="Severity_Score", markers=True,
                      color_discrete_sequence=[C_PURPLE], template=TPL, height=340)
        fig.update_layout(yaxis_title="Avg severity (0–11)")
        pchart(fig, key="tr_sev")
        cap("Severity blends cyber status, people affected and data sensitivity. An upward slope means "
            "breaches are getting more damaging, not just more frequent.")

    n = st.slider("Breach types to track", 3, 8, 5, key="tr_n")
    top = filtered["Incident_Type"].value_counts().head(n).index.tolist()
    td = (filtered[filtered["Incident_Type"].isin(top)]
          .groupby(["Date", "Incident_Type"]).size().reset_index(name="Reports").sort_values("Date"))
    if not td.empty:
        fig = px.line(td, x="Date", y="Reports", color="Incident_Type", markers=True, template=TPL, height=380)
        pchart(fig, key="tr_types")
        cap("Each line is one breach type. Steep climbs flag emerging threats worth prioritising in training and controls.")

with T_TRENDS:
    render_trends()

# ─────────────────────────────────────────────────────────────────────────────────
#  3 · SECTORS  (fragment)
# ─────────────────────────────────────────────────────────────────────────────────
@st.fragment
def render_sectors():
    if filtered.empty:
        st.warning("No reports match the current filters."); return
    eyebrow("By industry", "Sector breakdown")
    top_n = st.slider("Sectors to display", 5, 20, 10, key="se_n")
    c1, c2 = st.columns(2)
    with c1:
        sc = filtered.groupby(["Sector", "Incident_Category"]).size().reset_index(name="Reports")
        keep = sc.groupby("Sector")["Reports"].sum().nlargest(top_n).index
        fig = px.bar(sc[sc["Sector"].isin(keep)], y="Sector", x="Reports", color="Incident_Category",
                     barmode="stack", orientation="h", color_discrete_map=CAT_COLOURS, template=TPL, height=520)
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        pchart(fig, key="se_stack")
        cap("Total reports per sector, split by cyber (red) and non-cyber (blue). "
            "A long red segment marks a sector under sustained attack.")
    with c2:
        cr = filtered.groupby("Sector")["Is_Cyber"].mean().mul(100).round(1).reset_index()
        cr.columns = ["Sector", "Rate"]
        cr = cr.sort_values("Rate", ascending=False).head(top_n)
        tmap = filtered.groupby("Sector")["Sector_Risk_Tier"].first().to_dict()
        cr["Tier"] = cr["Sector"].map(tmap).fillna("Medium")
        fig = px.bar(cr, x="Rate", y="Sector", orientation="h", color="Tier",
                     color_discrete_map=TIER_COLOURS, template=TPL, height=520)
        fig.update_layout(yaxis=dict(categoryorder="total ascending"),
                          xaxis_title="% of sector reports that are cyber")
        pchart(fig, key="se_rate")
        cap("Share of each sector&rsquo;s breaches that are cyber. Tier colour uses a volume-adjusted rate, "
            "so a small sector with one cyber report isn&rsquo;t mislabelled high-risk.")

    st.divider()
    pick = st.selectbox("Drill into a sector", ["All sectors"] +
                        sorted(filtered["Sector"].dropna().unique().tolist()), key="se_pick")
    sd = filtered if pick == "All sectors" else filtered[filtered["Sector"] == pick]
    d1, d2 = st.columns(2)
    with d1:
        it = sd["Incident_Type"].value_counts().head(10).reset_index(); it.columns = ["Type", "Reports"]
        fig = px.bar(it, x="Reports", y="Type", orientation="h", template=TPL, height=360,
                     title="Most common breach types", color_discrete_sequence=[C_CYBER])
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        pchart(fig, key="se_types")
        cap("The breach types most often reported in the selected sector.")
    with d2:
        dt = sd["Data_Type"].value_counts().head(10).reset_index(); dt.columns = ["Data", "Reports"]
        fig = px.bar(dt, x="Reports", y="Data", orientation="h", template=TPL, height=360,
                     title="Data categories affected", color_discrete_sequence=[C_TEAL])
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        pchart(fig, key="se_data")
        cap("Which categories of personal data are exposed. Special-category data carries the strictest UK GDPR duties.")

with T_SECTORS:
    render_sectors()

# ─────────────────────────────────────────────────────────────────────────────────
#  4 · IMPACT & RISK
# ─────────────────────────────────────────────────────────────────────────────────
with T_IMPACT:
    if filtered.empty:
        st.warning("No reports match the current filters.")
    else:
        eyebrow("How bad", "Impact & severity", "Breach size and how severity is distributed across sectors.")
        imp = filtered.groupby(["No_Data_Subjects_Affected", "Incident_Category"]).size().reset_index(name="Reports")
        fig = px.bar(imp, x="No_Data_Subjects_Affected", y="Reports", color="Incident_Category", barmode="group",
                     color_discrete_map=CAT_COLOURS, category_orders={"No_Data_Subjects_Affected": BANDS},
                     template=TPL, height=360)
        fig.update_layout(xaxis_title="People affected (band)")
        pchart(fig, key="im_bands")
        cap("Breach size, smallest to largest. Cyber breaches skew toward the bigger bands because attackers target whole databases.")

        top12 = filtered["Sector"].value_counts().head(12).index.tolist()
        fig = px.box(filtered[filtered["Sector"].isin(top12)], x="Sector", y="Severity_Score",
                     color="Incident_Category", color_discrete_map=CAT_COLOURS, template=TPL, height=440)
        fig.update_layout(xaxis_tickangle=-35, yaxis_title="Severity (0–11)", xaxis_title="")
        pchart(fig, key="im_box")
        cap("Each box spans the middle 50% of severity scores for a sector; the line is the median. "
            "Taller, higher boxes mean more variable and more serious breaches.")

# ─────────────────────────────────────────────────────────────────────────────────
#  5 · FEATURE INSIGHTS  (engineered-feature deep dive — fragment)
# ─────────────────────────────────────────────────────────────────────────────────
@st.fragment
def render_features():
    if filtered.empty:
        st.warning("No reports match the current filters."); return
    eyebrow("Engineered signals", "Feature insights",
            "Patterns that only surface once raw fields are turned into severity, sensitivity and risk features.")
    n = st.slider("Sectors to include", 5, 15, 8, key="fi_n")
    tops = filtered["Sector"].value_counts().head(n).index.tolist()

    st.markdown("##### Average severity by sector and year")
    hm = (filtered[filtered["Sector"].isin(tops)].groupby(["Sector", "Year"])["Severity_Score"]
          .mean().round(2).reset_index().pivot(index="Sector", columns="Year", values="Severity_Score"))
    fig = px.imshow(hm, color_continuous_scale="RdYlGn_r", labels=dict(color="Avg severity"),
                    aspect="auto", template=TPL, height=420)
    pchart(fig, key="fi_heat")
    cap("Darker red means more severe on average. Reading a row left-to-right shows whether a sector is getting "
        "worse or better over time — the single clearest view of where risk is concentrating.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### How the engineered signals relate")
        cc = [c for c in ["Is_Cyber", "Is_High_Impact", "Impact_Score",
                          "Is_Special_Category", "Within_72hrs", "Severity_Score"] if c in filtered.columns]
        corr = filtered[cc].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                        aspect="auto", template=TPL, height=400)
        pchart(fig, key="fi_corr")
        cap("+1 (blue) means two signals move together; −1 (red) means they move apart. Severity correlates with its "
            "own inputs by design; treat the rest as association, not cause.")
    with c2:
        st.markdown("##### Breach scale — cyber vs non-cyber")
        fig = px.histogram(filtered, x="Impact_Score", color="Incident_Category", barmode="overlay", opacity=0.7,
                           color_discrete_map=CAT_COLOURS, nbins=6, template=TPL, height=400,
                           labels={"Impact_Score": "Impact score (1 = 1–9 people … 6 = 100k+)", "count": "Reports"})
        fig.update_layout(yaxis_title="Reports")
        pchart(fig, key="fi_hist")
        cap("Where each breach type sits on the size scale. Cyber (red) leaning right confirms attackers hit larger "
            "record sets than typical human-error breaches.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### Severity behind each ICO decision")
        ds = (filtered.groupby("Decision_Taken")["Severity_Score"].mean().round(2)
              .reset_index().sort_values("Severity_Score"))
        fig = px.bar(ds, x="Severity_Score", y="Decision_Taken", orientation="h",
                     color_discrete_sequence=[ACCENT], template=TPL, height=380)
        fig.update_layout(yaxis=dict(categoryorder="total ascending"), xaxis_title="Avg severity", yaxis_title="")
        pchart(fig, key="fi_dec")
        cap("If tougher outcomes line up with higher average severity, the ICO&rsquo;s enforcement is tracking real risk.")
    with c4:
        st.markdown("##### Cyber + special-category exposure")
        dual = filtered[(filtered["Is_Cyber"] == 1) & (filtered["Is_Special_Category"] == 1)]
        if len(dual) > 0:
            dh = dual["Sector"].value_counts().head(n).reset_index(); dh.columns = ["Sector", "Breaches"]
            fig = px.bar(dh, x="Breaches", y="Sector", orientation="h",
                         color_discrete_sequence=[C_PURPLE], template=TPL, height=380)
            fig.update_layout(yaxis=dict(categoryorder="total ascending"), yaxis_title="")
            pchart(fig, key="fi_dual")
            cap("Breaches that are <i>both</i> cyber and involve special-category data — the highest-exposure "
                "combination for regulatory action.")
        else:
            st.info("No breaches in the current selection are both cyber and special-category.")

    st.markdown("##### Sector → category → outcome")
    tm = (filtered[filtered["Sector"].isin(tops)]
          .groupby(["Sector", "Incident_Category", "Decision_Taken"]).size().reset_index(name="Reports"))
    fig = px.treemap(tm, path=["Sector", "Incident_Category", "Decision_Taken"], values="Reports",
                     color="Reports", color_continuous_scale="Blues", template=TPL, height=480)
    fig.update_layout(margin=dict(t=20, b=10))
    pchart(fig, key="fi_tree")
    cap("Block size is report volume; click a sector to drill into its cyber/non-cyber split and the outcomes that followed.")

with T_FEATURE:
    render_features()

# ─────────────────────────────────────────────────────────────────────────────────
#  5 · MODELLING  (fragment)
# ─────────────────────────────────────────────────────────────────────────────────
@st.fragment
def render_model():
    model_results, best_name = models_and_best()
    eyebrow("Prediction", "Can we predict cyber vs non-cyber?",
            "Trained on breach characteristics only — incident type is withheld to avoid target leakage.")
    st.info("**Why incident type is withheld.** The ICO assigns the Cyber / Non-Cyber label directly from the "
            "incident type (ransomware → Cyber, a mis-sent email → Non-Cyber). Feeding it to the model would let it "
            "re-read the answer and inflate accuracy toward 100% — textbook **target leakage**. Withholding it forces "
            "the model to learn from independent signals (sector, data type, scale, timing). Scores are lower, but honest.")

    name = st.selectbox("Model", list(model_results.keys()), key="md_pick")
    r = model_results[name]; rep = r["report"]
    m = st.columns(4)
    m[0].metric("Accuracy", f"{rep['accuracy']:.3f}")
    m[1].metric("Precision (cyber)", f"{rep.get('1', {}).get('precision', 0):.3f}")
    m[2].metric("Recall (cyber)", f"{rep.get('1', {}).get('recall', 0):.3f}")
    m[3].metric("Test ROC-AUC", f"{r['auc']:.3f}")
    st.caption("5-fold cross-validated ROC-AUC: **{:.3f} ± {:.3f}**  ·  cyber base rate: **{:.1%}** "
               "(accuracy only beats guessing if it clears this).".format(
                   r["cv_auc_mean"], r["cv_auc_std"], r["base_rate"]))

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines",
                                 name="AUC " + f"{r['auc']:.3f}", line=dict(color=C_CYBER, width=2.5)))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#cbd5e1"))
        fig.update_layout(template=TPL, height=350, xaxis_title="False positive rate", yaxis_title="True positive rate")
        pchart(fig, key="md_roc")
        cap("The closer the curve hugs the top-left, the better the model separates cyber from non-cyber. "
            "The diagonal is random guessing.")
    with c2:
        fig = px.imshow(r["cm"], labels=dict(x="Predicted", y="Actual", color="Reports"),
                        x=["Non-cyber", "Cyber"], y=["Non-cyber", "Cyber"], text_auto=True,
                        color_continuous_scale="Blues", template=TPL, height=350)
        pchart(fig, key="md_cm")
        cap("Correct calls sit on the diagonal; off-diagonal cells are mistakes — showing whether the model "
            "misses cyber breaches or over-flags them.")

    if r.get("perm_importance") is not None:
        pi = r["perm_importance"].sort_values("Importance")
        fig = px.bar(pi, x="Importance", y="Feature", orientation="h", template=TPL, height=400,
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(yaxis=dict(categoryorder="total ascending"),
                          xaxis_title="Drop in ROC-AUC when the feature is shuffled", yaxis_title="")
        pchart(fig, key="md_perm")
        cap("How far ROC-AUC falls when each feature is shuffled — bigger drop, more the model leans on it. "
            "Model-agnostic, and read per real feature rather than per one-hot column.")

    st.divider()
    eyebrow("Comparison", "All three models")
    comp = pd.DataFrame([{
        "Model": k, "Accuracy": round(v["report"]["accuracy"], 3),
        "Precision": round(v["report"].get("1", {}).get("precision", 0), 3),
        "Recall": round(v["report"].get("1", {}).get("recall", 0), 3),
        "F1": round(v["report"].get("1", {}).get("f1-score", 0), 3),
        "Test AUC": round(v["auc"], 3),
        "CV AUC (mean)": round(v["cv_auc_mean"], 3),
        "CV AUC (sd)": round(v["cv_auc_std"], 3),
        "Selected": "✓" if k == best_name else "",
    } for k, v in model_results.items()])
    st.dataframe(comp, width="stretch", hide_index=True)

    # ---- Data-driven recommendation ----
    lr_auc = model_results["Logistic Regression"]["cv_auc_mean"]
    top_auc = model_results[best_name]["cv_auc_mean"]
    gap = top_auc - lr_auc
    eyebrow("Recommendation", "Which model to use, and why")
    if best_name == "Logistic Regression" or gap < 0.03:
        rec = ("**Recommended: Logistic Regression.** For this dashboard the goal is *explanation* in a regulatory "
               "setting, not squeezing out the last fraction of accuracy. Logistic Regression is the most "
               "interpretable of the three (each feature has a direction and weight you can defend to a "
               "non-technical audience), the fastest to train and serve — which matters on a free hosting tier — "
               "and it produces well-behaved probabilities for the Risk Predictor. ")
        if best_name != "Logistic Regression":
            rec += ("The best cross-validated model is **" + best_name + "**, but it leads by only " +
                    f"{gap:.3f}" + " AUC — too small to justify giving up that transparency and speed. ")
        else:
            rec += "It also has the best cross-validated AUC here, so there is no accuracy trade-off to make. "
    else:
        rec = ("**Recommended: " + best_name + ".** It has the strongest cross-validated ROC-AUC (ahead of "
               "Logistic Regression by " + f"{gap:.3f}" + "), a margin large enough to prefer it on predictive "
               "performance. Tree-based models capture interactions between sector, data type and scale that a "
               "linear model cannot. The interpretability gap is covered by the permutation-importance chart above, "
               "which explains the model in terms of real features. Logistic Regression remains the better choice "
               "if transparency or training cost outweighs a few points of AUC. ")
    rec += ("The app selects the top cross-validated model automatically for the Risk Predictor, so the choice "
            "stays evidence-based as the data updates.")
    st.markdown('<div class="finding" style="border-left-color:#2a9d8f">' + rec + '</div>', unsafe_allow_html=True)

    with st.expander("Model card — assumptions, intended use, limitations"):
        st.markdown(
            "- **Task** — binary classification: cyber vs non-cyber.\n"
            "- **Intended use** — exploratory pattern-finding and teaching. Not for operational triage, "
            "compliance decisions, or any automated action affecting people.\n"
            "- **Features** — sector, data subject type, data category, people-affected band, time-to-report, "
            "year, and engineered flags (special-category, impact score, within-72h). Incident type is excluded.\n"
            "- **Validation** — stratified 80/20 hold-out plus 5-fold cross-validation; the best model is chosen "
            "on mean CV ROC-AUC. Class imbalance is handled with balanced class weights where supported.\n"
            "- **Limitations** — self-reported data only; sector labels inconsistent across years; the target is "
            "partly definitional; outputs are probabilities, not facts.")

with T_MODEL:
    render_model()

# ─────────────────────────────────────────────────────────────────────────────────
#  6 · RISK PREDICTOR  (fragment)
# ─────────────────────────────────────────────────────────────────────────────────
@st.fragment
def render_predictor():
    model_results, best_name = models_and_best()
    eyebrow("Estimator", "Cyber breach risk",
            "Model: " + best_name + "  ·  CV ROC-AUC " + f"{model_results[best_name]['cv_auc_mean']:.3f}")
    st.warning("Exploratory only — not a substitute for breach assessment. Report a breach at "
               "[ico.org.uk/report-a-breach](https://ico.org.uk/for-organisations/report-a-breach/).")
    feats = model_results[best_name]["feats"]
    time_opts = (sorted(df_full["Time_Taken_to_Report"].dropna().unique().tolist())
                 if "Time_Taken_to_Report" in df_full.columns else ["Unknown"])
    c = st.columns(3)
    p_s    = c[0].selectbox("Sector", sorted(df_full["Sector"].dropna().unique().tolist()), key="pr_s")
    p_dst  = c[0].selectbox("Who was affected", sorted(df_full["Data_Subject_Type"].dropna().unique().tolist()), key="pr_dst")
    p_dt   = c[1].selectbox("Data category", sorted(df_full["Data_Type"].dropna().unique().tolist()), key="pr_dt")
    p_band = c[1].selectbox("People affected", BANDS, key="pr_band")
    p_time = c[2].selectbox("Time to report", time_opts, key="pr_time")
    p_year = c[2].selectbox("Year", sorted(df_full["Year"].dropna().unique().tolist()), key="pr_year")
    st.caption("Incident type is deliberately not an input — see the leakage note on the Modelling tab.")

    if st.button("Estimate cyber probability", width="stretch", type="primary"):
        is_sc = is_special_category(p_dt); imp = impact_score_from_band(p_band); w72 = is_within_72(p_time)
        cand = {"Sector": p_s, "Data_Subject_Type": p_dst, "Data_Type": p_dt,
                "No_Data_Subjects_Affected": p_band, "Time_Taken_to_Report": p_time,
                "Year": p_year, "Is_Special_Category": is_sc, "Impact_Score": imp, "Within_72hrs": w72}
        X_new = pd.DataFrame([{k: cand[k] for k in feats}])
        proba = model_results[best_name]["pipe"].predict_proba(X_new)[0, 1]
        label = "Cyber" if proba >= 0.5 else "Non-cyber"
        sev = 3 * int(proba >= 0.5) + imp + is_sc * 2
        g, t = st.columns([1, 2])
        with g:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=round(proba * 100, 1), number={"suffix": "%"},
                title={"text": "Cyber probability"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": C_CYBER if proba >= 0.5 else C_NONCYBER},
                       "steps": [{"range": [0, 40], "color": "#e8f5f1"},
                                 {"range": [40, 70], "color": "#fdf3e0"},
                                 {"range": [70, 100], "color": "#fbe5e9"}],
                       "threshold": {"line": {"color": INK, "width": 3}, "value": 50}}))
            fig.update_layout(height=300, margin=dict(t=60, b=20), template=TPL)
            pchart(fig, key="pr_gauge")
        with t:
            msg = ("**Predicted: " + label + "**  \nCyber probability: **" + f"{proba*100:.1f}%" +
                   "**  \nSeverity score: **" + str(sev) + " / 11**  \nSpecial-category data: **" +
                   ("yes" if is_sc else "no") + "**  \nModel: " + best_name)
            (st.error if label == "Cyber" else st.success)(msg)

with T_PREDICT:
    render_predictor()

# ─────────────────────────────────────────────────────────────────────────────────
#  7 · DATA & METHOD
# ─────────────────────────────────────────────────────────────────────────────────
with T_DATA:
    eyebrow("Transparency", "Data &amp; methodology", "What&rsquo;s real, what&rsquo;s engineered, and where it comes from.")

    st.markdown("#### How to use this dashboard")
    st.markdown(
        "Filter by year, sector and category in the sidebar — every tab updates together. **Overview** is the "
        "executive summary, **Trends / Sectors / Impact &amp; Risk / Feature Insights** are the exploratory analysis, "
        "**Modelling** explains "
        "the cyber-vs-non-cyber classifier and recommends a model, and **Risk Predictor** turns that model into an "
        "interactive estimate. Reading notes sit beneath each chart.")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Completeness")
        miss = df_full.isnull().sum().reset_index(); miss.columns = ["Field", "Missing"]
        miss["%"] = (miss["Missing"] / len(df_full) * 100).round(1)
        st.dataframe(miss.sort_values("Missing", ascending=False), width="stretch", hide_index=True)
        cap("Records missing a sector are dropped from sector charts, which can undercount some industries.")
    with c2:
        st.markdown("#### Reports received per year")
        yc = df_full["Year"].value_counts().sort_index().reset_index(); yc.columns = ["Year", "Reports"]
        fig = px.bar(yc, x="Year", y="Reports", template=TPL, color_discrete_sequence=[ACCENT], height=300)
        pchart(fig, key="da_year")
        cap("Context for the trends, and a check on whether any year is under-reported.")

    st.markdown("#### Regulatory outcome by breach category")
    pv = filtered.groupby(["Decision_Taken", "Incident_Category"]).size().reset_index(name="Reports")
    fig = px.density_heatmap(pv, x="Incident_Category", y="Decision_Taken", z="Reports",
                             color_continuous_scale="Purples", template=TPL, height=380)
    pchart(fig, key="da_dec")
    cap("Darker cells are more common outcome–category pairings. Cyber clustering in &lsquo;investigation pursued&rsquo; "
        "would point to heavier regulatory scrutiny.")

    st.divider()
    st.markdown("""
#### What each field means

| Field | Meaning |
|---|---|
| **Incident category** | Cyber (malicious, third-party) or Non-cyber (human error, physical loss) — assigned by the ICO |
| **Incident type** | How the breach happened (ransomware, mis-sent email, lost device…) |
| **Data subject type** | Who was affected — customers, employees, patients |
| **Data category** | The personal data involved, including special-category data |
| **Decision taken** | ICO response — no further action, investigation pursued, etc. |
| **People affected** | Estimated band of individuals impacted |
| **Sector** | Reporting organisation&rsquo;s industry |
| **Time to report** | Hours from discovery to ICO notification |

#### Engineered features (analytical constructs, not ICO labels)

| Feature | How it&rsquo;s built | Range |
|---|---|---|
| **Severity score** | 3 (cyber) + 1–6 (impact band) + 2 (special-category) | 0–11 |
| **Is special-category** | 1 if the data category matches Article 9 keywords | 0 / 1 |
| **Impact score** | Ordinal encoding of the people-affected band | 1–6 |
| **Within 72h** | 1 if reported inside the Article 33 window | 0 / 1 |
| **Sector risk tier** | High / Medium / Low from *shrinkage-adjusted* cyber-rate tertiles | 3 levels |

#### Modelling method
The classifier predicts the cyber / non-cyber category from breach characteristics. **Incident type is excluded** to
avoid target leakage (the ICO derives the category from it). Validation uses a stratified hold-out plus 5-fold
cross-validation, and the best model is selected on mean cross-validated ROC-AUC. See the Modelling tab for the
model recommendation and rationale.

#### Limitations
- Self-reported breaches only — unreported incidents are invisible here.
- Sector labels are not always consistent across historic records.
- People-affected figures are estimates at the time of notification.
- The cyber / non-cyber target is partly definitional, so the model is interpretive, not a compliance tool.
""")

    st.divider()
    st.markdown("#### Download the filtered dataset")
    csv_path = pathlib.Path("ico_breach_data_filtered.csv")
    if csv_path.exists():
        with open(csv_path, "rb") as f:
            st.download_button("⬇️  Download CSV", f.read(), file_name=csv_path.name, mime="text/csv",
                               width="stretch",
                               help="Static export generated offline and stored in the repo — keeps the free-tier server light.")
    else:
        st.info("Place **ico_breach_data_filtered.csv** in the app folder to enable the download.")

    st.markdown("""
<br>**Sources** ·
[ICO breach trends](https://ico.org.uk/action-weve-taken/complaints-and-concerns-data-sets/data-security-incident-trends/) ·
[Glossary](https://ico.org.uk/action-weve-taken/complaints-and-concerns-data-sets/data-security-incident-trends/glossary-of-terms/) ·
[Report a breach](https://ico.org.uk/for-organisations/report-a-breach/personal-data-breach/)
""", unsafe_allow_html=True)

st.divider()
st.caption("Data: UK Information Commissioner's Office (ico.org.uk) · Built with Streamlit & Plotly · Figures update with the sidebar filters.")
