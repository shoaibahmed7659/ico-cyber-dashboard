import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICO Data Security Incident Explorer",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME CONSTANTS ──────────────────────────────────────────────────────────
TPL            = "plotly_white"
C_CYBER        = "#dc2626"
C_NONCYBER     = "#2563eb"
C_NEUTRAL      = "#0f766e"
C_WARN         = "#f59e0b"

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* tighter metric cards */
  [data-testid="metric-container"] {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 12px 16px;
  }
  [data-testid="metric-container"] label {
      font-size: 0.75rem !important;
      color: #64748b !important;
      text-transform: uppercase;
      letter-spacing: .05em;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
      font-size: 1.4rem !important;
      font-weight: 700;
      color: #0f172a !important;
  }
  /* sidebar header */
  section[data-testid="stSidebar"] h2 { margin-top: 0; }
  /* tab labels */
  button[data-baseweb="tab"] { font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ── HELPER ───────────────────────────────────────────────────────────────────
def note(title, body):
    with st.expander("ℹ️  " + title, expanded=False):
        st.markdown(body)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("ico_raw.csv")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "BI Reference"               : "BI_Reference",
        "Year"                       : "Year",
        "Quarter"                    : "Quarter",
        "Data Subject Type"          : "Data_Subject_Type",
        "Data Type"                  : "Data_Type",
        "Decision Taken"             : "Decision_Taken",
        "Incident Category"          : "Incident_Category",
        "Incident Type"              : "Incident_Type",
        "No. Data Subjects Affected" : "No_Data_Subjects_Affected",
        "Sector"                     : "Sector",
        "Time Taken to Report"       : "Time_Taken_to_Report",
    })
    q_map = {"Qtr 1": 2, "Qtr 2": 5, "Qtr 3": 8, "Qtr 4": 11}
    df["Month"] = df["Quarter"].map(q_map).fillna(1).astype(int)
    df["Date"]  = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    )
    bands = ["1 to 9", "10 to 99", "100 to 1k", "1k to 10k", "10k to 100k", "Over 100k"]
    df["No_Data_Subjects_Affected"] = pd.Categorical(
        df["No_Data_Subjects_Affected"], categories=bands, ordered=True
    )
    df["High_Impact"] = df["No_Data_Subjects_Affected"].isin(
        ["1k to 10k", "10k to 100k", "Over 100k"]
    )
    return df

df_full = load_data()

# ── TRAIN MODELS ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(df):
    data = df[df["Incident_Category"].isin(["Cyber", "Non Cyber"])].copy()
    data["No_Data_Subjects_Affected"] = data["No_Data_Subjects_Affected"].astype(str)
    y        = (data["Incident_Category"] == "Cyber").astype(int)
    features = ["Sector", "Data_Subject_Type", "Data_Type", "Incident_Type",
                "No_Data_Subjects_Affected", "Time_Taken_to_Report", "Year"]
    cats     = [f for f in features if f != "Year"]
    nums     = ["Year"]
    X        = data[features]
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
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random Forest",       RandomForestClassifier(n_estimators=150, max_depth=8,
                                                       random_state=42, n_jobs=-1)),
        ("Gradient Boosting",   GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                                            random_state=42)),
    ]
    results = {}
    for name, clf in clfs:
        pipe  = Pipeline([("preproc", preproc), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        yp    = pipe.predict(Xte)
        yprob = pipe.predict_proba(Xte)[:, 1]
        rep   = classification_report(yte, yp, output_dict=True)
        auc   = roc_auc_score(yte, yprob)
        fpr, tpr, _ = roc_curve(yte, yprob)
        cm    = confusion_matrix(yte, yp)
        entry = {"pipe": pipe, "report": rep, "auc": auc,
                 "fpr": fpr, "tpr": tpr, "cm": cm}
        if name == "Random Forest":
            try:
                ohe_s = pipe.named_steps["preproc"].named_transformers_["cat"].named_steps["ohe"]
                names = list(ohe_s.get_feature_names_out(cats)) + nums
                fi    = (pd.DataFrame({"Feature": names, "Importance": clf.feature_importances_})
                         .sort_values("Importance", ascending=False).head(20))
                entry["fi"] = fi
            except Exception:
                entry["fi"] = None
        results[name] = entry
    return results

model_results   = train_models(df_full)
best_name       = max(model_results, key=lambda k: model_results[k]["auc"])
best_pipe       = model_results[best_name]["pipe"]

# ── AUTO-INSIGHTS ────────────────────────────────────────────────────────────
def insights(df):
    if df.empty:
        return ["No data matches the current filters — try broadening your selection."]
    out   = []
    cyber = df["Incident_Category"] == "Cyber"
    if cyber.any():
        ts = df[cyber]["Sector"].value_counts().idxmax()
        tc = df[cyber]["Sector"].value_counts().iloc[0]
        out.append(
            "🔴 **" + ts + "** recorded the highest number of cyber breaches "
            "(" + str(tc) + " incidents) in this selection."
        )
    pct = round(cyber.mean() * 100, 1)
    if pct > 30:
        out.append(
            "⚠️ **" + str(pct) + "%** of breaches are cyber-related — above the dataset average of ~34%. "
            "The ICO defines a cyber breach as one involving a third party with malicious intent, "
            "such as phishing or malware."
        )
    else:
        out.append(
            "✅ Cyber breaches represent **" + str(pct) + "%** of incidents in this selection — "
            "within the typical range seen across the dataset."
        )
    hi = round(df["High_Impact"].mean() * 100, 1)
    if hi > 10:
        out.append(
            "📊 **" + str(hi) + "%** of breaches affected 1,000 or more people. "
            "The ICO notes that organisations must assess the potential harm to individuals "
            "when determining whether to report a breach."
        )
    tt  = df["Incident_Type"].value_counts().idxmax()
    ttc = df["Incident_Type"].value_counts().iloc[0]
    out.append(
        "📌 Most frequently reported breach type: **" + tt + "** (" + str(ttc) + " incidents). "
        "According to the ICO glossary, incident type describes how the breach occurred — "
        "whether through a deliberate act or an error."
    )
    top_dec = df["Decision_Taken"].value_counts().idxmax()
    out.append(
        "⚖️ The most common regulatory outcome in this selection is **" + top_dec + "**. "
        "The ICO's decision reflects whether formal or informal action was taken in response."
    )
    return out

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
try:
    st.sidebar.image(
        "https://ico.org.uk/media/about-the-ico/images/ico-logo-2019.png",
        use_container_width=True
    )
except Exception:
    st.sidebar.markdown("### 🔐 ICO Dashboard")

st.sidebar.markdown("## Filters")
st.sidebar.caption("Use the filters below to narrow the data shown across all tabs.")

years      = sorted(df_full["Year"].dropna().unique())
sectors    = sorted(df_full["Sector"].dropna().unique())
categories = sorted(df_full["Incident_Category"].dropna().unique())

year_sel   = st.sidebar.multiselect("Year", years, default=years)
sector_sel = st.sidebar.multiselect("Sector", sectors, default=sectors)
cat_sel    = st.sidebar.multiselect("Breach category", categories, default=categories)

filtered = df_full[
    df_full["Year"].isin(year_sel) &
    df_full["Sector"].isin(sector_sel) &
    df_full["Incident_Category"].isin(cat_sel)
].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Records in view", f"{len(filtered):,}")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data source:** [ICO Data Security Incident Trends](https://ico.org.uk/action-weve-taken/"
    "complaints-and-concerns-data-sets/data-security-incident-trends/)  \n"
    "Data covers personal data breach reports received by the ICO.  \n"
    "Latest update: up to Q2 2025."
)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🔐 ICO Data Security Incident Explorer")
st.markdown(
    "This tool visualises personal data breaches self-reported to the UK "
    "**Information Commissioner's Office (ICO)** between 2019 and Q2 2025. "
    "The ICO publishes this data to help organisations understand breach trends "
    "and take appropriate action under the UK GDPR."
)

# ── TABS ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "📈 Trends",
    "🏢 Sector Analysis",
    "⚠️ Impact Explorer",
    "🔍 Data Quality",
    "🤖 Predictive Model",
    "🔮 Risk Predictor",
    "💡 Key Insights",
    "📋 About the Data",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    if filtered.empty:
        st.warning("No data matches the selected filters. Please adjust the sidebar.")
    else:
        total     = len(filtered)
        pct_cyber = round((filtered["Incident_Category"] == "Cyber").mean() * 100, 1)
        pct_hi    = round(filtered["High_Impact"].mean() * 100, 1)
        yr_range  = (str(min(year_sel)) + " – " + str(max(year_sel))) if year_sel else "N/A"
        top_s     = filtered["Sector"].value_counts().idxmax()
        n_secs    = filtered["Sector"].nunique()
        max_yr    = filtered["Year"].max()

        # KPI row
        st.markdown("### Headline Figures")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total Breach Reports",      f"{total:,}")
        k2.metric("Cyber Breaches",            str(pct_cyber) + "%")
        k3.metric("High-Impact Breaches",      str(pct_hi) + "%")
        k4.metric("Period Covered",            yr_range)
        k5.metric("Most Reported Sector",      top_s)
        k6.metric("Sectors in Selection",      str(n_secs))

        note(
            "What do these figures mean?",
            "- **Total Breach Reports** — all personal data breach notifications received by the ICO in the selected period.  \n"
            "- **Cyber Breaches** — the share of breaches classified by the ICO as cyber-related. The ICO defines a cyber breach as one "
            "involving a third party with malicious intent, such as phishing or a ransomware attack.  \n"
            "- **High-Impact Breaches** — breaches where 1,000 or more people's personal data was affected.  \n"
            "- **Most Reported Sector** — the sector that submitted the highest number of breach reports in this selection.  \n"
            "  \n*Source: ICO Data Security Incident Trends, [ico.org.uk](https://ico.org.uk/action-weve-taken/complaints-and-concerns-data-sets/data-security-incident-trends/)*"
        )

        st.markdown("---")
        # Insights banner
        st.markdown("### What the data tells us")
        for ins in insights(filtered):
            st.info(ins)

        st.markdown("---")
        # Time series + category split — both as bar/line (NO PIE)
        c_left, c_right = st.columns([3, 2])
        with c_left:
            st.markdown("#### Breach reports over time")
            tdf = (
                filtered.groupby(["Date", "Incident_Category"])
                .size().reset_index(name="Breach Reports").sort_values("Date")
            )
            fig_t = px.line(
                tdf, x="Date", y="Breach Reports", color="Incident_Category",
                markers=True,
                color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER},
                template=TPL, height=340
            )
            fig_t.update_layout(legend_title="Breach Category", margin=dict(t=20, b=10))
            st.plotly_chart(fig_t, use_container_width=True)
            note(
                "How to read this chart",
                "Each line shows the number of breach reports submitted to the ICO per quarter. "
                "**Red** = cyber breaches (malicious/technical origin). **Blue** = non-cyber breaches "
                "(such as data sent to the wrong recipient, or lost paperwork). "
                "A rising red line over time may indicate growing cyber threat activity or improved reporting by organisations."
            )

        with c_right:
            st.markdown("#### Breach category split")
            cat_bar = filtered["Incident_Category"].value_counts().reset_index()
            cat_bar.columns = ["Category", "Count"]
            cat_bar["Share (%)"] = (cat_bar["Count"] / cat_bar["Count"].sum() * 100).round(1)
            fig_cb = px.bar(
                cat_bar, x="Category", y="Count",
                color="Category", text="Share (%)",
                color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER},
                template=TPL, height=340
            )
            fig_cb.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_cb.update_layout(showlegend=False, margin=dict(t=20, b=10),
                                 yaxis_title="Number of Breach Reports")
            st.plotly_chart(fig_cb, use_container_width=True)
            note(
                "How to read this chart",
                "This bar chart shows the total number of breach reports split between cyber and non-cyber categories. "
                "The percentage label on each bar shows the proportional share. "
                "Unlike a pie chart, the bar format makes it easier to compare absolute volumes as well as proportions."
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Breach Report Trends Over Time")
    st.markdown(
        "Explore how breach reporting volumes have changed year-on-year and quarter-on-quarter. "
        "The ICO publishes this data quarterly; the most recent update covers up to Q2 2025."
    )

    if filtered.empty:
        st.warning("No data for the selected filters.")
    else:
        gran = st.selectbox("Group by", ["Year", "Quarter (Year + Qtr)"], key="gran")
        work = filtered.copy()
        work["TimeBucket"] = (
            work["Year"].astype(str) + " " + work["Quarter"].astype(str)
            if gran == "Quarter (Year + Qtr)"
            else work["Year"].astype(str)
        )

        trend = work.groupby(["TimeBucket", "Incident_Category"]).size().reset_index(name="Breach Reports")
        fig_tr = px.bar(
            trend, x="TimeBucket", y="Breach Reports", color="Incident_Category",
            barmode="group",
            color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER},
            template=TPL, height=400
        )
        fig_tr.update_layout(xaxis_title="Period", legend_title="Breach Category")
        st.plotly_chart(fig_tr, use_container_width=True)
        note(
            "How to read this chart",
            "Each pair of bars represents a time period. The **red bar** = cyber breaches; the **blue bar** = non-cyber breaches. "
            "Comparing the height of the bars across periods helps identify whether breach reporting is increasing, decreasing, or stable. "
            "**Note on 2025:** the 2025 figures are lower than previous years because the dataset only covers up to Q2 2025 "
            "(January to June). The full year's data is not yet available. This is not a reduction in breaches — "
            "it reflects a partial year of reporting."
        )

        st.markdown("---")
        st.markdown("#### Cyber share as a proportion of all breaches (stacked area)")
        share = work.groupby(["TimeBucket", "Incident_Category"]).size().reset_index(name="Count")
        fig_a = px.area(
            share, x="TimeBucket", y="Count", color="Incident_Category",
            groupnorm="fraction",
            color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER},
            template=TPL, height=360
        )
        fig_a.update_layout(yaxis_title="Proportion of all breach reports", legend_title="Breach Category",
                            yaxis_tickformat=".0%")
        st.plotly_chart(fig_a, use_container_width=True)
        note(
            "How to read this chart",
            "This chart shows the **share** of cyber breaches relative to all breaches over time. "
            "Unlike the bar chart above, this removes the effect of overall volume changes and focuses purely on proportion. "
            "If the red area is growing, it means cyber breaches are becoming a larger share of all breaches reported — "
            "even if the total number of reports has not changed."
        )

        st.markdown("---")
        st.markdown("#### Top breach types over time")
        n_t = st.slider("Number of breach types to display", 3, 8, 5, key="n_types")
        top_types = filtered["Incident_Type"].value_counts().head(n_t).index.tolist()
        type_df   = (
            filtered[filtered["Incident_Type"].isin(top_types)]
            .groupby(["Date", "Incident_Type"]).size().reset_index(name="Reports")
            .sort_values("Date")
        )
        if not type_df.empty:
            fig_tt = px.line(
                type_df, x="Date", y="Reports", color="Incident_Type",
                markers=True, template=TPL, height=400
            )
            fig_tt.update_layout(legend_title="Breach Type")
            st.plotly_chart(fig_tt, use_container_width=True)
            note(
                "How to read this chart",
                "Each line represents one of the most commonly reported breach types in the dataset. "
                "Breach type is defined by the ICO as describing how the breach occurred — whether through a deliberate act or an accidental error. "
                "Rising lines for a particular type may indicate a growing threat that organisations should prioritise in their security planning."
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SECTOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Breach Reports by Sector")
    st.markdown(
        "The ICO collects breach reports from organisations across all sectors. "
        "Sectors with more reports may reflect larger data processing volumes, "
        "greater public awareness of reporting obligations, or higher levels of cyber threat activity. "
        "The ICO notes that sector labels are not always applied consistently, particularly in historic data."
    )

    if not filtered.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            top_n = st.slider("Number of sectors to display", 5, 20, 10, key="topn_s")
            sc    = filtered.groupby(["Sector", "Incident_Category"]).size().reset_index(name="Reports")
            tops  = sc.groupby("Sector")["Reports"].sum().nlargest(top_n).index
            fig_s = px.bar(
                sc[sc["Sector"].isin(tops)], y="Sector", x="Reports",
                color="Incident_Category", barmode="stack", orientation="h",
                color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER},
                template=TPL, height=520
            )
            fig_s.update_layout(yaxis=dict(categoryorder="total ascending"),
                                legend_title="Breach Category",
                                xaxis_title="Number of Breach Reports")
            st.plotly_chart(fig_s, use_container_width=True)
            note(
                "How to read this chart",
                "Each horizontal bar represents a sector. The total length = total breach reports from that sector. "
                "The **red segment** shows cyber-related breaches; the **blue segment** shows non-cyber breaches. "
                "A longer red segment relative to blue indicates a sector with a higher proportion of cyber incidents. "
                "Sectors appearing at the top of the chart have the highest total number of reports."
            )

        with col_b:
            st.markdown("#### Cyber breach rate by sector")
            st.markdown(
                "This chart shows the **percentage of each sector's breach reports that are classified as cyber breaches** "
                "by the ICO. A higher percentage means a greater share of that sector's breaches involved malicious intent."
            )
            cyber_rate = (
                filtered.groupby("Sector")["Incident_Category"]
                .apply(lambda x: round((x == "Cyber").mean() * 100, 1))
                .reset_index()
            )
            cyber_rate.columns = ["Sector", "Cyber Breach Rate (%)"]
            cyber_rate = cyber_rate.sort_values("Cyber Breach Rate (%)", ascending=False).head(20)
            fig_cr = px.bar(
                cyber_rate, x="Cyber Breach Rate (%)", y="Sector", orientation="h",
                color="Cyber Breach Rate (%)", color_continuous_scale="Reds",
                template=TPL, height=520,
                labels={"Cyber Breach Rate (%)": "Cyber Breach Rate (%)"}
            )
            fig_cr.update_layout(
                coloraxis_showscale=False,
                yaxis=dict(categoryorder="total ascending"),
                xaxis_title="% of that sector's reports classified as cyber breaches"
            )
            st.plotly_chart(fig_cr, use_container_width=True)
            note(
                "How to read this chart",
                "This bar shows what proportion of a sector's total breach reports are cyber-related. "
                "For example, a value of 60% means 6 in every 10 breaches from that sector were classified "
                "by the ICO as cyber breaches. This helps identify which sectors face a predominantly cyber threat "
                "versus those where human error or physical data loss is more common."
            )

        st.markdown("---")
        st.markdown("#### Drill down into a specific sector")
        sector_opts   = ["All sectors"] + sorted(filtered["Sector"].dropna().unique().tolist())
        sector_choice = st.selectbox("Select a sector", sector_opts, key="sector_dd")
        sec_df        = filtered if sector_choice == "All sectors" else filtered[filtered["Sector"] == sector_choice]

        d1, d2 = st.columns(2)
        with d1:
            it = sec_df["Incident_Type"].value_counts().head(10).reset_index()
            it.columns = ["Breach Type", "Reports"]
            fig_it = px.bar(
                it, x="Reports", y="Breach Type", orientation="h",
                template=TPL, height=380,
                title="Most common breach types — " + sector_choice,
                color_discrete_sequence=[C_CYBER]
            )
            fig_it.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig_it, use_container_width=True)
            note(
                "How to read this chart",
                "This shows the most frequently reported breach types for the selected sector. "
                "Breach types describe how the breach occurred — e.g. phishing (cyber) or data posted to wrong recipient (non-cyber). "
                "Identifying the dominant breach type helps organisations in that sector focus their prevention efforts."
            )
        with d2:
            dt = sec_df["Data_Type"].value_counts().head(10).reset_index()
            dt.columns = ["Data Category", "Reports"]
            fig_dt = px.bar(
                dt, x="Reports", y="Data Category", orientation="h",
                template=TPL, height=380,
                title="Data categories affected — " + sector_choice,
                color_discrete_sequence=[C_NEUTRAL]
            )
            fig_dt.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig_dt, use_container_width=True)
            note(
                "How to read this chart",
                "This shows which categories of personal data were most frequently compromised in reported breaches for this sector. "
                "Data type is defined by the ICO as the category of data compromised — including whether it is special category data "
                "(e.g. health data, racial or ethnic origin) or data about criminal convictions. "
                "Special category data carries stricter protection obligations under UK GDPR."
            )
    else:
        st.warning("No data for the selected filters.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — IMPACT EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Breach Impact — How Many People Were Affected?")
    st.markdown(
        "The ICO records the number of data subjects (people) whose personal data was involved in each breach. "
        "Organisations report the maximum number likely to be affected at the time of notification. "
        "These figures are grouped into bands rather than exact counts."
    )

    if not filtered.empty:
        band_order = ["1 to 9", "10 to 99", "100 to 1k", "1k to 10k", "10k to 100k", "Over 100k"]
        imp = (
            filtered.groupby(["No_Data_Subjects_Affected", "Incident_Category"])
            .size().reset_index(name="Breach Reports")
        )
        fig_imp = px.bar(
            imp, x="No_Data_Subjects_Affected", y="Breach Reports",
            color="Incident_Category", barmode="group",
            color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER},
            category_orders={"No_Data_Subjects_Affected": band_order},
            template=TPL, height=420
        )
        fig_imp.update_layout(
            xaxis_title="Number of people whose data was affected (band)",
            legend_title="Breach Category"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        note(
            "How to read this chart",
            "The x-axis shows bands representing the number of individuals whose personal data was involved in a breach. "
            "The y-axis shows how many breach reports fall into each band. "
            "**Taller bars on the right** indicate high-impact breaches affecting large numbers of people. "
            "Cyber breaches (red) tend to affect more people than non-cyber breaches (blue), "
            "as malicious attacks often target entire databases or systems rather than individual records. "
            "The ICO asks organisations to estimate the maximum number of affected individuals if the exact figure is unknown."
        )

        st.markdown("---")
        st.markdown("#### Compare two time periods side by side")
        all_yrs = sorted(df_full["Year"].dropna().unique().tolist())
        midpt   = all_yrs[len(all_yrs) // 2]
        p1, p2  = st.columns(2)
        with p1:
            period_a = st.multiselect(
                "Period A — earlier years", all_yrs,
                default=[y for y in all_yrs if y < midpt], key="pa"
            )
        with p2:
            period_b = st.multiselect(
                "Period B — later years", all_yrs,
                default=[y for y in all_yrs if y >= midpt], key="pb"
            )

        if period_a and period_b:
            cmp_col = st.selectbox(
                "What to compare",
                ["Incident_Category", "Sector", "Incident_Type", "Data_Type"],
                format_func=lambda x: {
                    "Incident_Category": "Breach Category (Cyber vs Non-Cyber)",
                    "Sector": "Sector",
                    "Incident_Type": "Breach Type",
                    "Data_Type": "Data Category Affected"
                }.get(x, x),
                key="cmp_metric"
            )
            da    = df_full[df_full["Year"].isin(period_a)]
            db    = df_full[df_full["Year"].isin(period_b)]
            ca    = da[cmp_col].value_counts().reset_index()
            cb    = db[cmp_col].value_counts().reset_index()
            ca.columns = [cmp_col, "Period A"]
            cb.columns = [cmp_col, "Period B"]
            mg    = ca.merge(cb, on=cmp_col, how="outer").fillna(0)
            mg    = mg.sort_values("Period A", ascending=False).head(15)
            melt  = mg.melt(id_vars=cmp_col, var_name="Period", value_name="Reports")
            fig_c = px.bar(
                melt, x="Reports", y=cmp_col, color="Period",
                barmode="group", orientation="h", template=TPL, height=450
            )
            fig_c.update_layout(yaxis=dict(categoryorder="total ascending"),
                                yaxis_title="", xaxis_title="Number of Breach Reports")
            st.plotly_chart(fig_c, use_container_width=True)
            note(
                "How to read this chart",
                "This comparison shows how breach volumes have changed between two periods you select. "
                "Longer bars for Period B (right group) compared to Period A indicate an increase. "
                "This can help identify whether specific sectors or breach types are becoming more or less prevalent over time."
            )
        else:
            st.info("Select at least one year in each period above to run the comparison.")
    else:
        st.warning("No data for the selected filters.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### Data Quality & Completeness")
    st.markdown(
        "Before drawing conclusions from any dataset, it is important to understand its completeness "
        "and any structural limitations. The ICO acknowledges in its own documentation that sector labels "
        "are not always applied consistently, particularly in historic data."
    )

    miss = df_full.isnull().sum().reset_index()
    miss.columns = ["Field", "Missing Values"]
    miss["Missing (%)"] = (miss["Missing Values"] / len(df_full) * 100).round(1)
    miss["ICO Definition"] = miss["Field"].map({
        "Sector"                    : "The type of organisation that reported the breach",
        "Data_Type"                 : "The category of personal data compromised",
        "Data_Subject_Type"         : "The type of person affected (e.g. customers, employees)",
        "Incident_Type"             : "How the breach occurred — deliberate act or error",
        "Incident_Category"         : "Whether the breach is cyber or non-cyber",
        "Decision_Taken"            : "The ICO's regulatory response to the breach report",
        "No_Data_Subjects_Affected" : "Estimated number of people whose data was affected",
        "Time_Taken_to_Report"      : "Hours taken to report the breach after discovery",
    }).fillna("—")
    st.dataframe(miss.sort_values("Missing Values", ascending=False), use_container_width=True)
    note(
        "Why do missing values matter?",
        "Missing values in key fields such as Sector or Incident Type can affect the accuracy of charts. "
        "For example, if 15% of records have no Sector label, sector-level charts may undercount certain industries. "
        "The ICO notes it is working to improve sector classification consistency in future updates."
    )

    if not filtered.empty:
        st.markdown("---")
        st.markdown("#### Distribution of records by year")
        yr_cnt = df_full["Year"].value_counts().sort_index().reset_index()
        yr_cnt.columns = ["Year", "Breach Reports"]
        fig_yr = px.bar(
            yr_cnt, x="Year", y="Breach Reports", template=TPL,
            color_discrete_sequence=[C_NEUTRAL],
            title="Total breach reports received by the ICO — by year"
        )
        fig_yr.update_layout(xaxis_title="Year", yaxis_title="Breach Reports")
        st.plotly_chart(fig_yr, use_container_width=True)
        note(
            "Why is 2025 lower than other years?",
            "The 2025 figure covers only **Q1 and Q2 (January to June 2025)**. "
            "The ICO last updated this dataset in October 2025, with data up to Q2 2025. "
            "The lower bar for 2025 does **not** indicate fewer breaches — it reflects a partial year of data. "
            "For fair year-on-year comparisons, either exclude 2025 or compare only Q1–Q2 across years."
        )

        st.markdown("---")
        st.markdown("#### Breach category vs regulatory decision (cross-tabulation)")
        pivot = (
            filtered.groupby(["Decision_Taken", "Incident_Category"])
            .size().reset_index(name="Reports")
        )
        fig_pv = px.density_heatmap(
            pivot, x="Incident_Category", y="Decision_Taken", z="Reports",
            color_continuous_scale="Blues", template=TPL, height=450,
            labels={"Incident_Category": "Breach Category",
                    "Decision_Taken": "Regulatory Decision",
                    "Reports": "Number of Reports"}
        )
        st.plotly_chart(fig_pv, use_container_width=True)
        note(
            "How to read this heatmap",
            "Darker blue = more breach reports in that combination of category and regulatory decision. "
            "The ICO's regulatory decision reflects whether any formal or informal action was taken. "
            "If cyber breaches are more concentrated in 'Investigation Pursued' cells, it suggests they are treated "
            "as higher risk by the ICO than non-cyber breaches. This is consistent with ICO guidance that cyber breaches "
            "often carry a higher risk to individuals' rights and freedoms."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICTIVE MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### Predicting Breach Category: Cyber vs Non-Cyber")
    st.markdown(
        "Three classification models were trained on the ICO breach dataset to predict whether a reported breach "
        "is likely to be cyber or non-cyber. Models were trained on 80% of the data and evaluated on the remaining 20%. "
        "This is an exploratory tool and is not intended for operational use."
    )

    m_choice = st.selectbox("Select a model to inspect", list(model_results.keys()), key="mc")
    res = model_results[m_choice]
    rep = res["report"]

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy",              str(round(rep["accuracy"], 3)))
    mc2.metric("Precision — Cyber",     str(round(rep.get("1", {}).get("precision", 0), 3)))
    mc3.metric("Recall — Cyber",        str(round(rep.get("1", {}).get("recall", 0), 3)))
    mc4.metric("ROC-AUC Score",         str(round(res["auc"], 3)))

    note(
        "What do these metrics mean?",
        "- **Accuracy** — the percentage of all records the model correctly classified (cyber or non-cyber).  \n"
        "- **Precision (Cyber)** — of all records the model predicted as cyber, what share truly are cyber. "
        "High precision = fewer false alarms.  \n"
        "- **Recall (Cyber)** — of all records that are genuinely cyber breaches, what share the model correctly identified. "
        "High recall = fewer missed cyber incidents.  \n"
        "- **ROC-AUC** — a score from 0 to 1 measuring the model's overall ability to distinguish cyber from non-cyber. "
        "1.0 = perfect; 0.5 = no better than random guessing. A score above 0.8 is generally considered strong."
    )

    st.markdown("---")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("#### ROC Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=res["fpr"], y=res["tpr"], mode="lines",
            name=m_choice + " (AUC = " + str(round(res["auc"], 3)) + ")",
            line=dict(color=C_CYBER, width=2)
        ))
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="grey"))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            template=TPL, height=380
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        note(
            "How to read the ROC Curve",
            "The ROC curve plots the model's true positive rate (correctly identified cyber breaches) "
            "against its false positive rate (non-cyber breaches incorrectly flagged as cyber) at different thresholds. "
            "A curve that hugs the top-left corner is better. The dashed diagonal = random guessing. "
            "The AUC (Area Under the Curve) summarises performance in a single number."
        )
    with rc2:
        st.markdown("#### Confusion Matrix")
        cm_val = res["cm"]
        fig_cm = px.imshow(
            cm_val,
            labels=dict(x="Predicted", y="Actual", color="Reports"),
            x=["Non-Cyber", "Cyber"], y=["Non-Cyber", "Cyber"],
            text_auto=True, color_continuous_scale="Blues",
            template=TPL, height=380
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        note(
            "How to read the Confusion Matrix",
            "This 2x2 grid shows prediction accuracy.  \n"
            "- **Top-left:** correctly predicted Non-Cyber (true negatives)  \n"
            "- **Bottom-right:** correctly predicted Cyber (true positives)  \n"
            "- **Top-right:** Non-Cyber breaches incorrectly flagged as Cyber (false positives)  \n"
            "- **Bottom-left:** Cyber breaches the model missed (false negatives)  \n"
            "Ideally the off-diagonal numbers should be as small as possible."
        )

    if m_choice == "Random Forest" and res.get("fi") is not None:
        st.markdown("---")
        st.markdown("#### Which factors most influence the prediction?")
        fi = res["fi"]
        fig_fi = px.bar(
            fi.sort_values("Importance"), x="Importance", y="Feature",
            orientation="h", template=TPL, height=520,
            color="Importance", color_continuous_scale="Blues",
            title="Feature Importance — Random Forest"
        )
        fig_fi.update_layout(yaxis=dict(categoryorder="total ascending"),
                             coloraxis_showscale=False,
                             xaxis_title="Relative Importance Score")
        st.plotly_chart(fig_fi, use_container_width=True)
        note(
            "How to read Feature Importance",
            "Feature importance shows how much each input variable contributed to the model's predictions. "
            "Higher bars = that factor has more influence over whether a breach is classified as cyber or non-cyber. "
            "For example, if 'Incident_Type' has a high importance score, it means the type of breach "
            "(e.g. phishing vs lost paperwork) is one of the strongest indicators of whether the breach is cyber-related."
        )

    st.markdown("---")
    st.markdown("#### All models at a glance")
    comp = []
    for k, v in model_results.items():
        comp.append({
            "Model"                 : k,
            "Accuracy"              : round(v["report"]["accuracy"], 3),
            "Precision (Cyber)"     : round(v["report"].get("1", {}).get("precision", 0), 3),
            "Recall (Cyber)"        : round(v["report"].get("1", {}).get("recall", 0), 3),
            "F1 Score (Cyber)"      : round(v["report"].get("1", {}).get("f1-score", 0), 3),
            "ROC-AUC"               : round(v["auc"], 3),
            "Best?"                 : "✅" if k == best_name else ""
        })
    st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)
    st.caption("Best model selected by ROC-AUC: **" + best_name + "**")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RISK PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### Cyber Breach Risk Estimator")
    st.markdown(
        "Select the characteristics of a hypothetical breach report and the best-performing model "
        "(**" + best_name + "**, ROC-AUC = " + str(round(model_results[best_name]["auc"], 3)) + ") "
        "will estimate the probability that it would be classified as a cyber breach by the ICO."
    )
    st.warning(
        "**Important:** This tool is for exploratory and illustrative purposes only. "
        "It is trained on historical ICO-reported breach data and should not be used to determine "
        "whether an actual breach should be reported to the ICO. "
        "If you have suffered a breach, visit [ico.org.uk/report-a-breach](https://ico.org.uk/for-organisations/report-a-breach/)."
    )

    band_opts = ["1 to 9", "10 to 99", "100 to 1k", "1k to 10k", "10k to 100k", "Over 100k"]
    pA, pB, pC = st.columns(3)
    with pA:
        p_sector  = st.selectbox("Sector", sorted(df_full["Sector"].dropna().unique().tolist()), key="p_s")
        p_subject = st.selectbox("Who was affected",
                                  sorted(df_full["Data_Subject_Type"].dropna().unique().tolist()), key="p_dst")
    with pB:
        p_dtype   = st.selectbox("Data category compromised",
                                  sorted(df_full["Data_Type"].dropna().unique().tolist()), key="p_dt")
        p_inc     = st.selectbox("How did the breach occur",
                                  sorted(df_full["Incident_Type"].dropna().unique().tolist()), key="p_it")
    with pC:
        p_band    = st.selectbox("Approximate number of people affected", band_opts, key="p_band")
        p_time    = st.selectbox("Time taken to report",
                                  sorted(df_full["Time_Taken_to_Report"].dropna().unique().tolist()), key="p_time")
        p_year    = st.selectbox("Year", sorted(df_full["Year"].dropna().unique().tolist()), key="p_year")

    if st.button("Estimate cyber breach probability", use_container_width=True):
        X_new = pd.DataFrame([{
            "Sector"                    : p_sector,
            "Data_Subject_Type"         : p_subject,
            "Data_Type"                 : p_dtype,
            "Incident_Type"             : p_inc,
            "No_Data_Subjects_Affected" : p_band,
            "Time_Taken_to_Report"      : p_time,
            "Year"                      : p_year,
        }])
        proba  = best_pipe.predict_proba(X_new)[0, 1]
        label  = "Cyber" if proba >= 0.5 else "Non-Cyber"
        pct_s  = str(round(proba * 100, 1)) + "%"

        g_col, t_col = st.columns([1, 2])
        with g_col:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba * 100, 1),
                number={"suffix": "%"},
                title={"text": "Estimated Cyber<br>Breach Probability"},
                gauge={
                    "axis"     : {"range": [0, 100]},
                    "bar"      : {"color": C_CYBER if proba >= 0.5 else C_NONCYBER},
                    "steps"    : [
                        {"range": [0,  40], "color": "#dcfce7"},
                        {"range": [40, 70], "color": "#fef9c3"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": 50}
                }
            ))
            fig_g.update_layout(height=300, margin=dict(t=60, b=20))
            st.plotly_chart(fig_g, use_container_width=True)

        with t_col:
            st.markdown("#### Result")
            msg = "Predicted category: **" + label + "**  \nEstimated probability: **" + pct_s + "**  \nModel used: " + best_name
            if label == "Cyber":
                st.error(msg)
            else:
                st.success(msg)
            st.markdown(
                "**What this means:** Based on similar historical breach reports in the ICO dataset, "
                "a breach with these characteristics has an estimated **" + pct_s + "** probability of "
                "being classified as a cyber breach. "
                "Cyber breaches are defined by the ICO as those with a clear online or technological element "
                "involving a third party with malicious intent."
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — KEY INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("### Key Findings from the Data")
    st.markdown(
        "The following findings are generated from the currently filtered dataset. "
        "They reflect patterns observed in breach reports submitted to the ICO and should be "
        "interpreted in the context of the data's limitations (see the **About the Data** tab)."
    )

    if filtered.empty:
        st.warning("No data available for the current filters.")
    else:
        for ins in insights(filtered):
            st.info(ins)

        st.markdown("---")
        st.markdown("#### Regulatory response: cyber vs non-cyber breaches")
        st.markdown(
            "The ICO's decision reflects the severity of the breach and whether formal or informal regulatory action was appropriate. "
            "Under Article 33 of the UK GDPR, organisations must report breaches to the ICO within 72 hours of discovery."
        )
        dec = (
            filtered.groupby(["Decision_Taken", "Incident_Category"])
            .size().reset_index(name="Reports")
        )
        fig_dec = px.bar(
            dec, x="Decision_Taken", y="Reports", color="Incident_Category",
            barmode="group",
            color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER},
            template=TPL, height=400
        )
        fig_dec.update_layout(
            xaxis_title="ICO Regulatory Decision",
            yaxis_title="Number of Breach Reports",
            legend_title="Breach Category"
        )
        st.plotly_chart(fig_dec, use_container_width=True)
        note(
            "How to read this chart",
            "Each group of bars shows how many cyber and non-cyber breaches received each type of regulatory decision. "
            "If cyber breaches more frequently result in 'Investigation Pursued', this aligns with ICO guidance "
            "that cyber breaches often carry a higher risk to individuals and may require closer scrutiny."
        )

        if "Time_Taken_to_Report" in filtered.columns:
            st.markdown("---")
            st.markdown("#### Reporting speed by sector — are organisations meeting the 72-hour requirement?")
            st.markdown(
                "Under the UK GDPR (Article 33), organisations are required to report a personal data breach to the ICO "
                "within **72 hours** of becoming aware of it. The ICO's glossary defines 'time taken to report' as the "
                "number of hours between discovery and notification."
            )
            top8 = filtered["Sector"].value_counts().head(8).index.tolist()
            rpt  = (
                filtered[filtered["Sector"].isin(top8)]
                .groupby(["Sector", "Incident_Category"])
                .size().reset_index(name="Reports")
            )
            if not rpt.empty:
                fig_r = px.bar(
                    rpt, x="Reports", y="Sector", color="Incident_Category",
                    barmode="group", orientation="h",
                    color_discrete_map={"Cyber": C_CYBER, "Non Cyber": C_NONCYBER},
                    template=TPL, height=420
                )
                fig_r.update_layout(yaxis=dict(categoryorder="total ascending"),
                                    legend_title="Breach Category",
                                    xaxis_title="Number of Breach Reports")
                st.plotly_chart(fig_r, use_container_width=True)

        st.markdown("---")
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered data as CSV",
            csv_bytes, "ico_breach_data_filtered.csv", "text/csv",
            use_container_width=True
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — ABOUT THE DATA
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown("### About This Dashboard & the Underlying Data")
    st.markdown("""
#### Data Source
This dashboard is built on the **ICO Data Security Incident Trends** dataset,
published by the [UK Information Commissioner's Office (ICO)](https://ico.org.uk).

The ICO is the UK's independent authority for upholding information rights. It publishes
breach report data to help organisations understand the breach landscape and meet their
obligations under the **UK General Data Protection Regulation (UK GDPR)** and the
**Data Protection Act 2018**.

> *"We publish this information to help organisations understand what to look out for and
> help them to take appropriate action."*
> — ICO, Data Security Incident Trends

---

#### What the data contains
Each row represents a single personal data breach self-reported to the ICO by an organisation.
The ICO defines a personal data breach as:

> *"A breach of security leading to the accidental or unlawful destruction, loss, alteration,
> unauthorised disclosure of, or access to, personal data."*

The dataset includes the following fields (using ICO's own definitions from the
[ICO Glossary of Terms](https://ico.org.uk/action-weve-taken/complaints-and-concerns-data-sets/data-security-incident-trends/glossary-of-terms/)):

| Field | ICO Definition |
|---|---|
| **Breach Category** | Whether the incident is cyber (malicious, third-party) or non-cyber (e.g. human error, physical loss) |
| **Breach Type** | How the breach occurred — deliberate act or error |
| **Data Subject Type** | Who was affected: customers, employees, patients, children, etc. |
| **Data Category** | What type of personal data was compromised, including special category data |
| **Regulatory Decision** | The ICO's response: No Further Action, Investigation Pursued, or other formal action |
| **Number of People Affected** | Estimated band of individuals whose data was involved |
| **Sector** | The type of organisation that reported the breach |
| **Time to Report** | Hours from breach discovery to ICO notification |

---

#### Reporting obligation
Under Article 33 of the UK GDPR, organisations must notify the ICO of a personal data breach
**within 72 hours** of becoming aware of it, where the breach is likely to result in a risk
to individuals' rights and freedoms.

---

#### Data coverage and limitations
- Data covers **2019 to Q2 2025** (January–June 2025). The ICO updates the dataset quarterly.
- **2025 figures are partial** — they cover only Q1 and Q2. The lower bar for 2025 in trend charts
  does not indicate fewer breaches; it reflects incomplete annual data.
- **Sector labels** are not always applied consistently, particularly in historic records.
  The ICO acknowledges this limitation and is working to improve classification in future releases.
- This dataset covers **self-reported breaches only**. Breaches that were not discovered or not
  reported are not reflected in the data.
- The number of data subjects affected is an **estimate provided at the time of reporting**.
  Organisations are advised to indicate the maximum number that may be affected.

---

#### Predictive model — important notice
The machine learning models in this dashboard were trained on historical ICO breach data for
exploratory and analytical purposes. They are **not endorsed by the ICO** and should not be
used to determine whether a real breach requires reporting. For breach reporting guidance,
visit [ico.org.uk/report-a-breach](https://ico.org.uk/for-organisations/report-a-breach/).

---

#### Further reading
- [ICO Data Security Incident Trends](https://ico.org.uk/action-weve-taken/complaints-and-concerns-data-sets/data-security-incident-trends/)
- [ICO Glossary of Terms](https://ico.org.uk/action-weve-taken/complaints-and-concerns-data-sets/data-security-incident-trends/glossary-of-terms/)
- [ICO: Incident Categories](https://ico.org.uk/action-weve-taken/complaints-and-concerns-data-sets/data-security-incident-trends/glossary-of-terms/incident-categories/)
- [UK GDPR Article 33 — Breach Notification](https://ico.org.uk/for-organisations/report-a-breach/personal-data-breach/)
- [ICO: Responding to a Cybersecurity Incident](https://ico.org.uk/media2/migrated/2614816/responding-to-a-cybersecurity-incident.pdf)
""")

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: UK Information Commissioner's Office (ICO) — "
    "ico.org.uk/action-weve-taken/complaints-and-concerns-data-sets/data-security-incident-trends/ | "
    "Built with Streamlit and Plotly"
)
