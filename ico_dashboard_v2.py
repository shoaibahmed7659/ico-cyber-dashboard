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
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="ICO Data Security Incidents – Cyber Risk Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

PLOTLY_TEMPLATE = "plotly_white"
COLOR_CYBER   = "#dc2626"
COLOR_NONCYBER = "#2563eb"
COLOR_NEUTRAL = "#0f766e"
COLOR_WARN    = "#f59e0b"

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def section_note(title: str, text: str):
    with st.expander(title, expanded=False):
        st.write(text)

# -------------------------------------------------------
# LOAD + CLEAN DATA
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("ico_raw.csv")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "BI Reference"              : "BI_Reference",
        "Year"                      : "Year",
        "Quarter"                   : "Quarter",
        "Data Subject Type"         : "Data_Subject_Type",
        "Data Type"                 : "Data_Type",
        "Decision Taken"            : "Decision_Taken",
        "Incident Category"         : "Incident_Category",
        "Incident Type"             : "Incident_Type",
        "No. Data Subjects Affected": "No_Data_Subjects_Affected",
        "Sector"                    : "Sector",
        "Time Taken to Report"      : "Time_Taken_to_Report",
    })

    quarter_to_month = {"Qtr 1": 2, "Qtr 2": 5, "Qtr 3": 8, "Qtr 4": 11}
    df["Month"] = df["Quarter"].map(quarter_to_month).fillna(1).astype(int)
    df["Date"]  = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    )

    band_order = ["1 to 9", "10 to 99", "100 to 1k", "1k to 10k", "10k to 100k", "Over 100k"]
    df["No_Data_Subjects_Affected"] = pd.Categorical(
        df["No_Data_Subjects_Affected"], categories=band_order, ordered=True
    )

    # Derived flag
    high_bands = ["1k to 10k", "10k to 100k", "Over 100k"]
    df["High_Impact"] = df["No_Data_Subjects_Affected"].isin(high_bands)

    return df

df_full = load_data()

# -------------------------------------------------------
# TRAIN MODEL (cached on full dataset)
# -------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_models(df):
    data = df[df["Incident_Category"].isin(["Cyber", "Non Cyber"])].copy()
    y = (data["Incident_Category"] == "Cyber").astype(int)

    features = [
        "Sector", "Data_Subject_Type", "Data_Type",
        "Incident_Type", "No_Data_Subjects_Affected",
        "Time_Taken_to_Report", "Year"
    ]
    cat_feats = [f for f in features if f != "Year"]
    num_feats = ["Year"]

    preproc = ColumnTransformer([
        ("cat", Pipeline([
            ("imp",    SimpleImputer(strategy="most_frequent")),
            ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_feats),
        ("num", SimpleImputer(strategy="median"), num_feats),
    ])

    X = data[features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    results = {}
    for name, clf in [
        ("Logistic Regression",    LogisticRegression(max_iter=1000)),
        ("Random Forest",          RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
        ("Gradient Boosting",      GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)),
    ]:
        pipe = Pipeline([("preproc", preproc), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        rep     = classification_report(y_test, y_pred, output_dict=True)
        auc     = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        cm      = confusion_matrix(y_test, y_pred)
        results[name] = {
            "pipe": pipe, "report": rep, "auc": auc,
            "fpr": fpr, "tpr": tpr, "cm": cm,
            "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba
        }

        # Feature importances (RF only)
        if name == "Random Forest":
            try:
                ohe_names = (
                    results[name]["pipe"]
                    .named_steps["preproc"]
                    .named_transformers_["cat"]
                    .named_steps["ohe"]
                    .get_feature_names_out(cat_feats)
                )
                all_names = list(ohe_names) + num_feats
                importances = clf.feature_importances_
                fi_df = (
                    pd.DataFrame({"Feature": all_names, "Importance": importances})
                    .sort_values("Importance", ascending=False)
                    .head(20)
                )
                results[name]["feature_importance"] = fi_df
            except Exception:
                results[name]["feature_importance"] = None

    return results, X_test, y_test

model_results, X_test_global, y_test_global = train_models(df_full)
best_model_name = max(model_results, key=lambda k: model_results[k]["auc"])
best_pipe       = model_results[best_model_name]["pipe"]

# -------------------------------------------------------
# AUTO-INSIGHTS
# -------------------------------------------------------
def generate_insights(filtered):
    insights = []
    if filtered.empty:
        return ["No data matches the current filters — try broadening your selection."]

    top_sector = (
        filtered[filtered["Incident_Category"] == "Cyber"]["Sector"]
        .value_counts().idxmax()
        if (filtered["Incident_Category"] == "Cyber").any() else "N/A"
    )
    insights.append(
        f"🔴 **{top_sector}** reports the highest number of cyber incidents in the current selection."
    )

    cyber_pct = (filtered["Incident_Category"] == "Cyber").mean() * 100
    if cyber_pct > 30:
        insights.append(
            f"⚠️ Cyber incidents account for **{cyber_pct:.1f}%** of all incidents — above the typical baseline."
        )
    else:
        insights.append(
            f"✅ Cyber incidents account for **{cyber_pct:.1f}%** — within a normal range."
        )

    high_pct = filtered["High_Impact"].mean() * 100
    if high_pct > 10:
        insights.append(
            f"📊 **{high_pct:.1f}%** of incidents are high-impact (1k+ data subjects affected), "
            f"which may indicate systemic data handling weaknesses."
        )

    top_type = filtered["Incident_Type"].value_counts().idxmax()
    insights.append(f"📌 Most frequent incident type: **{top_type}**.")

    return insights

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.image(
    "https://ico.org.uk/media/about-the-ico/images/ico-logo-2019.png",
    use_container_width=True
)
st.sidebar.markdown("## Filters")

years      = sorted(df_full["Year"].dropna().unique())
sectors    = sorted(df_full["Sector"].dropna().unique())
categories = sorted(df_full["Incident_Category"].dropna().unique())

year_sel   = st.sidebar.multiselect("Year", years, default=years)
sector_sel = st.sidebar.multiselect("Sector", sectors, default=sectors)
cat_sel    = st.sidebar.multiselect("Incident category", categories, default=categories)

st.sidebar.markdown("---")
st.sidebar.caption(f"Records in view: **{len(df_full[df_full['Year'].isin(year_sel) & df_full['Sector'].isin(sector_sel) & df_full['Incident_Category'].isin(cat_sel)]):,}**")

filtered = df_full[
    df_full["Year"].isin(year_sel) &
    df_full["Sector"].isin(sector_sel) &
    df_full["Incident_Category"].isin(cat_sel)
].copy()

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.title("🔐 ICO Data Security Incidents – Cyber Risk Dashboard")
st.caption(
    "Interactive analytics dashboard built on UK ICO public data-security incident data. "
    "Explore patterns, assess sector risk, and use the predictive model to estimate cyber risk."
)

# -------------------------------------------------------
# TAB LAYOUT
# -------------------------------------------------------
tabs = st.tabs([
    "📊 Overview",
    "📈 Trends",
    "🏢 Sector Analysis",
    "⚠️ Impact Explorer",
    "🔍 EDA & Data Quality",
    "🤖 Predictive Model",
    "🔮 Risk Predictor",
    "💡 BI Insights",
    "📋 Data Notes",
])

# =======================================================
# TAB 0 – OVERVIEW
# =======================================================
with tabs[0]:
    st.subheader("Overview")
    st.info(
        "Quick snapshot of the dataset after applying your sidebar filters. "
        "KPI cards give you the headline numbers at a glance."
    )
    if filtered.empty:
        st.warning("No data for the selected filters.")
    else:
        total     = len(filtered)
        pct_cyber = (filtered["Incident_Category"] == "Cyber").mean() * 100
        pct_hi    = filtered["High_Impact"].mean() * 100
        yr_range  = f"{min(year_sel)} – {max(year_sel)}" if year_sel else "N/A"
        top_s     = filtered["Sector"].value_counts().idxmax()
        n_sectors = filtered["Sector"].nunique()

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Incidents",         f"{total:,}")
        c2.metric("% Cyber",                 f"{pct_cyber:.1f}%")
        c3.metric("% High-Impact",           f"{pct_hi:.1f}%")
        c4.metric("Years Covered",           yr_range)
        c5.metric("Top Reporting Sector",    top_s)
        c6.metric("Sectors in View",         str(n_sectors))

        section_note(
            "What do these KPIs mean?",
            "Total Incidents = all rows after filters. Cyber % = share classified as cyber-related. "
            "High-Impact % = share affecting 1,000+ data subjects. Top Sector = most frequently "
            "appearing sector in the current selection."
        )

        st.markdown("---")
        st.subheader("🔍 Automatic Insights")
        for ins in generate_insights(filtered):
            st.write(f"- {ins}")

        st.markdown("---")
        r1, r2 = st.columns([2, 1])
        with r1:
            st.subheader("Incidents over time")
            time_df = (
                filtered.groupby(["Date", "Incident_Category"])
                .size().reset_index(name="Incidents").sort_values("Date")
            )
            fig_t = px.line(
                time_df, x="Date", y="Incidents",
                color="Incident_Category", markers=True,
                color_discrete_map={"Cyber": COLOR_CYBER, "Non Cyber": COLOR_NONCYBER},
                template=PLOTLY_TEMPLATE, height=380
            )
            fig_t.update_layout(legend_title="Category")
            st.plotly_chart(fig_t, use_container_width=True)
        with r2:
            st.subheader("By category")
            cat_cnt = filtered["Incident_Category"].value_counts().reset_index()
            cat_cnt.columns = ["Category", "Count"]
            fig_c = px.pie(
                cat_cnt, names="Category", values="Count",
                color="Category",
                color_discrete_map={"Cyber": COLOR_CYBER, "Non Cyber": COLOR_NONCYBER},
                template=PLOTLY_TEMPLATE, height=380
            )
            st.plotly_chart(fig_c, use_container_width=True)

# =======================================================
# TAB 1 – TRENDS
# =======================================================
with tabs[1]:
    st.subheader("Trends")
    st.info("Explore how incidents change over time, broken down by different dimensions.")

    granularity = st.selectbox("Time granularity", ["Quarterly", "Yearly"], key="gran")
    work = filtered.copy()
    if granularity == "Quarterly":
        work["TimeBucket"] = work["Year"].astype(str) + " " + work["Quarter"]
    else:
        work["TimeBucket"] = work["Year"].astype(str)

    trend = work.groupby(["TimeBucket", "Incident_Category"]).size().reset_index(name="Incidents")
    fig_trend = px.bar(
        trend, x="TimeBucket", y="Incidents", color="Incident_Category",
        barmode="group",
        color_discrete_map={"Cyber": COLOR_CYBER, "Non Cyber": COLOR_NONCYBER},
        template=PLOTLY_TEMPLATE, height=420
    )
    fig_trend.update_layout(xaxis_title="Period", yaxis_title="Incidents", legend_title="Category")
    st.plotly_chart(fig_trend, use_container_width=True)

    section_note(
        "How to read this chart",
        "Each cluster of bars shows Cyber vs Non-Cyber incidents for a given time period. "
        "Upward trends in Cyber bars may indicate growing exposure or better reporting."
    )

    # Stacked area — share of cyber over time
    st.markdown("---")
    st.subheader("Cyber share over time (stacked area)")
    share_df = (
        work.groupby(["TimeBucket", "Incident_Category"])
        .size().reset_index(name="Incidents")
    )
    fig_area = px.area(
        share_df, x="TimeBucket", y="Incidents", color="Incident_Category",
        groupnorm="fraction",
        color_discrete_map={"Cyber": COLOR_CYBER, "Non Cyber": COLOR_NONCYBER},
        template=PLOTLY_TEMPLATE, height=380
    )
    fig_area.update_layout(yaxis_title="Share of incidents", legend_title="Category")
    st.plotly_chart(fig_area, use_container_width=True)

    # Top incident types trend
    st.markdown("---")
    st.subheader("Top incident types over time")
    n_types = st.slider("Number of types to show", 3, 8, 5, key="n_types")
    top_types = filtered["Incident_Type"].value_counts().head(n_types).index
    type_df = (
        filtered[filtered["Incident_Type"].isin(top_types)]
        .groupby(["Date", "Incident_Type"]).size().reset_index(name="Incidents")
        .sort_values("Date")
    )
    fig_tt = px.line(
        type_df, x="Date", y="Incidents", color="Incident_Type",
        markers=True, template=PLOTLY_TEMPLATE, height=400
    )
    st.plotly_chart(fig_tt, use_container_width=True)

# =======================================================
# TAB 2 – SECTOR ANALYSIS
# =======================================================
with tabs[2]:
    st.subheader("Sector Analysis")
    st.info("Understand which sectors report the most incidents and where cyber risk is concentrated.")

    col_a, col_b = st.columns(2)
    with col_a:
        top_n_s = st.slider("Top N sectors", 5, 20, 10, key="topn_s")
        sec_cnt = (
            filtered.groupby(["Sector", "Incident_Category"])
            .size().reset_index(name="Incidents")
        )
        top_sectors = (
            sec_cnt.groupby("Sector")["Incidents"].sum()
            .nlargest(top_n_s).index
        )
        sec_top = sec_cnt[sec_cnt["Sector"].isin(top_sectors)]
        fig_s = px.bar(
            sec_top, y="Sector", x="Incidents", color="Incident_Category",
            barmode="stack", orientation="h",
            color_discrete_map={"Cyber": COLOR_CYBER, "Non Cyber": COLOR_NONCYBER},
            template=PLOTLY_TEMPLATE, height=500
        )
        fig_s.update_layout(yaxis=dict(categoryorder="total ascending"), legend_title="Category")
        st.plotly_chart(fig_s, use_container_width=True)
        section_note(
            "How to read this",
            "Each bar = total incidents for a sector split by type. Longer red segments = "
            "higher cyber exposure. Use this to identify which sectors are most at risk."
        )

    with col_b:
        st.subheader("Cyber % by sector (heatmap)")
        cyber_by_sector = (
            filtered.groupby("Sector")["Incident_Category"]
            .apply(lambda x: (x == "Cyber").mean() * 100)
            .reset_index(name="Cyber_%")
            .sort_values("Cyber_%", ascending=False)
            .head(20)
        )
        fig_h = px.bar(
            cyber_by_sector, x="Cyber_%", y="Sector", orientation="h",
            color="Cyber_%", color_continuous_scale="Reds",
            template=PLOTLY_TEMPLATE, height=500
        )
        fig_h.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_h, use_container_width=True)

    # Sector drill-down
    st.markdown("---")
    st.subheader("Sector drill-down")
    sector_choice = st.selectbox(
        "Select a sector to drill into",
        ["All"] + sorted(filtered["Sector"].dropna().unique()),
        key="sector_dd"
    )
    sec_data = filtered if sector_choice == "All" else filtered[filtered["Sector"] == sector_choice]
    r1, r2 = st.columns(2)
    with r1:
        it_cnt = sec_data["Incident_Type"].value_counts().head(10).reset_index()
        it_cnt.columns = ["Incident_Type", "Count"]
        fig_it = px.bar(
            it_cnt, x="Count", y="Incident_Type", orientation="h",
            template=PLOTLY_TEMPLATE, height=380,
            title=f"Top incident types – {sector_choice}"
        )
        fig_it.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_it, use_container_width=True)
    with r2:
        dt_cnt = sec_data["Data_Type"].value_counts().head(10).reset_index()
        dt_cnt.columns = ["Data_Type", "Count"]
        fig_dt = px.pie(
            dt_cnt, names="Data_Type", values="Count",
            template=PLOTLY_TEMPLATE, height=380,
            title=f"Data types affected – {sector_choice}"
        )
        st.plotly_chart(fig_dt, use_container_width=True)

# =======================================================
# TAB 3 – IMPACT EXPLORER
# =======================================================
with tabs[3]:
    st.subheader("Impact Explorer")
    st.info(
        "Examine how many data subjects are affected by incident type and category, "
        "and compare two time periods side-by-side."
    )

    imp_df = (
        filtered.groupby(["No_Data_Subjects_Affected", "Incident_Category"])
        .size().reset_index(name="Incidents")
    )
    fig_imp = px.bar(
        imp_df, x="No_Data_Subjects_Affected", y="Incidents",
        color="Incident_Category", barmode="group",
        color_discrete_map={"Cyber": COLOR_CYBER, "Non Cyber": COLOR_NONCYBER},
        template=PLOTLY_TEMPLATE, height=420,
        category_orders={"No_Data_Subjects_Affected": [
            "1 to 9", "10 to 99", "100 to 1k", "1k to 10k", "10k to 100k", "Over 100k"
        ]}
    )
    fig_imp.update_layout(xaxis_title="Data subjects affected (band)", legend_title="Category")
    st.plotly_chart(fig_imp, use_container_width=True)
    section_note(
        "How to read this",
        "Bands on the x-axis represent the number of people whose data was involved. "
        "Taller bars on the right-hand side indicate higher-impact incidents."
    )

    # Period comparison
    st.markdown("---")
    st.subheader("Period-on-period comparison")
    all_years = sorted(df_full["Year"].dropna().unique())
    midpoint  = all_years[len(all_years) // 2]

    col1, col2 = st.columns(2)
    with col1:
        period_a = st.multiselect(
            "Period A (years)", all_years,
            default=[y for y in all_years if y < midpoint], key="pa"
        )
    with col2:
        period_b = st.multiselect(
            "Period B (years)", all_years,
            default=[y for y in all_years if y >= midpoint], key="pb"
        )

    if period_a and period_b:
        df_a = df_full[df_full["Year"].isin(period_a)]
        df_b = df_full[df_full["Year"].isin(period_b)]
        compare_metric = st.selectbox(
            "Compare by", ["Incident_Category", "Sector", "Incident_Type", "Data_Type"],
            key="cmp_metric"
        )
        cnt_a = df_a[compare_metric].value_counts().rename("Period A").reset_index()
        cnt_b = df_b[compare_metric].value_counts().rename("Period B").reset_index()
        cnt_a.columns = [compare_metric, "Period A"]
        cnt_b.columns = [compare_metric, "Period B"]
        merged = cnt_a.merge(cnt_b, on=compare_metric, how="outer").fillna(0)
        merged = merged.sort_values("Period A", ascending=False).head(15)
        melted = merged.melt(id_vars=compare_metric, var_name="Period", value_name="Count")
        fig_cmp = px.bar(
            melted, x="Count", y=compare_metric, color="Period",
            barmode="group", orientation="h",
            template=PLOTLY_TEMPLATE, height=450
        )
        fig_cmp.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.info("Select at least one year in each period to compare.")

# =======================================================
# TAB 4 – EDA & DATA QUALITY
# =======================================================
with tabs[4]:
    st.subheader("EDA & Data Quality")
    st.info(
        "Understand the structure and completeness of the dataset. "
        "These charts help assess whether patterns in the dashboard are based on clean, reliable data."
    )

    # Missing values
    missing = df_full.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing"]
    missing["Missing%"] = (missing["Missing"] / len(df_full) * 100).round(2)
    missing = missing.sort_values("Missing", ascending=False)

    st.markdown("### Missing value analysis")
    st.dataframe(missing, use_container_width=True)
    section_note(
        "Why check missing values?",
        "Missing values in key fields (Sector, Data Type, Incident Type) can distort charts "
        "and model outputs. A high missing rate in any field should be flagged before relying on it."
    )

    st.markdown("---")
    st.markdown("### Univariate distributions")
    col1, col2 = st.columns(2)
    with col1:
        cat_dist = filtered["Incident_Category"].value_counts().reset_index()
        cat_dist.columns = ["Category", "Count"]
        st.plotly_chart(
            px.bar(cat_dist, x="Category", y="Count", template=PLOTLY_TEMPLATE,
                   title="Incident Category", color="Category",
                   color_discrete_map={"Cyber": COLOR_CYBER, "Non Cyber": COLOR_NONCYBER}),
            use_container_width=True
        )
    with col2:
        dec_dist = filtered["Decision_Taken"].value_counts().head(10).reset_index()
        dec_dist.columns = ["Decision", "Count"]
        st.plotly_chart(
            px.bar(dec_dist, x="Count", y="Decision", orientation="h",
                   template=PLOTLY_TEMPLATE, title="Decision Taken (top 10)"),
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### Decision vs Incident Category (cross-tab heatmap)")
    pivot = (
        filtered.groupby(["Decision_Taken", "Incident_Category"])
        .size().reset_index(name="Count")
    )
    fig_pivot = px.density_heatmap(
        pivot, x="Incident_Category", y="Decision_Taken", z="Count",
        color_continuous_scale="Blues", template=PLOTLY_TEMPLATE, height=450
    )
    st.plotly_chart(fig_pivot, use_container_width=True)

    section_note(
        "How to read the cross-tab heatmap",
        "Darker cells = more incidents in that combination of decision and category. "
        "This helps identify whether cyber incidents are more likely to trigger investigations."
    )

    # Year distribution
    st.markdown("---")
    year_cnt = df_full["Year"].value_counts().sort_index().reset_index()
    year_cnt.columns = ["Year", "Count"]
    st.plotly_chart(
        px.bar(year_cnt, x="Year", y="Count", template=PLOTLY_TEMPLATE,
               title="Incidents per Year (full dataset)", color_discrete_sequence=[COLOR_NEUTRAL]),
        use_container_width=True
    )

# =======================================================
# TAB 5 – PREDICTIVE MODEL
# =======================================================
with tabs[5]:
    st.subheader("Predictive Model – Cyber vs Non-Cyber Classification")
    st.info(
        "Three machine learning models were trained and compared. "
        "Select a model below to inspect its performance metrics, ROC curve, "
        "confusion matrix, and feature importance."
    )

    model_choice = st.selectbox("Select model to inspect", list(model_results.keys()), key="mc")
    res = model_results[model_choice]
    rep = res["report"]

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{rep['accuracy']:.3f}")
    c2.metric("Precision (Cyber)", f"{rep.get('1', {}).get('precision', 0):.3f}")
    c3.metric("Recall (Cyber)",    f"{rep.get('1', {}).get('recall', 0):.3f}")
    c4.metric("ROC-AUC",  f"{res['auc']:.3f}")

    section_note(
        "What do these metrics mean?",
        "Accuracy = overall correctness. Precision (Cyber) = of all predicted cyber incidents, "
        "how many really are cyber. Recall (Cyber) = of all real cyber incidents, how many "
        "the model caught. ROC-AUC = overall discriminative ability (1.0 = perfect; 0.5 = random)."
    )

    st.markdown("---")
    col_roc, col_cm = st.columns(2)

    with col_roc:
        st.subheader("ROC Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=res["fpr"], y=res["tpr"],
            mode="lines", name=f"{model_choice} (AUC={res['auc']:.3f})",
            line=dict(color=COLOR_CYBER, width=2)
        ))
        fig_roc.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line=dict(dash="dash", color="grey")
        )
        fig_roc.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            template=PLOTLY_TEMPLATE, height=380
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        st.subheader("Confusion Matrix")
        cm = res["cm"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Non Cyber", "Cyber"], y=["Non Cyber", "Cyber"],
            text_auto=True, color_continuous_scale="Blues",
            template=PLOTLY_TEMPLATE, height=380
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        section_note(
            "How to read the confusion matrix",
            "Top-left = correctly predicted Non-Cyber. Bottom-right = correctly predicted Cyber. "
            "Off-diagonal cells = errors. High off-diagonal values suggest the model struggles "
            "to distinguish between the two classes."
        )

    # Feature importance (RF)
    if model_choice == "Random Forest" and res.get("feature_importance") is not None:
        st.markdown("---")
        st.subheader("Feature Importance (Random Forest)")
        fi = res["feature_importance"]
        fig_fi = px.bar(
            fi.sort_values("Importance"), x="Importance", y="Feature",
            orientation="h", template=PLOTLY_TEMPLATE, height=500,
            color="Importance", color_continuous_scale="Blues"
        )
        fig_fi.update_layout(yaxis=dict(categoryorder="total ascending"), coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)
        section_note(
            "What does feature importance show?",
            "Features higher on the list have more influence over whether an incident is "
            "classified as Cyber or Non-Cyber. This helps interpret what drives the model's decisions."
        )

    # Model comparison table
    st.markdown("---")
    st.subheader("Model comparison")
    comp = pd.DataFrame([
        {
            "Model": k,
            "Accuracy": f"{v['report']['accuracy']:.3f}",
            "Precision (Cyber)": f"{v['report'].get('1', {}).get('precision', 0):.3f}",
            "Recall (Cyber)":    f"{v['report'].get('1', {}).get('recall', 0):.3f}",
            "F1 (Cyber)":        f"{v['report'].get('1', {}).get('f1-score', 0):.3f}",
            "ROC-AUC":           f"{v['auc']:.3f}",
        }
        for k, v in model_results.items()
    ])
    st.dataframe(comp, use_container_width=True)
    st.caption(f"✅ Best model by ROC-AUC: **{best_model_name}**")

# =======================================================
# TAB 6 – RISK PREDICTOR
# =======================================================
with tabs[6]:
    st.subheader("🔮 Risk Predictor – Live Cyber Risk Estimation")
    st.info(
        "Select the characteristics of a hypothetical incident and the best model "
        f"(**{best_model_name}**) will estimate the probability it is cyber-related."
    )

    cA, cB, cC = st.columns(3)
    with cA:
        p_sector  = st.selectbox("Sector",            sorted(df_full["Sector"].dropna().unique()), key="p_s")
        p_subject = st.selectbox("Data subject type", sorted(df_full["Data_Subject_Type"].dropna().unique()), key="p_dst")
    with cB:
        p_dtype   = st.selectbox("Data type",         sorted(df_full["Data_Type"].dropna().unique()), key="p_dt")
        p_inc     = st.selectbox("Incident type",     sorted(df_full["Incident_Type"].dropna().unique()), key="p_it")
    with cC:
        p_band    = st.selectbox("Data subjects affected", sorted(df_full["No_Data_Subjects_Affected"].dropna().unique()), key="p_band")
        p_time    = st.selectbox("Time taken to report",   sorted(df_full["Time_Taken_to_Report"].dropna().unique()), key="p_time")
        p_year    = st.selectbox("Assumed year",           sorted(df_full["Year"].dropna().unique()), key="p_year")

    if st.button("🔍 Estimate Cyber Risk", use_container_width=True):
        X_new = pd.DataFrame([{
            "Sector": p_sector, "Data_Subject_Type": p_subject,
            "Data_Type": p_dtype, "Incident_Type": p_inc,
            "No_Data_Subjects_Affected": p_band,
            "Time_Taken_to_Report": p_time, "Year": p_year
        }])
        proba = best_pipe.predict_proba(X_new)[0, 1]
        label = "Cyber" if proba >= 0.5 else "Non Cyber"

        col_gauge, col_text = st.columns([1, 2])
        with col_gauge:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=proba * 100,
                title={"text": "Cyber Risk %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": COLOR_CYBER if proba >= 0.5 else COLOR_NONCYBER},
                    "steps": [
                        {"range": [0,  40], "color": "#dcfce7"},
                        {"range": [40, 70], "color": "#fef9c3"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": 50}
                },
                delta={"reference": 50}
            ))
            fig_g.update_layout(height=300)
            st.plotly_chart(fig_g, use_container_width=True)

        with col_text:
            if label == "Cyber":
                st.error(
                    f"**Classification: {label}**  
"
                    f"Estimated cyber probability: **{proba*100:.1f}%**  
"
                    f"Model used: *{best_model_name}*"
                )
            else:
                st.success(
                    f"**Classification: {label}**  
"
                    f"Estimated cyber probability: **{proba*100:.1f}%**  
"
                    f"Model used: *{best_model_name}*"
                )
            st.caption(
                "This model is trained on UK ICO historical incident data. "
                "It is intended for exploratory analysis and academic demonstration, "
                "not for operational decision-making."
            )

# =======================================================
# TAB 7 – BI INSIGHTS
# =======================================================
with tabs[7]:
    st.subheader("💡 Business Intelligence Insights")
    st.info(
        "Plain-language interpretation of what the data shows. "
        "Written for non-technical decision-makers."
    )

    if filtered.empty:
        st.warning("No data available for the current filters.")
    else:
        st.markdown("### Automated findings")
        for ins in generate_insights(filtered):
            st.write(f"- {ins}")

        # Decision outcomes
        st.markdown("---")
        st.subheader("Regulatory decisions – cyber vs non-cyber")
        dec_comp = (
            filtered.groupby(["Decision_Taken", "Incident_Category"])
            .size().reset_index(name="Count")
        )
        fig_dec = px.bar(
            dec_comp, x="Decision_Taken", y="Count", color="Incident_Category",
            barmode="group",
            color_discrete_map={"Cyber": COLOR_CYBER, "Non Cyber": COLOR_NONCYBER},
            template=PLOTLY_TEMPLATE, height=420
        )
        fig_dec.update_layout(xaxis_title="Regulatory decision", legend_title="Category")
        st.plotly_chart(fig_dec, use_container_width=True)
        section_note(
            "Why does the regulatory decision matter?",
            "The ICO's decision (No Further Action vs Investigation Pursued) indicates the "
            "severity of the reported incident. If cyber incidents consistently attract more "
            "formal investigations, this suggests they are treated as more serious by regulators."
        )

        # Reporting time analysis
        st.markdown("---")
        st.subheader("Reporting delay by sector and category")
        if "Time_Taken_to_Report" in filtered.columns:
            rpt = (
                filtered.groupby(["Sector", "Incident_Category"])["Time_Taken_to_Report"]
                .value_counts().reset_index(name="Count")
            )
            top_sec_rpt = filtered["Sector"].value_counts().head(8).index
            rpt_top = rpt[rpt["Sector"].isin(top_sec_rpt)]
            fig_rpt = px.density_heatmap(
                rpt_top,
                x="Incident_Category", y="Sector", z="Count",
                color_continuous_scale="YlOrRd", template=PLOTLY_TEMPLATE, height=420
            )
            st.plotly_chart(fig_rpt, use_container_width=True)

        # Download filtered
        st.markdown("---")
        csv_dl = filtered.to_csv(index=False)
        st.download_button(
            "⬇️ Download filtered dataset as CSV",
            csv_dl, "ico_filtered.csv", "text/csv",
            use_container_width=True
        )

# =======================================================
# TAB 8 – DATA NOTES
# =======================================================
with tabs[8]:
    st.subheader("📋 Data Notes & Methodology")
    st.markdown("""
### Data source
- **Dataset:** UK ICO Data Security Incident Trends (publicly available)
- **URL:** https://ico.org.uk/action-weve-taken/complaints-and-concerns-data-sets/data-security-incident-trends/
- **Columns used:** BI Reference, Year, Quarter, Data Subject Type, Data Type, Decision Taken,
  Incident Category, Incident Type, No. Data Subjects Affected, Sector, Time Taken to Report

### Data preparation
- Column names standardised and whitespace removed.
- Quarter mapped to representative month (Qtr 1 → Feb, Qtr 2 → May, etc.) to create a `Date` field.
- Impact bands treated as ordered categorical variable.
- High-impact flag created: incidents affecting ≥ 1,000 data subjects.

### Predictive modelling
- **Target variable:** Incident_Category (Cyber = 1, Non Cyber = 0)
- **Features:** Sector, Data Subject Type, Data Type, Incident Type, Impact Band, Reporting Time, Year
- **Models compared:** Logistic Regression, Random Forest, Gradient Boosting
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Best model deployed in Risk Predictor:** selected by ROC-AUC on hold-out test set (80/20 split, stratified)

### Limitations
- Model is trained on historical ICO reports and may not generalise to unreported incidents.
- Impact bands are ordinal ranges, not exact counts — precision is limited.
- The dashboard is designed for academic demonstration and exploratory analysis, not operational risk management.
- Temporal patterns should be interpreted cautiously as data collection practices may have changed over the period covered.

### Assessment context
- **Student:** Shoaib Ahmed | University of East Anglia (UEA)
- **Module:** Business Analytics
- **Dashboard deployed at:** https://ico-cyber-dashboard-sakhan.streamlit.app/
- **GitHub:** https://github.com/shoaibahmed7659/ico-cyber-dashboard
""")

st.markdown("---")
st.caption(
    "ICO Cyber Risk Dashboard · Built with Streamlit & Plotly · "
    "Data: UK Information Commissioner's Office · UEA Business Analytics Assessment"
)
