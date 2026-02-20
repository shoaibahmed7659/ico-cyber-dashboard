import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="ICO Data Security Incidents – Cyber Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# LOAD + CLEAN DATA
# -------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ico_raw.csv")

    df = df.rename(columns={
        "BI Reference": "BI_Reference",
        "Year": "Year",
        "Quarter": "Quarter",
        "Data Subject Type": "Data_Subject_Type",
        "Data Type": "Data_Type",
        "Decision Taken": "Decision_Taken",
        "Incident Category": "Incident_Category",
        "Incident Type": "Incident_Type",
        "No. Data Subjects Affected": "No_Data_Subjects_Affected",
        "Sector": "Sector",
        "Time Taken to Report": "Time_Taken_to_Report"
    })

    quarter_to_month = {"Qtr 1": 2, "Qtr 2": 5, "Qtr 3": 8, "Qtr 4": 11}
    df["Month"] = df["Quarter"].map(quarter_to_month).fillna(1).astype(int)
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" +
                                df["Month"].astype(str) + "-01")

    band_order = [
        "1 to 9",
        "10 to 99",
        "100 to 1k",
        "1k to 10k",
        "10k to 100k",
        "Over 100k"
    ]
    df["No_Data_Subjects_Affected"] = pd.Categorical(
        df["No_Data_Subjects_Affected"],
        categories=band_order,
        ordered=True
    )

    return df

df = load_data()

# -------------------------------------------------------
# SIMPLE DEMO MODEL – Cyber vs Non‑Cyber
# -------------------------------------------------------
@st.cache_resource
def train_demo_model(df):
    data = df.copy()
    data = data[data["Incident_Category"].isin(["Cyber", "Non Cyber"])]

    y = (data["Incident_Category"] == "Cyber").astype(int)

    features = [
        "Sector",
        "Data_Subject_Type",
        "Data_Type",
        "Incident_Type",
        "No_Data_Subjects_Affected",
        "Time_Taken_to_Report",
        "Year"
    ]
    X = data[features]

    cat_features = [
        "Sector",
        "Data_Subject_Type",
        "Data_Type",
        "Incident_Type",
        "No_Data_Subjects_Affected",
        "Time_Taken_to_Report"
    ]
    num_features = ["Year"]

    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preproc),
            ("logreg", LogisticRegression(max_iter=1000))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)

    # For your report: quick metrics in text form
    y_pred = model.predict(X_test)
    rep = classification_report(y_test, y_pred, output_dict=False)
    return model, rep

model, model_report_text = train_demo_model(df)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.title("Filters")

years = sorted(df["Year"].dropna().unique())
sectors = sorted(df["Sector"].dropna().unique())
categories = sorted(df["Incident_Category"].dropna().unique())

year_sel = st.sidebar.multiselect("Year", years, default=years)
sector_sel = st.sidebar.multiselect("Sector", sectors, default=sectors)
cat_sel = st.sidebar.multiselect("Incident category", categories, default=categories)

filtered = df[
    df["Year"].isin(year_sel)
    & df["Sector"].isin(sector_sel)
    & df["Incident_Category"].isin(cat_sel)
].copy()

# -------------------------------------------------------
# HEADER + KPIs
# -------------------------------------------------------
st.title("ICO Data Security Incidents – Cyber Risk Dashboard")
st.caption("For exploring UK ICO data‑security incidents in a clear, interactive, lay‑friendly way.")

total_incidents = len(filtered)
pct_cyber = (filtered["Incident_Category"] == "Cyber").mean() * 100 if total_incidents > 0 else 0.0

high_bands = ["1k to 10k", "10k to 100k", "Over 100k"]
high_impact = filtered[filtered["No_Data_Subjects_Affected"].isin(high_bands)]
pct_high_impact = len(high_impact) / total_incidents * 100 if total_incidents > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total incidents", f"{total_incidents:,}", help="Total rows in the ICO incident file after filters.")
c2.metric("% Cyber incidents", f"{pct_cyber:0.1f}%",
          help="Share of incidents recorded as 'Cyber' in the Incident Category field.")
c3.metric("% High‑impact incidents", f"{pct_high_impact:0.1f}%",
          help="Incidents affecting 1k+ data subjects; you can adjust this threshold in the code.")
c4.metric("Years covered", f"{min(year_sel)} – {max(year_sel)}" if year_sel else "N/A",
          help="Filtered year range currently displayed.")

# -------------------------------------------------------
# ROW 1 – TIME & CATEGORY
# -------------------------------------------------------
r1c1, r1c2 = st.columns([2, 1])

with r1c1:
    st.subheader("Incidents over time")
    if len(filtered) > 0:
        time_counts = (
            filtered.groupby(["Date", "Incident_Category"])
            .size()
            .reset_index(name="Incidents")
            .sort_values("Date")
        )
        fig_time = px.line(
            time_counts,
            x="Date",
            y="Incidents",
            color="Incident_Category",
            markers=True,
            template="plotly_white",
            hover_data={"Incidents": ":,", "Date": "|%b %Y"}
        )
        fig_time.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of incidents",
            legend_title="Category",
            height=400
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No data for the selected filters.")

with r1c2:
    st.subheader("By incident category")
    if len(filtered) > 0:
        cat_counts = filtered["Incident_Category"].value_counts().reset_index()
        cat_counts.columns = ["Incident_Category", "Incidents"]
        fig_cat = px.bar(
            cat_counts,
            x="Incident_Category",
            y="Incidents",
            text="Incidents",
            template="plotly_white"
        )
        fig_cat.update_traces(textposition="outside")
        fig_cat.update_layout(
            xaxis_title="Category",
            yaxis_title="Incidents",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("No data for the selected filters.")

# -------------------------------------------------------
# ROW 2 – SECTOR HEATMAP + INCIDENT TYPE TREND
# -------------------------------------------------------
st.markdown("---")
r2c1, r2c2 = st.columns([1.4, 1.6])

with r2c1:
    st.subheader("Sector vs category (where are incidents reported?)")
    if len(filtered) > 0:
        pivot = (
            filtered.groupby(["Sector", "Incident_Category"])
            .size()
            .reset_index(name="Incidents")
        )
        fig_heat = px.density_heatmap(
            pivot,
            x="Incident_Category",
            y="Sector",
            z="Incidents",
            color_continuous_scale="Blues",
            template="plotly_white"
        )
        fig_heat.update_layout(
            xaxis_title="Incident category",
            yaxis_title="Sector",
            height=500
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No data for the selected filters.")

with r2c2:
    st.subheader("Top incident types – trend over time")
    if len(filtered) > 0:
        # Let user choose how many types to show (to avoid clutter)
        max_types = st.slider("Number of top incident types to show",
                              min_value=3, max_value=8, value=5, step=1)

        top_types = filtered["Incident_Type"].value_counts().head(max_types).index
        subset = filtered[filtered["Incident_Type"].isin(top_types)]
        type_time = (
            subset.groupby(["Date", "Incident_Type"])
            .size()
            .reset_index(name="Incidents")
            .sort_values("Date")
        )

        # Use area chart for smoother, more readable view
        fig_type_trend = px.area(
            type_time,
            x="Date",
            y="Incidents",
            color="Incident_Type",
            groupnorm="fraction",  # show as share over time
            template="plotly_white"
        )
        fig_type_trend.update_layout(
            xaxis_title="Date",
            yaxis_title="Share of incidents",
            legend_title="Incident type",
            height=520
        )
        st.plotly_chart(fig_type_trend, use_container_width=True)
    else:
        st.info("No data for the selected filters.")


# -------------------------------------------------------
# ROW 3 – IMPACT DISTRIBUTION + SECTOR DRILLDOWN
# -------------------------------------------------------
st.markdown("---")
st.subheader("Impact and sector drill‑down")

r3c1, r3c2 = st.columns([1.3, 1])

with r3c1:
    if len(filtered) > 0:
        impact_counts = (
            filtered.groupby(["No_Data_Subjects_Affected", "Incident_Category"])
            .size()
            .reset_index(name="Incidents")
        )
        fig_impact = px.bar(
            impact_counts,
            x="No_Data_Subjects_Affected",
            y="Incidents",
            color="Incident_Category",
            barmode="group",
            template="plotly_white"
        )
        fig_impact.update_layout(
            xaxis_title="Data subjects affected (band)",
            yaxis_title="Incidents",
            height=400
        )
        st.plotly_chart(fig_impact, use_container_width=True)
    else:
        st.info("No data for the selected filters.")

with r3c2:
    # Sector drilldown: select a sector and see incident‑type breakdown
    sector_choice = st.selectbox(
        "Drill‑down: choose a sector",
        options=["All"] + list(sorted(filtered["Sector"].dropna().unique()))
    )
    if len(filtered) > 0:
        if sector_choice != "All":
            sec_data = filtered[filtered["Sector"] == sector_choice]
        else:
            sec_data = filtered

        sec_counts = (
            sec_data["Incident_Type"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        sec_counts.columns = ["Incident_Type", "Incidents"]
        fig_sector_drill = px.bar(
            sec_counts,
            y="Incident_Type",
            x="Incidents",
            orientation="h",
            template="plotly_white"
        )
        fig_sector_drill.update_layout(
            xaxis_title="Incidents",
            yaxis_title=f"Incident type ({sector_choice})",
            height=400
        )
        st.plotly_chart(fig_sector_drill, use_container_width=True)
    else:
        st.info("No data for the selected filters.")

# -------------------------------------------------------
# PREDICTION PANEL – DEMO MODEL INTEGRATED
# -------------------------------------------------------
st.markdown("---")
st.header("Incident risk explorer – live demo model")

st.write(
    "Select a hypothetical incident and the model will estimate the probability "
    "that it is **Cyber‑related** (vs Non Cyber). This is a simple demo model "
    "based on historical ICO reports."
)

cA, cB, cC = st.columns(3)

with cA:
    p_sector = st.selectbox("Sector", sorted(df["Sector"].dropna().unique()))
    p_subject = st.selectbox("Data subject type",
                             sorted(df["Data_Subject_Type"].dropna().unique()))

with cB:
    p_dtype = st.selectbox("Data type", sorted(df["Data_Type"].dropna().unique()))
    p_inc_type = st.selectbox("Incident type",
                              sorted(df["Incident_Type"].dropna().unique()))

with cC:
    p_band = st.selectbox("Data subjects affected (band)",
                          sorted(df["No_Data_Subjects_Affected"].dropna().unique()))
    p_time = st.selectbox("Time taken to report",
                          sorted(df["Time_Taken_to_Report"].dropna().unique()))
    p_year = st.selectbox("Assumed year", sorted(df["Year"].dropna().unique()))

if st.button("Estimate cyber risk"):
    # Prepare single row for prediction
    X_new = pd.DataFrame([{
        "Sector": p_sector,
        "Data_Subject_Type": p_subject,
        "Data_Type": p_dtype,
        "Incident_Type": p_inc_type,
        "No_Data_Subjects_Affected": p_band,
        "Time_Taken_to_Report": p_time,
        "Year": p_year
    }])
    proba_cyber = model.predict_proba(X_new)[0, 1]
    label = "Cyber" if proba_cyber >= 0.5 else "Non Cyber"

    st.success(
        f"Estimated probability this incident is **Cyber‑related**: "
        f"{proba_cyber*100:0.1f}% (classified as **{label}**)."
    )
    st.caption("This is a simple logistic regression model trained only for demonstration. "
               "For the report you can discuss its limits and how you would improve it.")

with st.expander("Model performance (for your report)"):
    st.text(model_report_text)

st.markdown("---")
st.caption(
    "Dashboard built with Streamlit and Plotly. "
    "Designed to answer ‘what’s happening, where, and how risky is it?’ "
    "for non‑technical and technical audiences."
)
