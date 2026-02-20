import streamlit as st
import pandas as pd
import plotly.express as px

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
    # Change to your actual file name / sheet name
    df = pd.read_csv("ico_raw.csv")  # or .csv if you export to CSV

    # Expected columns (case‑sensitive):
    # 'BI Reference','Year','Quarter','Data Subject Type','Data Type',
    # 'Decision Taken','Incident Category','Incident Type',
    # 'No. Data Subjects Affected','Sector','Time Taken to Report'

    # Standardise column names for easier coding
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

    # Build a rough Date from Year + Quarter
    quarter_to_month = {"Qtr 1": 2, "Qtr 2": 5, "Qtr 3": 8, "Qtr 4": 11}
    df["Month"] = df["Quarter"].map(quarter_to_month).fillna(1).astype(int)
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" +
                                df["Month"].astype(str) + "-01")

    # Order for impact bands
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
# SIDEBAR FILTERS
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
# HEADER
# -------------------------------------------------------
st.title("ICO Data Security Incidents – Cyber Risk Dashboard")
st.caption("For exploring UK ICO data‑security incidents in a clear, lay‑friendly way.")

# -------------------------------------------------------
# KPIs
# -------------------------------------------------------
total_incidents = len(filtered)
pct_cyber = (filtered["Incident_Category"] == "Cyber").mean() * 100 if total_incidents > 0 else 0.0

high_bands = ["1k to 10k", "10k to 100k", "Over 100k"]
high_impact = filtered[filtered["No_Data_Subjects_Affected"].isin(high_bands)]
pct_high_impact = len(high_impact) / total_incidents * 100 if total_incidents > 0 else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Total incidents", f"{total_incidents:,}")
c2.metric("% Cyber incidents", f"{pct_cyber:0.1f}%")
c3.metric("% High‑impact incidents*", f"{pct_high_impact:0.1f}%")
st.caption("*Here defined as incidents affecting 1k+ data subjects (you can adjust this in the code).")

# -------------------------------------------------------
# ROW 1 – TIME TREND + CATEGORY BREAKDOWN
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
            template="plotly_white"
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
# ROW 2 – SECTOR HEATMAP + INCIDENT TYPE
# -------------------------------------------------------
st.markdown("---")
r2c1, r2c2 = st.columns([1.4, 1])

with r2c1:
    st.subheader("Sector vs category")
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
    st.subheader("Top incident types")
    if len(filtered) > 0:
        type_counts = (
            filtered["Incident_Type"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        type_counts.columns = ["Incident_Type", "Incidents"]
        fig_type = px.bar(
            type_counts,
            y="Incident_Type",
            x="Incidents",
            orientation="h",
            template="plotly_white"
        )
        fig_type.update_layout(
            xaxis_title="Incidents",
            yaxis_title="Incident type",
            height=500
        )
        st.plotly_chart(fig_type, use_container_width=True)
    else:
        st.info("No data for the selected filters.")

# -------------------------------------------------------
# ROW 3 – IMPACT
# -------------------------------------------------------
st.markdown("---")
st.subheader("Impact – number of data subjects affected")

r3c1, r3c2 = st.columns(2)

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
    if len(filtered) > 0:
        pie_counts = (
            filtered["No_Data_Subjects_Affected"]
            .value_counts()
            .reset_index()
        )
        pie_counts.columns = ["Band", "Incidents"]
        fig_pie = px.pie(
            pie_counts,
            names="Band",
            values="Incidents",
            hole=0.4,
            template="plotly_white"
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No data for the selected filters.")

# -------------------------------------------------------
# PREDICTION PANEL PLACEHOLDER
# -------------------------------------------------------
st.markdown("---")
st.header("Incident risk explorer (model to be added)")

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

st.button("Estimate risk (model coming soon)")

st.caption(
    "This section will show probabilities once your classification model "
    "is trained and integrated."
)

st.markdown("---")
st.caption("Built with Streamlit and Plotly for an interactive, lay‑friendly view of ICO incidents.")