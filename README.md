# ICO Cyber Risk Dashboard

This repository contains a Streamlit dashboard built on the UK Information Commissioner’s Office (ICO) public **Data Security Incident Trends** dataset. The app is intended as a simple, interactive way for non‑technical and technical users to explore patterns in cyber and non‑cyber incidents across sectors, incident types and time.

Live app: https://ico-cyber-dashboard-sakhan.streamlit.app/

---

## Overview

The dashboard:

- Loads ICO data security incident records from `ico_raw.csv`.
- Lets users filter by **Year**, **Sector** and **Incident Category** (Cyber / Non‑Cyber).
- Displays headline KPIs, including:
  - Total number of incidents
  - Percentage of cyber incidents
  - Percentage of high‑impact incidents (1k+ data subjects affected)
- Provides interactive visualisations:
  - Incidents over time
  - Incident category breakdown
  - Sector vs category heatmap
  - Top incident types and their trends
  - Impact distribution by number of data subjects affected
- Includes a simple demo **classification model** (logistic regression) that estimates the probability that a hypothetical incident is cyber‑related, based on selected characteristics.

This version should be treated as a **trial / proof‑of‑concept**. It is suitable for coursework and initial exploration, and can be extended with stronger models, richer drill‑downs and export options.

---

## Dataset

- **Name:** ICO Data Security Incident Trends  
- **Source:** UK Information Commissioner’s Office – public open data  
- **File:** `ico_raw.csv`  
- **Domain:** Information security, data protection and regulatory reporting across multiple UK sectors.

The data includes, among other fields:

- Year and Quarter
- Sector
- Incident Category (Cyber / Non‑Cyber)
- Incident Type
- Data Subject Type
- Data Type
- Number of data subjects affected (banded)
- Time taken to report

---

## Running the app locally

### 1. Clone the repository

```bash
git clone https://github.com/shoaibahmed7659/ico-cyber-dashboard.git
cd ico-cyber-dashboard
