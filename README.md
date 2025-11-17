# Predictive Modeling of Property Sale Requisitions

**Advanced Programming 2025 — HEC Lausanne**  
**Author:** Loris Bossel  
**Instructor:** Prof. Simon Scheidegger  
**Course:** Data Science & Advanced Programming (DSAP 2025)  
**Start Date:** 03 November 2025 <br>
**End Date:** 15 December 2025 

---

# Abstract


# Table of Contents

1. [Introduction](#1-introduction)  
2. [Background](#2-background)  
3. [Design & Architecture](#3-design&architecture)  
4. [Implementation](#4-implementation)  
5. [Evaluation](#5-evaluation)  
6. [Discussion](#discussion)  
7. [Conclusion](#conclusion)  


# 1. Introduction

Property sale requisitions (réquisitions de vente) represent the terminal stage of Switzerland's debt collection legal framework, occurring when real estate assets are liquidated to satisfy outstanding creditor obligations. In Canton Fribourg, these proceedings serve as quantifiable indicators of regional financial distress, reflecting underlying dynamics in credit markets and housing stability. Monitoring and forecasting these events provides valuable insights into regional financial stability and housing market tension, supporting decision-making in debt enforcement administration and public policy formulation. Current forecasting approaches in Swiss cantonal administration rely primarily on historical averages and linear extrapolation, methodologies that fail to capture complex temporal patterns and multivariate dependencies. This project addresses this methodological gap by developing a machine learning forecasting pipeline to predict monthly property sale requisitions using 30 years of administrative data (1994-2024) obtained from the Swiss Federal Statistical Office. The central research question is whether historical debt collection proceedings and housing market indicators can be leveraged to forecast property requisitions with sufficient accuracy to support operational planning in cantonal administration. The project pursues three primary objectives: first, develop a predictive machine learning model maintaining rigorous temporal validation to prevent information leakage while achieving strong generalization performance on held-out test data; second, implement a rolling forecast methodology to generate 24-month predictions for 2025-2026, providing actionable intelligence for resource allocation in Canton Fribourg's debt enforcement offices.

# 2. Background

# 3. Design & Architecture

## 2.1 Data Sources
- **Poursuites par mois depuis 1994** — Monthly statistics on debt collection proceedings.  
   [Open data, Swiss FR](https://opendata.swiss/fr/dataset/betreibungen-pro-monat-ab-1994)  
- **Logements vacants depuis 1975** — Annual statistics on vacant dwellings and vacancy rates.  
   [Open data, Swiss FR](https://opendata.swiss/fr/dataset/leerstehende-wohnungen-anzahl-quote-seit-1975)

The vacancy data are replicated monthly and merged with the poursuites dataset by **district** and **year**.

## 2.2 Data Preparation
<!--
- Align datasets temporally and geographically.  
- Handle missing values and ensure time-series consistency.  
- Restrict analysis to 1994–2024. -->

## 2.3 Feature Engineering



## 2.4 Modeling and Validation
- Models: **ARIMA**, **Random Forest Regressor**, **XGBoost**.  
- Temporal split:
  - Train: 1994–2015  
  - Validation: 2016–2020  
  - Test: 2021–2024  
- Evaluation metrics: **MAE**, **RMSE**.

# 4. Implementation

# 5. Evaluation

# 6. Discussion

# 7. Conclusion


---

# References

1. **Swiss Federal Statistical Office (FSO).** (2024). *Poursuites par mois depuis 1994*.  
   [https://opendata.swiss/fr/dataset/betreibungen-pro-monat-ab-1994](https://opendata.swiss/fr/dataset/betreibungen-pro-monat-ab-1994)  
2. **Swiss Federal Statistical Office (FSO).** (2024). *Logements vacants depuis 1975*.  
   [https://opendata.swiss/fr/dataset/leerstehende-wohnungen-anzahl-quote-seit-1975](https://opendata.swiss/fr/dataset/leerstehende-wohnungen-anzahl-quote-seit-1975)  

---

# Appendices