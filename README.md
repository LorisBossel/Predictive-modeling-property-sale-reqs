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
2. [Methodology](#2-methodology)  
3. [Results](#3-results)  
4. [Discussion](#4-discussion)  
5. [Conclusion](#5-conclusion)  
6. [References](#references)  


# 1. Introduction

This project aims to forecast the monthly number of *réquisitions de vente* (property sale requisitions) in the **Canton of Fribourg**, Switzerland, an indicator of financial distress within local economies.  <br>

Using open datasets from the **Swiss Federal Statistical Office**, the project develops a predictive model combining **financial and housing indicators** through time-series analysis and machine learning. <br>
The workflow includes data preparation, feature engineering, exploratory analysis, model development, and evaluation using **ARIMA**, **Random Forest**, and **XGBoost** models.

## 1.1 Background and Motivation
*Réquisitions de vente* represent the final stage in debt collection procedures, occurring when properties are sold due to unpaid obligations.  
Monitoring and forecasting these events provides valuable insights into **regional financial stability** and **housing market tension**.  
This project seeks to build a **predictive time-series model** capable of anticipating such trends to support decision-making in debt management and public administration.

## 1.2 Problem Statement
The challenge is to forecast future *réquisitions de vente* based on historical data of **debt collection proceedings (poursuites)** and **vacant dwellings (logements vacants)**.

## 1.3 Objectives
- Collect and clean open data from the Swiss Federal Statistical Office (FSO).  
- Engineer lagged temporal features to capture delayed effects.  
- Build and evaluate predictive models using ARIMA, Random Forest, and XGBoost.  
- Compare their performance using **MAE** and **RMSE**.

# 2. Methodology

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

# 3. Results


# 4. Discussion


# 5. Conclusion


---

# References

1. **Swiss Federal Statistical Office (FSO).** (2024). *Poursuites par mois depuis 1994*.  
   [https://opendata.swiss/fr/dataset/betreibungen-pro-monat-ab-1994](https://opendata.swiss/fr/dataset/betreibungen-pro-monat-ab-1994)  
2. **Swiss Federal Statistical Office (FSO).** (2024). *Logements vacants depuis 1975*.  
   [https://opendata.swiss/fr/dataset/leerstehende-wohnungen-anzahl-quote-seit-1975](https://opendata.swiss/fr/dataset/leerstehende-wohnungen-anzahl-quote-seit-1975)  