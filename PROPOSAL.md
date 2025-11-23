# Project Proposal - Capstone Data Science
## Predictive Modeling of Property Sale Requisitions in Canton Fribourg

**Student:** Loris Bossel 
**Instructors:** Prof. Simon Scheidegger
**Date:** October 2024
---

This project aims to build a time-series model to predict the number of property sale requisitions (Réquisitions de vente) in the Canton of Fribourg.

The goal is to anticipate financial distress trends to support proactive resource planning in debt collection services.

Two open datasets from the Swiss Federal Statistical Office (FSO) will be used:
- **Poursuites par mois depuis 1994**: monthly data on debt collection proceedings — a proxy for financial distress.  
  https://opendata.swiss/fr/dataset/betreibungen-pro-monat-ab-1994
- **Logements vacants depuis 1975**: annual data on vacant dwellings, reflecting real estate dynamics.  
  https://opendata.swiss/fr/dataset/leerstehende-wohnungen-anzahl-quote-seit-1975

The vacancy data will be replicated monthly and merged with the poursuites dataset by district and year.

## 1. Objective

Following academic feedback, the project focuses only on prediction, not causal inference.

**Research Question:** Can machine learning models accurately forecast monthly property sale requisitions in Canton Fribourg 1-6 months ahead, and which modeling approach (classical time series vs. modern machine learning) provides superior predictive performance?

Specifically:
1. What forecast accuracy is achievable for different horizons (1, 3, 6 months)?
2. How do ARIMA, Random Forest, and XGBoost compare in capturing temporal patterns?
3. Which predictors (lags, seasonality, vacancy rates) drive predictions most strongly?

The aim is to forecast monthly réquisitions de vente using financial and housing indicators.

Data will be split for temporal validation:
- Train: 1994–2015
- Validation: 2016–2020
- Test: 2021–2024

Three models will be compared: ARIMA, Random Forest, and XGBoost.

## 2. Methodology

1. **Data preparation**: Load and merge datasets by district and year; clean, align, and interpolate missing values; handle French character encoding.

2. **Feature engineering**: Create lagged predictors (3–6 months) using pandas groupby and shift operations within districts to prevent information leakage; create time-based variables including cyclical month encoding (sine/cosine), normalized year trend, and seasonal indicators.

3. **EDA**: Analyze trends, correlations, and lags to understand temporal patterns and validate feature choices.

4. **Modeling**: Train three approaches with distinct strengths:
   - ARIMA: Classical baseline capturing linear temporal dependencies
   - Random Forest: Ensemble method for non-linear relationships and feature interactions
   - XGBoost: Gradient boosting with regularization for superior performance
   
   Compare models using MAE, RMSE, MAPE, and R² on validation set for model selection.

5. **Validation**: Evaluate generalization on unseen test years (2021-2024) to ensure realistic performance estimates and assess robustness across different economic conditions.

## 3. Expected Outcome

A concise, interpretable forecasting model capturing the temporal relationship between financial stress and property sales, and a comparison of classical and ML-based approaches for economic time-series prediction.

The project will deliver actionable monthly forecasts for 2025-2026 to support Canton Fribourg's administrative resource planning.

---

**Loris Bossel, Student 22-211-890**