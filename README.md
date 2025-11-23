# Forecasting Property Sale Requisitions in Canton Fribourg, Switzerland

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A machine learning forecasting pipeline to predict monthly property sale requisitions (*réquisitions de vente*) in Canton Fribourg using 30 years of Swiss administrative data (1994-2024). This project develops an end-to-end predictive system combining debt collection statistics with housing market indicators through time series analysis and ensemble learning methods.

## Project Overview

Property sale requisitions represent the terminal stage of Switzerland's debt collection legal framework, serving as quantifiable indicators of regional financial distress. This project addresses the challenge of forecasting these events to support operational planning in cantonal debt enforcement administration.

**Key Features:**
- **30 years of administrative data** from Swiss Federal Statistical Office
- **Three ML models:** ARIMA (baseline), Random Forest, XGBoost
- **Rigorous temporal validation** to divide the entire dataset based on the time (train: 1994-2015, validation: 2016-2020, test: 2021-2024)
- **24-month rolling forecasts** for 2025-2026 with iterative lag feature updates
- **Reproducible pipeline** with automated testing via GitHub Actions

## Technical Report

**[View Complete Technical Report (PDF)](docs/report.pdf)**

## 🚀 Quick Start

### Prerequisites

- **Python 3.10 or higher**
- **Git** (for cloning repository)
- **pip** (Python package manager)

### Installation

#### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Predictive-modeling-property-sale-reqs.git
cd Predictive-modeling-property-sale-reqs
```

#### 2. Create virtual environment

**Windows:**
```powershell
python -m venv myenv
myenv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv myenv
source myenv/bin/activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
---

```

## Automated pipeline

The ML pipeline consists of five independent modules that execute sequentially:

### Step 1: Data Preprocessing
```bash
python src/preprocessing.py
```

**Output:** `data/processed/processed_dataset.csv`
---

### Step 2: Feature Engineering
```bash
python src/feature_engineering.py
```

**Output:** `data/featured/featured_dataset.csv`
---

### Step 3: Model Training
```bash
python src/model.py
```

**Output:** `models/*.pkl` (trained models)
---

### Step 4: Model Evaluation
```bash
python src/evaluation.py
```

**Output:** `results/*.png` and `results/results.csv`
---

### Step 5: Generate Forecasts
```bash
python src/forecast.py
```

**Output:** `results/forecast_2025_2026.csv`

---

## Full Documentation

- **[Technical Report (PDF)](docs/report.pdf)** - Complete methodology and results
- **[Project Proposal](PROPOSAL.md)** - Original project objectives
- **[AI Usage Documentation](AI_usage.md)** - AI tools used during development
- **[Exploratory Analysis](notebooks/exploration.pdf)** - Initial data exploration

---

## Author

**Author:** Loris Bossel<br>
**Institution:** HEC Lausanne  
**Course:** Advanced Programming<br>
**Email:** loris.bossel@unil.ch 

---

**Last Updated:** November 2024
`