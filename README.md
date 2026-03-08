# 👋 Welcome to My Data Science Portfolio

Hi, I'm **Sandra**!

This repository showcases selected data science and machine learning projects developed during my MSc in Data Science and personal work. My focus is on building **end-to-end ML pipelines** with production-ready practices, including model training, evaluation, deployment, and orchestration.

Feel free to explore!

---

## 📊 Projects

### 1. Customer Churn Prediction (End-to-End ML Pipeline)
A **production-ready machine learning pipeline** for predicting customer churn, featuring:
- Data preprocessing and feature engineering
- Model training and evaluation
- Model explainability with SHAP
- Experiment tracking with MLflow
- Workflow orchestration with Airflow
- Containerization with Docker

**Tech Stack:** Python · Pandas · Scikit-learn · Docker · Airflow · MLflow · SQL

---

### 2. Microsoft Malware Prediction
A **binary classification project** to predict whether a Windows machine will be infected by malware based on telemetry data from Windows Defender.

This project tackles a **large-scale dataset** (8.9M rows) from a Kaggle competition, applying a complete supervised learning workflow:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Handling class imbalance
- Model selection and hyperparameter tuning
- Performance evaluation with appropriate metrics
- Model interpretation

The project was developed as part of my **Master's degree in Data Science** and demonstrates:
- Proper ML methodology (EDA-first approach)
- Efficient code practices (functions, list comprehensions)
- Comprehensive model evaluation beyond standard metrics

**Tech Stack:** Python · Pandas · NumPy · Scikit-learn · Matplotlib/Seaborn

> **Dataset:** [Microsoft Malware Prediction - Kaggle](https://www.kaggle.com/c/microsoft-malware-prediction)

---

### 3. Exploratory Analysis of Railway Ticket Prices
An **in-depth exploratory data analysis (EDA)** of railway journeys and ticket prices, with emphasis on understanding data before modeling.

This project focuses on:
- Thorough data cleaning and preprocessing
- Statistical analysis and pattern recognition
- Anomaly detection and outlier analysis
- Route and city-level insights
- Data validation and assumption testing
- Clear, publication-quality visualizations

The work received **top evaluation** for analytical depth, code structure, clarity of reasoning, and data visualization quality.

**Tech Stack:** Python · Pandas · NumPy · Plotly

> **Note:** Dataset provided for educational purposes.

### 4. Sales Forecasting for Capital Raise Decision (Time Series)

A **time series forecasting project** developed as part of my Master's degree in Data Science, with a real business objective: predicting weekly sales for December 2023 to support a capital raise decision for a retail company.

The project compares three forecasting approaches on a dataset of 394K transactions:

- **SARIMAX** with automatic order selection via `auto_arima` and Black Friday as an exogenous variable
- **Prophet** with custom monthly seasonality and UK public holidays
- **XGBoost** with recursive forecasting, lag features, and rolling statistics

Key highlights:
- Segmented modeling (UK vs. Rest of World) based on EDA insights
- Model interpretability with SHAP values
- Temporal cross-validation to validate Prophet robustness
- Confidence intervals on final predictions for business reporting
- XGBoost achieved **11.9% MAPE** at company level, selected as the final model

**Tech Stack:** Python · Pandas · Statsmodels · Prophet · XGBoost · SHAP · Matplotlib/Seaborn

> **Dataset:** UCI Online Retail dataset adapted for educational purposes.

---

## 🛠️ Technical Skills

**Languages & Libraries:** Python · Pandas · NumPy · Scikit-learn · SQL  
**ML Ops & Tools:** Docker · Airflow · MLflow · Git  
**Visualization:** Plotly · Matplotlib · Seaborn  
**Practices:** End-to-End ML Pipelines · Model Explainability · Experiment Tracking

---
## 📫 Get in Touch

Feel free to explore the projects or reach out via [LinkedIn](https://www.linkedin.com/in/your-profile)!

---

*More projects coming soon...*
