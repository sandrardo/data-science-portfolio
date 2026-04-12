# Sales Forecasting with Time Series Models

**Course assignment - Time Series module | MSc in Data Science, Nuclio Digital School**

End-to-end forecasting pipeline applied to a fictional e-commerce retailer (TodoVentas S.A.) with presence in 10 European countries. The goal is to generate weekly sales forecasts for December 2023 to support a capital increase decision.

---

## Business context

TodoVentas S.A. records daily sales transactions across 10 countries. The UK dominates total revenue (roughly 8x higher average weekly sales than the Rest of the World), and shows clear seasonality around Black Friday and Christmas. The Rest of the World segment is more irregular with no identifiable seasonal pattern.

The finance team needs a December 2023 forecast broken down by segment (UK / Rest of the World) to present to shareholders.

---

## Dataset

`retail_todo_ventas.csv` - daily retail transactions with columns:
- `Date` - transaction date
- `Country` - country of sale
- `TotalSales` - gross sales in GBP

The raw data covers December 2022 to December 2023. After filtering out returns and zero-sales days, the data is aggregated into weekly series using `W-MON` frequency.

> The dataset is not included in this repository.

---

## Methodology

The analysis is split into two independent series that are modelled separately and then summed:

**UK series** - strong seasonality, high sales volume  
**Rest of the World series** - irregular pattern, lower volume

### Pipeline overview

```
Raw CSV
  -> EDA + descriptive analysis
      -> Stationarity tests (ADF)
      -> ACF / PACF analysis
          -> Train / test split (last 8 weeks as holdout)
              -> Three models trained and evaluated
                  -> Best model selected per segment (MAPE criterion)
                      -> Final forecast for December 2023
```

### Models

| Model | Description |
|---|---|
| **SARIMAX** | Classical statistical model. Orders selected automatically via `auto_arima`. Black Friday included as exogenous variable. |
| **Prophet** | Meta's forecasting library. Configured with custom seasonality and `growth='flat'` to avoid unrealistic extrapolation on short history. |
| **XGBoost** | Gradient boosting with temporal feature engineering (lags 1, 2, 4, 13 weeks; rolling mean; calendar features). Iterative one-step-ahead prediction for the test period. |

### Evaluation metrics

MAE, RMSE and MAPE calculated on the holdout test set (8 weeks). Model selection based on MAPE, which is scale-independent and allows fair comparison between the two segments.

### Results summary

**UK segment**

| Model | MAPE |
|---|---|
| XGBoost | 15.6% |
| SARIMAX | 46.9% |
| Prophet | 107.4% |

**Rest of the World segment**

| Model | MAPE |
|---|---|
| XGBoost | 24.7% |
| SARIMAX | 33.4% |
| Prophet | 82.8% |

XGBoost wins in both segments. SARIMAX confidence intervals are also computed for the final forecast to quantify uncertainty for the business audience.

### Interpretability

SHAP values are computed for both XGBoost models. Key findings:
- In **UK**, `semana` (week of year) is by far the most influential feature, with high-numbered weeks (Oct-Dec) pushing predictions upward. `lag_1` and `black_friday` also contribute significantly.
- In **Rest of the World**, `lag_13` (one quarter back) ranks above `lag_1`, suggesting the segment follows a slower, quarterly-aligned rhythm rather than the short-term momentum seen in UK.

---

## Repository structure

```
.
+-- SeriesTemporales_Sandra_Rodriguez.ipynb   # Main analysis notebook
+-- retail_todo_ventas.csv                    # Source data (not included)
+-- README.md
```

---

## Stack

- **Python 3.x**
- `pandas`, `numpy` - data manipulation
- `matplotlib`, `seaborn`, `plotly` - visualisation
- `statsmodels` - SARIMAX, ADF test, ACF/PACF
- `pmdarima` - auto_arima for automatic order selection
- `prophet` - Meta's forecasting model
- `xgboost` - gradient boosting regressor
- `shap` - model interpretability
- `sklearn` - evaluation metrics

---

## How to run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly statsmodels pmdarima prophet xgboost shap scikit-learn
   ```
3. Place `retail_todo_ventas.csv` in the project root
4. Run `SeriesTemporales_Sandra_Rodriguez.ipynb` top to bottom

---

## Author

Sandra Rodríguez Domínguez  
MSc in Data Science and AI - Nuclio Digital School / EUNEIZ
