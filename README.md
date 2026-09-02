# Predictive Procurement: Advanced Time Series Forecasting Methods for Inventory Management



> **Optimising procurement via classical, ML, and DL forecasting of Nielsen book sales.**



[![Current Project Status](https://img.shields.io/badge/Status-Actively_Refactoring-orange.svg)](#)



![Image of time series data from the book The Very Hungry Caterpillar with a train/test split and associated forecast and confidence intervals. Forecast generated with Auto-ARIMA with MAPE of 0.164. Residuals center around a mean of zero and make a rough normal distribution. ACF/PACF of residuals shows no significant autocorrelations.](assets/tvhc_auto_arima_plot.png)



## ✦ Overview

- **Problem Statement:** Demand forecasting is complicated by long-term seasonality, trend effects, and anomalous periods like COVID-19 closures, leading to costly stock mismanagement.
- **Objective:** Develop and compare models to interpret past sales data to accurately forecast future sales patterns, enabling proactive restocking decisions.
- **Impact:** Reduces costly inventory shortages and maximizes revenue by allowing independent publishers to reliably adjust stock levels for seasonal spikes.



## ✦ Tech Stack

**Python**, **pandas**, **pmdarima**, **sktime**, **XGBoost**, **TensorFlow/Keras**, **KerasTuner**, **statsmodels**



## ✦ Data

- **Source(s):** Nielsen's BookScan transactional data across UK retailers.
- **Size:** 227,224 rows (resampled to weekly).
- **Notable Characteristics:** Strong annual seasonality, anomalous periods of zero sales due to COVID-19, and non-stationary trends in some titles.



## ✦ Methodology

- **Preprocessing:** Applied an advanced, season-aware STL imputation method to replace missing data during COVID-19 closures to ensure model generalisability and forecasting performance.
- **Feature Engineering:** Resampled sales data weekly, filling missing weeks with zeros to ensure a consistent time series index.
- **Modelling:** Compared Auto-ARIMA, tuned XGBoost, and optimized LSTM networks, combining them into sequential and parallel hybrid models to leverage their individual strengths.
- **Evaluation:** Evaluated using Mean Absolute Error (MAE) for sales value expected error and Mean Absolute Percentage Error (MAPE) for scale-independent comparison.



## ✦ Key Results and Outputs

- Auto-ARIMA provided the best single-architecture baseline (MAE: 1466 for 'The Very Hungry Caterpillar').
- Parallel ARIMA-LSTM hybrid models showed robust performance, optimally weighting forecasts to compensate for individual model errors.



## ✦ Roadmap and Limitations

- **Limitation:** Non-deterministic deep learning models (LSTMs) showed fluctuating performance and sometimes failed to outperform statistical baselines.
- **Future Work:** Apply vector embeddings to title clusters (multiple ISBNs for same book) to analyze sales decay mechanisms and train multiple model iterations for robust ensemble predictions.