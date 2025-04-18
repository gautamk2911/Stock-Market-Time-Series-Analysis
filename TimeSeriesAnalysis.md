
# 📈 ARIMA Time Series Forecasting on Apple Stock Prices

## 🧰 **1. Libraries and Setup**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
```

- `numpy`, `pandas`: For numerical and data operations
- `matplotlib.pyplot`: For plotting
- `warnings`: To suppress warning messages

---

## 🧠 **2. Time Series Model Libraries**

```python
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
```

- Includes ARIMA model and statistical tools for time series analysis.

```python
get_ipython().run_line_magic('matplotlib', 'inline')
```

- Ensures plots display inline in a Jupyter notebook.

---

## 📥 **3. Load the Dataset**

```python
df = pd.read_csv('aapl.csv', parse_dates=['Date'])
df.head()
```

- Loads historical Apple stock data.
- Ensures 'Date' is parsed as datetime.

---

## 📊 **4. Exploratory Data Analysis**

```python
print(df.describe())
print(df.dtypes)
```

- Provides statistical summary and datatype info.

---

## 🔢 **5. Extract 'Date' and 'Close' Columns**

```python
df1 = df[['Date', 'Close']]
df_ts = df1.set_index('Date').sort_index()
```

- Sets 'Date' as index to form a time series.

---

## ❓ **6. Handling Missing Values**

```python
df_ts.Close.fillna(method='pad', inplace=True)
```

- Forward fills missing values.

---

## 📉 **7. Plot Time Series**

```python
df_ts.plot()
```

- Visualizes the closing stock price over time.

---

## 🧪 **8. Test for Stationarity (Dickey-Fuller Test)**

```python
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries): ...
```

- A function that performs the Augmented Dickey-Fuller test and prints results.

---

## 📐 **9. Rolling Statistics**

```python
rolmean = ts.rolling(window=365).mean()
rolstd = ts.rolling(window=365).std()
```

- Computes rolling mean and standard deviation to visualize trends and variability.

---

## 🔁 **10. Data Transformations**

- **Log Transformation**:
```python
ts_logScale = np.log(ts)
```

- **Moving Average**:
```python
movingAverage = ts_logScale.rolling(window=12).mean()
```

- **Exponential Weighted Average**:
```python
exponentialDecayWeightedAverage = ts_logScale.ewm(halflife=12).mean()
```

- **Log Difference**:
```python
ts_LogDiffShifting = ts_logScale - ts_logScale.shift()
```

---

## 🧩 **11. Seasonal Decomposition**

```python
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_logScale, freq=30)
```

- Decomposes the time series into trend, seasonal, and residual components.

---

## 🧭 **12. ACF and PACF Plots**

```python
from statsmodels.tsa.stattools import acf, pacf
```

- Helps identify AR and MA terms for the ARIMA model by analyzing lags.

---

## 📈 **13. ARIMA Model Fitting**

```python
model = ARIMA(ts_logScale, order=(1,1,1))
results_ARIMA = model.fit(disp=-1)
```

- Fits the ARIMA model with specified parameters (p=1, d=1, q=1).

---

## 🔮 **14. Predictions and Inversion**

```python
predictions_ARIMA_log = ...
predictions_ARIMA = np.exp(predictions_ARIMA_log)
```

- Reverts the log transformation to get actual forecasted values.
- Plots predictions against original data.

---

## 📅 **15. Forecast Future Values**

```python
results_ARIMA.forecast(14)
```

- Forecasts the next 14 time steps.

---

## 📌 **Key Takeaways**

- Preprocessing and transformation are crucial before modeling.
- Stationarity is essential for time series modeling.
- ARIMA is a powerful method when ACF/PACF analysis is used correctly.
- Log and differencing transformations help stabilize the variance and mean.
- Always invert transformations to interpret predictions meaningfully.

---
