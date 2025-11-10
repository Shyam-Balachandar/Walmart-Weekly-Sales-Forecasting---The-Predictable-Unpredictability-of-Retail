# %%
import pandas as pd

# Load dataset
df = pd.read_csv(r"D:\BA Notes\Projects\Walmart Sales Forecast\train\train.csv")

# Preview data
df.head()

# %%
# Convert date
df['Date'] = pd.to_datetime(df['Date'])

# Aggregate total sales by week
weekly_sales = (
    df.groupby('Date')['Weekly_Sales']
    .sum()
    .reset_index()
    .rename(columns={'Weekly_Sales':'total_sales'})
)

print(weekly_sales)

# %%
# Sort by date
weekly_sales = weekly_sales.sort_values('Date')

print(weekly_sales.head())

# %%
# Check frequency
weekly_sales = weekly_sales.set_index('Date').asfreq('W-FRI')  # Each Friday
weekly_sales['total_sales'] = weekly_sales['total_sales'].fillna(method='ffill')

# %%
# This code removes extreme high outliers in total_sales by:
# Calculating the IQR range,
# Finding an upper threshold (Q3 + 1.5 × IQR), and
# Replacing any sales above that limit with the threshold value.
# Outlier capping
q1 = weekly_sales['total_sales'].quantile(0.25)
q3 = weekly_sales['total_sales'].quantile(0.75)
iqr = q3 - q1
upper = q3 + 1.5 * iqr
weekly_sales['total_sales'] = weekly_sales['total_sales'].clip(upper=upper)

# %%
# Exploratory Data Analysis (EDA)
# This code creates a line chart showing how Walmart’s total weekly sales 
# have changed over time — helping you visualize trends, patterns, or seasonality in the data.

import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
plt.plot(weekly_sales.index, weekly_sales['total_sales'])
plt.title("Weekly Walmart Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# %%
# Rolling mean:
# This code calculates an 8-week rolling average to smooth your weekly sales data and plots both the actual and smoothed trends.
# It helps you easily identify the underlying pattern in Walmart’s sales over time without the noise of weekly fluctuations.
# Rolling mean:
weekly_sales['rolling_8'] = weekly_sales['total_sales'].rolling(window=8).mean()
plt.figure(figsize=(12,5))
plt.plot(weekly_sales['total_sales'], label='Actual')
plt.plot(weekly_sales['rolling_8'], label='8-week Rolling Avg')
plt.title("8-week rolling average")
plt.legend()
plt.show()

# %%
# Seasonal Decomposition:
# This code decomposes your weekly sales data into its trend, seasonal, and random components using an additive model with a yearly (52-week) cycle.
# The plot helps you clearly see long-term sales trends, seasonal shopping patterns, and unpredictable fluctuations in Walmart’s weekly sales.

from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(weekly_sales['total_sales'], model='additive', period=52)
decomp.plot()
plt.gcf().autofmt_xdate()   # Automatically format and rotate dates
plt.tight_layout()           # Adjust spacing so labels fit well
plt.show()

# %%
# Check Stationarity:
# This code performs the Augmented Dickey-Fuller (ADF) test to check if your weekly sales data is stationary.
# If the p-value < 0.05, your data is stationary (ready for forecasting).
# If not, you’ll need to make it stationary — usually by differencing or removing trend/seasonality before building models like ARIMA.

from statsmodels.tsa.stattools import adfuller
result = adfuller(weekly_sales['total_sales'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# %%
# If p-value > 0.05 → not stationary → apply differencing.
# This code performs first-order differencing on total_sales to remove trends and make the data stationary — a key step before applying forecasting models like ARIMA.
# The new column 'diff_1' holds the week-to-week change in sales.

# weekly_sales['diff_1'] = weekly_sales['total_sales'].diff().dropna()

# %%
# Train-Test Split

# This code prepares your data for forecasting by:
# Copying only the total_sales column, and
# Splitting it into:
# Training set: all weeks except the last 12
# Testing set: the most recent 12 weeks.

# This allows you to train the model on past data and evaluate its accuracy on unseen future weeks.

data = weekly_sales[['total_sales']].copy()
train = data.iloc[:-12]
test = data.iloc[-12:]

# %%
# SARIMA Model

# This code builds and trains a SARIMA (Seasonal ARIMA) model on your weekly sales data.
# It uses both non-seasonal (1,1,1) and seasonal (1,1,1,52) parameters to capture short-term trends and yearly seasonality.
# Finally, it prints a summary of the model’s performance and parameter estimates.

import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm

# SARIMA (seasonal period = 52 weeks)
sarima_model = sm.tsa.SARIMAX(
    train['total_sales'],
    order=(1,1,1),
    seasonal_order=(1,1,1,52),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit()

print(sarima_model.summary())

# %%
# Forecast:

# This code uses the trained SARIMA model to forecast future weekly sales (for the test period), extracts predicted values and confidence intervals, and plots them along with the actual data.
# The result lets you compare model predictions vs. real sales and visualize the forecast uncertainty.

pred = sarima_model.get_forecast(steps=len(test))
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int()

plt.figure(figsize=(12,5))
plt.plot(train.index, train['total_sales'], label='Train')
plt.plot(test.index, test['total_sales'], label='Test', color='black')
plt.plot(pred_mean.index, pred_mean, label='SARIMA Forecast')
plt.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], color='gray', alpha=0.2)
plt.legend()
plt.show()

# %%
# Prophet Model

# This code builds a Prophet model to forecast Walmart’s weekly sales.
# It:
# Prepares the data in Prophet’s required format (ds, y),
# Trains the model on historical data,
# Forecasts the next 12 weeks, and
# Visualizes both the overall forecast and its trend/seasonal components.

from prophet import Prophet

prophet_df = weekly_sales.reset_index()[['Date', 'total_sales']]
prophet_df.columns = ['ds', 'y']

prophet_train = prophet_df.iloc[:-12]
prophet_test  = prophet_df.iloc[-12:]

m = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False
)
m.fit(prophet_train)

future = m.make_future_dataframe(periods=12, freq='W-FRI')
forecast = m.predict(future)

m.plot(forecast)
plt.title("Prophet Forecast")
plt.show()

m.plot_components(forecast)
plt.show()

# %%
# Model Evaluation

# This code evaluates and compares SARIMA and Prophet models using three key metrics:
# MAE: average absolute error
# RMSE: root mean squared error
# MAPE: average percentage error

# It then displays both models’ results in a single table, helping you decide which forecast is more accurate and reliable.

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# SARIMA
y_true = test['total_sales']
y_pred_sarima = pred_mean
mae_sarima = mean_absolute_error(y_true, y_pred_sarima)
rmse_sarima = np.sqrt(mean_squared_error(y_true, y_pred_sarima))
mape_sarima = np.mean(np.abs((y_true - y_pred_sarima) / y_true)) * 100

# Prophet
forecast_tail = forecast.set_index('ds').iloc[-12:]['yhat']
mae_prophet = mean_absolute_error(y_true, forecast_tail)
rmse_prophet = np.sqrt(mean_squared_error(y_true, forecast_tail))
mape_prophet = np.mean(np.abs((y_true - forecast_tail) / y_true)) * 100

comparison = pd.DataFrame({
    'Model': ['SARIMA', 'Prophet'],
    'MAE': [mae_sarima, mae_prophet],
    'RMSE': [rmse_sarima, rmse_prophet],
    'MAPE': [mape_sarima, mape_prophet]
})
print(comparison)

# %%
