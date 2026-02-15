import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox

rainfall_data = pd.read_csv("Rainfall_data.csv")

print(rainfall_data.head())
print(f"\n")

print(rainfall_data.describe())
print(f"\n")

print(type(rainfall_data))
print(f"\n")

print(rainfall_data.info())
print(rainfall_data.columns)
print(rainfall_data.shape)

rainfall_data["Date"] = pd.to_datetime(rainfall_data[["Year", "Month", "Day"]])
rainfall_data.set_index("Date", inplace=True)
rainfall_data = rainfall_data.drop(columns=["Year", "Month", "Day"])

fig, axes = plt.subplots(2, 2)

rainfall_data["Specific Humidity"].plot(ax=axes[0, 0], title="Specific Humidity")
rainfall_data["Relative Humidity"].plot(ax=axes[1, 0], title="Relative Humidity")
rainfall_data["Temperature"].plot(ax=axes[0, 1], title="Temperature")
rainfall_data["Precipitation"].plot(ax=axes[1,1], title="Precipitation")

plt.show()

Mumbai = rainfall_data["Precipitation"]
Mumbai.index.freq = 'MS'
print(Mumbai.head())

Mumbai.plot(title = "Precipitation")
plt.show()

seasonal_decompose(Mumbai, model = "additive").plot()
plt.show()

plot_acf(Mumbai)
plt.show()

diff_Mumbai = Mumbai.diff().dropna()
diff12_diff_Mumbai = diff_Mumbai.diff(12).dropna()
diff12_Mumbai = Mumbai.diff(12).dropna()

ax1 = plt.subplot2grid((2,2), (0,0))
ax2 = plt.subplot2grid((2,2), (0,1))
ax3 = plt.subplot2grid((2,2), (1,0), colspan=2)

plot_acf(diff_Mumbai, ax=ax1)
plot_acf(diff12_diff_Mumbai, ax=ax2)
plot_acf(diff12_Mumbai, ax=ax3)

plt.show()

fig, axes = plt.subplots(1, 2)

plot_acf(diff12_Mumbai, lags = 50, ax=axes[0])
plot_pacf(diff12_Mumbai, lags = 50, ax=axes[1])

plt.show()

fit_1 = ARIMA(Mumbai, order = (0,0,0), seasonal_order=(0,1,1,12)).fit()
fit_2 = ARIMA(Mumbai, order = (0,0,0), seasonal_order=(1,1,1,12)).fit()
fit_3 = ARIMA(Mumbai, order = (1,0,0), seasonal_order=(1,1,1,12)).fit()

print(fit_1.aicc)
print(fit_2.aicc)
print(fit_3.aicc)

print(shapiro(fit_1.resid))
print(shapiro(fit_2.resid))
print(shapiro(fit_3.resid))

print(acorr_ljungbox(fit_1.resid, lags=12))
print(acorr_ljungbox(fit_2.resid, lags=12))
print(acorr_ljungbox(fit_3.resid, lags=12))

fig, axes = plt.subplots(3, 2)

plot_acf(fit_1.resid, ax=axes[0, 0])
plot_acf(fit_2.resid, ax=axes[1, 0])
plot_acf(fit_3.resid, ax=axes[2, 0])

axes[0, 1].hist(fit_1.resid, edgecolor='black', bins=50)
axes[1, 1].hist(fit_2.resid, edgecolor='black', bins=50)
axes[2, 1].hist(fit_3.resid, edgecolor='black', bins=50)

plt.tight_layout()

plt.show()
