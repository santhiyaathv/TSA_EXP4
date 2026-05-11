# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 11.05.2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000
data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.
4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

# Ignore warnings for clean output
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv('Video_Games_Sales.csv')

# Clean column names
data.columns = data.columns.str.strip()

# Convert year column to datetime
data['Year_of_Release'] = pd.to_datetime(data['Year_of_Release'], format='%Y', errors='coerce')

# Remove missing values
data = data.dropna(subset=['Year_of_Release', 'Global_Sales'])

# Set index
data.set_index('Year_of_Release', inplace=True)

# Aggregate yearly sales
data = data.groupby(data.index).sum(numeric_only=True)

# Set frequency
data = data.asfreq('YE')

# Fill missing values
data['Global_Sales'] = data['Global_Sales'].ffill()

# Select time series
X = data['Global_Sales']

# Set figure size
plt.rcParams['figure.figsize'] = [12, 6]

# Plot original data
plt.plot(X)
plt.title('Original Data (Global Sales)')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.show()

# ACF and PACF
plt.subplot(2, 1, 1)
plot_acf(X, lags=min(10, len(X)//2), ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=min(10, len(X)//2), ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

# ARMA(1,1)

N = 200

arma11_model = ARIMA(X, order=(1, 1, 1)).fit()

phi1 = arma11_model.params.get('ar.L1', 0)
theta1 = arma11_model.params.get('ma.L1', 0)

ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 200])
plt.show()

plot_acf(ARMA_1)
plt.title("ACF - ARMA(1,1)")
plt.show()

plot_pacf(ARMA_1)
plt.title("PACF - ARMA(1,1)")
plt.show()

# ARMA(2,2)

arma22_model = ARIMA(X, order=(2, 1, 2)).fit()

phi1 = arma22_model.params.get('ar.L1', 0)
phi2 = arma22_model.params.get('ar.L2', 0)
theta1 = arma22_model.params.get('ma.L1', 0)
theta2 = arma22_model.params.get('ma.L2', 0)

ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=300)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 300])
plt.show()

plot_acf(ARMA_2)
plt.title("ACF - ARMA(2,2)")
plt.show()

plot_pacf(ARMA_2)
plt.title("PACF - ARMA(2,2)")
plt.show()

```
### OUTPUT:

#### SIMULATED ARMA(1,1) PROCESS:
<img width="1027" height="532" alt="image" src="https://github.com/user-attachments/assets/035936fc-f032-479e-8a66-12ca014c6971" />

#### Partial Autocorrelation
<img width="1017" height="524" alt="image" src="https://github.com/user-attachments/assets/bdb7c449-a061-4c01-80fb-5f1322b2079c" />

#### Autocorrelation
<img width="1018" height="537" alt="image" src="https://github.com/user-attachments/assets/89bdd649-240e-404d-9684-4eb01e8b273d" />

#### SIMULATED ARMA(2,2) PROCESS:
<img width="1011" height="533" alt="image" src="https://github.com/user-attachments/assets/3cc6acf9-1121-4cdd-b492-e129dd7acbfc" />

#### Partial Autocorrelation
<img width="1028" height="522" alt="image" src="https://github.com/user-attachments/assets/191997b3-d36a-44e1-9356-85958c0f7af0" />


#### Autocorrelation
<img width="1012" height="535" alt="image" src="https://github.com/user-attachments/assets/9b9cb2c6-38f5-41a5-a988-28dfe3fe0f34" />

### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
