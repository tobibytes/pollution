# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



# %%
stations_string = """,STATION 1 EKSUTH ADO,Unnamed: 1,PM2.5,Unnamed: 3,Unnamed: 4,Unnamed: 5,Unnamed: 6,Unnamed: 7,STATION 2 FMC IDO,Unnamed: 9,PM2.5.1,Unnamed: 11,SO2,Unnamed: 13,Unnamed: 14,STATION 3 GENERAL HOSPITAL IKERE,Unnamed: 16,PM 2.5,Unnamed: 18,SO2.1,Unnamed: 20,STATION4,MOBA LGA,Unnamed: 23,PM 2.5.1,Unnamed: 25,SO2.2,Unnamed: 27,STATION5,UNIMEDTH ONDO,Unnamed: 30,PM2.5.2,Unnamed: 32,SO2.3,Unnamed: 34,Unnamed: 35,STATION 6,FMC OWO,Unnamed: 38,PM2.5.3,Unnamed: 40,SO2.4,Unnamed: 42,Unnamed: 43,STATION 7,IDANRE,PM 2.5.2,Unnamed: 47,SO2.5,Unnamed: 49,STATION8,IJU,Unnamed: 52,PM2.5.4,Unnamed: 54,SO2.6,Unnamed: 56,Unnamed: 57,STATION 9,ILE OLUJI,Unnamed: 60,PM2.5.5,Unnamed: 62,SO2.7,Unnamed: 64,Unnamed: 65,STATION 10,ORE GEN HOSP,Unnamed: 68,PM2.5.6,Unnamed: 70,SO2.8,Unnamed: 72,Unnamed: 73,STATION11,OAUTHC IFE,Unnamed: 76,PM2.5.7,SO2.9,Unnamed: 79,Unnamed: 80,Unnamed: 81,STATION 12 SDA,Unnamed: 83,PM 2.5.3,Unnamed: 85,SO2.10,Unnamed: 87"""
stations = []
for i in stations_string.split(','):
    if 'STATION' in i:
        stations.append(i)


# %%
# print(stations)

# # %%
# columns = ['DATE','TOTAL','TIME','MEAN','SO2 TIME','MEAN']
# for name in stations:
#     with open(f'./data/{name.replace(' ', '_')}.csv', 'w') as f:
#         f.write(','.join(columns) + '\n')

# # %%
# with open('PETER PYTHON (2) (2).csv', 'r') as file:
#     data = file.read()
# new_data = data.split('\n')[:61]

# for line in range(len(new_data)):
#     each_rows = new_data[line].split('|')
    
#     for row, row_data in enumerate(each_rows, start=1):
#         if row == 1:
#             row_data = ','.join(row_data.split(',')[1:])
#         else:
#             row_data = ','.join(row_data.split(','))
#         # print(stations, row)
#         with open(f'./data/{stations[row-1].replace(' ', '_')}.csv', 'a') as f:
#             f.writelines(row_data + '\n')

# %%
df = pd.read_csv('./data/STATION_1_EKSUTH_ADO.csv')
df.head()

# %%
# formatting
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

# %%
df[['MEAN', 'MEAN.1']].plot(title='Monthly Pollution Trends')
plt.ylabel('Concentration')
plt.show()

# %%
df['month'] = df.index.month
df['year'] = df.index.year


X = df[['TOTAL', 'month', 'year']]
y = df['MEAN']

model = LinearRegression().fit(X, y)
predicted = model.predict(X)


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

asthma_df = pd.read_excel('./POLLUTANTS ANALYSIS (1)\POLLUTANTS ANALYSIS/ALL VARIABLES.xlsx')
asthma_df = asthma_df.dropna()
asthma_df.columns = asthma_df.columns.str.strip().str.replace(' ', '_')

asthma_df = asthma_df[asthma_df['ASTHMA'].notnull()]

feature_cols = [
    'PM2.5', 'SO2',
    'FEMALES', 'MALES',
    '1-20_YRS', '21-40_YRS', '41-60_YRS', '61-80_YRS', '81-100_YRS',
    'YEAR'
]

target_col = 'ASTHMA'

X = asthma_df[feature_cols]
y = asthma_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print(coef_df.sort_values(by='Coefficient', ascending=False))

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Asthma Cases")
plt.ylabel("Predicted Asthma Cases")
plt.title("Actual vs Predicted Asthma Cases")
plt.grid(True)
plt.show()


# %%



# %% [markdown]
# # Time Series Model

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("./data/STATION_1_EKSUTH_ADO.csv")

# Parse the date and set it as index
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

# Rename for clarity
df.rename(columns={'MEAN': 'PM2.5'}, inplace=True)

# Plot the trend
df['PM2.5'].plot(figsize=(10, 4), title='PM2.5 Concentration Over Time', ylabel='PM2.5')
plt.grid(True)
plt.show()


# %%
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(df['PM2.5'], order=(1,1,1))  # (p,d,q) can be tuned
model_fit = model.fit()

# Forecast the next 6 months
forecast = model_fit.forecast(steps=6)

# Show forecast
print("Next 6 months forecast:")
print(forecast)

# Plot forecast
plt.figure(figsize=(10,4))
plt.plot(df['PM2.5'], label='Observed')
plt.plot(forecast.index, forecast.values, label='Forecast', linestyle='--')
plt.legend()
plt.title('PM2.5 Forecast')
plt.grid(True)
plt.show()


# %%
from prophet import Prophet

# Prepare dataframe
prophet_df = df.reset_index()[['DATE', 'PM2.5']]
prophet_df.columns = ['ds', 'y']

# Fit the model
model = Prophet()
model.fit(prophet_df)

# Create future dataframe (24 months)
future = model.make_future_dataframe(periods=24, freq='MS')
forecast = model.predict(future)

# Plot
model.plot(forecast)
plt.title("Prophet Forecast - PM2.5")
plt.grid(True)
plt.show()


# %%
df['PM2.5'].rolling(window=6).mean().plot(figsize=(10,4), title="6-Month Rolling Average - PM2.5")
plt.grid(True)
plt.show()

# %% [markdown]
# # Asthma Predictive Modelling

# %%
asthma_df = pd.read_excel('./asthma/ALL VARIABLES.xlsx')
asthma_df = asthma_df.drop(['PM2.5', 'SO2'], axis=1)
age_cols = ['1-20 YRS', '21-40 YRS', '41-60 YRS', '61-80 YRS', '81-100 YRS']
asthma_df[age_cols] = asthma_df[age_cols].fillna(asthma_df[age_cols].median())
# or .median() depending on skew

# %%
all_yearly_pollution = []

for name in stations:
    cur_station = pd.read_csv(f'./data/{name.replace(" ", "_")}.csv')
    cur_station.columns = ['DATE', 'TOTAL', 'TIME', 'PM2.5', 'SO2_TIME', 'SO2', 'Location']
    cur_station['DATE'] = pd.to_datetime(cur_station['DATE'])
    cur_station['YEAR'] = cur_station['DATE'].dt.year

    # Ensure numeric types
    cur_station['PM2.5'] = pd.to_numeric(cur_station['PM2.5'], errors='coerce')
    cur_station['SO2'] = pd.to_numeric(cur_station['SO2'], errors='coerce')

    # Group by YEAR
    yearly_pollution = cur_station.groupby('YEAR')[['PM2.5', 'SO2']].mean().reset_index()
    yearly_pollution['Location'] = cur_station['Location'].iloc[0]
    all_yearly_pollution.append(yearly_pollution)


# Combine everything
final_df = pd.concat(all_yearly_pollution, ignore_index=True)


# %%
merged_df = pd.merge(asthma_df, final_df, on=['YEAR', 'Location'], how='left')

# %%
# merged_df = merged_df[merged_df['Location'] == 'EKSUTH']
merged_df.head()


# %%
# Only use numeric features
features = ['PM2.5', 'SO2', 'YEAR']

target = 'ASTHMA'

X = merged_df[features]
y = merged_df[target]
# X = asthma_df[features]
# y = asthma_df[target]


# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict
y_pred = linear_model.predict(X_test)

# Evaluate
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))


# %%
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Asthma Cases")
plt.ylabel("Predicted Asthma Cases")
plt.title("Actual vs Predicted Asthma")
plt.grid(True)
plt.show()


# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Reuse your cleaned and merged data
features = ['PM2.5', 'SO2']

target = 'ASTHMA'
X = merged_df[features]
y = merged_df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize and fit Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"🌲 Random Forest R²: {r2:.3f}")
print(f"🌲 Random Forest RMSE: {rmse:.2f}")


# %%
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Asthma Cases")
plt.ylabel("Predicted Asthma Cases")
plt.title("Actual vs Predicted Asthma")
plt.grid(True)
plt.show()


# %%
from xgboost import XGBRegressor

regressor_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
regressor_model.fit(X_train, y_train)

y_pred = regressor_model.predict(X_test)

print("XGBoost R²:", r2_score(y_test, y_pred))
print("XGBoost RMSE:", root_mean_squared_error(y_test, y_pred, ))


# %%
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Asthma Cases")
plt.ylabel("Predicted Asthma Cases")
plt.title("Actual vs Predicted Asthma")
plt.grid(True)
plt.show()


# %%
def predict_asthma(pm25, so2, model=rf):
    input_data = pd.DataFrame([{
        'PM2.5': pm25,
        'SO2': so2,
    }])
    
    prediction = model.predict(input_data)[0]
    return round(prediction)


# %%
estimated = predict_asthma(
    pm25=59.96,
    so2=0.59,
    model=rf
)

print(f"🫁 Estimated asthma cases: {estimated}")
