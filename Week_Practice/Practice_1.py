import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

# %%
data = pd.read_excel('https://github.com/Limwail/machine-learning-practice/raw/refs/heads/master/datasets/SeoulBikeData.xlsx', engine='openpyxl', 
                     parse_dates=[0])
data.head(10)
data.info()

# %%
data['Date'] = pd.to_datetime(data['Date'])

# %%
sns.boxplot(data, x='Hour', y='Rented Bike Count')

# %%
data_daily = data.groupby('Date').agg({
    'Rented Bike Count': 'sum',
    'Hour': 'last',
    'Temperature': 'mean',
    'Humidity': 'mean',
    'Wind speed': 'mean',
    'Visibility': 'mean',
    'Dew point temperature': 'mean',
    'Solar Radiation': 'mean',
    'Rainfall': 'sum',
    'Snowfall': 'sum',
    'Seasons': 'first',
    'Holiday': 'first',
    'Functioning Day': 'first',}).reset_index().drop('Hour', axis=1)

# %%
ax = sns.lineplot(data, x='Date', y='Rented Bike Count', err_style=None)
ax2 = ax.twinx()
sns.lineplot(data, x='Date', weights=1000, y= "Temperature", ax=ax2, err_style=None, linestyle='--')

# %%

print(data_daily['Holiday'].unique)
data_daily['Holiday'] = data_daily['Holiday'].map({
    'Holiday': 1,
    'No Holiday': 0
    })
data_daily['Functioning Day'] = data_daily['Functioning Day'].map({
    'Yes': 1,
    'No': 0
    })
# %%
season = pd.get_dummies(data_daily['Seasons'], drop_first=True).astype(int)
data_cleaned = pd.concat([data_daily, season], axis=1).drop(columns=['Seasons', 'Date'])

# %%

sns.heatmap(data_cleaned.corr(), center=0, cmap='coolwarm', annot=True)

# %%

X = data_cleaned.drop('Rented Bike Count', axis=1)
y = data_cleaned['Rented Bike Count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# %%
model = LinearRegression().fit(X_train, y_train)
prediction = model.predict(X_test)
prediction = np.maximum(prediction, 0)

r_sq_upd = model.score(X_train, y_train)
mae_upd = mean_absolute_error(y_test, prediction)
mape_upd = mean_absolute_percentage_error(y_test, prediction)
rmse = root_mean_squared_error(y_test, prediction)

print(f'R2:{r_sq_upd:.2f} | MAE: {mae_upd:.2f} | MAPE: {mape_upd:.2f}')

# %%

plt.scatter(x=y_test, y=prediction)
plt.plot([0, 35000], [0, 35000], color='red')

# %%
full_prediction = model.predict(scaler.transform(X))
plt.plot(data_daily['Date'], y, label='True', color='blue')
plt.plot(data_daily['Date'], full_prediction, label='Prediction', color='orange')








