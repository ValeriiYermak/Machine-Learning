import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
# %%
df = pd.read_csv('C:/Projects/MasterSc/ML/HW/HW_2/Rain in Australia/weatherAUS.csv')
data = df.copy()

# %%
data.info()

# %%
data.describe()

# %%
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

data = data.drop(columns=['Date'])

# %%
print()
print(data.isnull().sum().sort_values(ascending=False))

# %%
missing = data.isnull().sum()
missing_percent = (missing / len(df)) * 100
print()
print(missing_percent[missing_percent > 0].sort_values(ascending=False))

# %%
data = data.drop(columns=['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'Location'])

# %%
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

categorical_cols.remove('RainTomorrow')

# %%
numerical_cols.append('Year')
categorical_cols.append('Month')

features = numerical_cols + categorical_cols

# %%
max_year = data['Year'].max()
data_train = data[data['Year']<max_year]
data_test = data[data['Year'] == max_year]

# %%
X_train = data_train[features]
X_test = data_test[features]
y_train = data_train['RainTomorrow']
y_test = data_test['RainTomorrow']

# %%
train_mask = y_train.notnull()
X_train = X_train[train_mask]
y_train = y_train[train_mask]

test_mask = y_test.notnull()
X_test = X_test[test_mask]
y_test = y_test[test_mask]

# %%
print()
print('Пропуски у y_train:', y_train.isnull().sum())
print('Пропуски у y_test:', y_test.isnull().sum())

# %%
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear'))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print()
print('First step')
print(classification_report(y_test, y_pred))

# %%
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols)
])

pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=37)),  # Додаємо SMOTE після препроцесінгу
    ('classifier', LogisticRegression(solver='liblinear', random_state=37))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print()
print('Second step')
print(classification_report(y_test, y_pred))

