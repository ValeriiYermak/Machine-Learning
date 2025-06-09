import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import category_encoders as ce
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# %%
train_url = 'https://raw.githubusercontent.com/goitacademy/MACHINE-LEARNING-NEO/main/datasets/mod_04_hw_train_data.csv'
valid_url = 'https://raw.githubusercontent.com/goitacademy/MACHINE-LEARNING-NEO/main/datasets/mod_04_hw_valid_data.csv'

train_df = pd.read_csv(train_url)
valid_df = pd.read_csv(valid_url)

# %%
train_data = train_df.copy()
valid_data = valid_df.copy()

# %%
print("\nTrain preview:")
print(train_data.head())

# %%
train_data.info()

# %%
msno.matrix(train_data)

# %%
print()
print('Train_data before removing empty values')
print(train_data.isnull().sum())
print()
print('Valid_data before removing empty values')
print(valid_data.isnull().sum())
print("Train shape:", train_data.shape)
print("Valid shape:", valid_data.shape)

# %%
train_data = train_data.dropna()
valid_data = valid_data.dropna()

print()
print('Train_data after removing empty values')
print(train_data.isnull().sum())
print()
print('Valid_data after removing empty values')
print(valid_data.isnull().sum())
print("Train shape:", train_data.shape)
print("Valid shape:", valid_data.shape)

# %%
target = 'Salary'
num_features = ['Experience', 'Age']
cat_features = ['Qualification', 'University', 'Role', 'Cert']

print("\nSalary description:")
print(train_data['Salary'].describe())

print("\nCategorical variables and number of unique values:")
for col in cat_features:
    print(f"{col}: {train_data[col].nunique()}")
# %%
print("\nTrain columns:", list(train_data.columns))
print("Valid columns:", list(valid_data.columns))

print("\nThe difference between the columns:")
print(set(valid_data.columns) - set(train_data.columns))

# %%
train_data = train_data.drop(columns=["Name", "Phone_Number"])
valid_data = valid_data.drop(columns=["Name", "Phone_Number"])

print()
print("\nTrain columns after removing unimportant columns:", list(train_data.columns))
print("Valid columns after removing unimportant columns:", list(valid_data.columns))

print("\nThe difference between the columns:")
print()
print(set(valid_data.columns) - set(train_data.columns))

# %%
def calculate_age(birthday_str):
    try:
        birthdate = datetime.strptime(birthday_str, "%d/%m/%Y")
        today = datetime.today()
        return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    except Exception as e:
        print(f"Error: {birthday_str} — {e}")
        return np.nan

# %%
train_data["Age"] = train_data["Date_Of_Birth"].apply(calculate_age)
valid_data["Age"] = valid_data["Date_Of_Birth"].apply(calculate_age)

print(train_data['Date_Of_Birth'].head(6))

# %%
train_data = train_data.drop(columns=["Date_Of_Birth"])
valid_data = valid_data.drop(columns=["Date_Of_Birth"])

# %%
print("\nTrain columns:", list(train_data.columns))
print("Valid columns:", list(valid_data.columns))

print("\nThe difference between the columns:")
print(set(valid_data.columns) - set(train_data.columns))

print(train_data['Age'].sort_values().head(8))

# %%
def clean_data(train_data, valid_data):
    train_data['Age'] = pd.to_numeric(train_data['Age'], errors='coerce')
    valid_data['Age'] = pd.to_numeric(valid_data['Age'], errors='coerce')

    train_data = train_data.dropna(subset=['Age'])
    valid_data = valid_data.dropna(subset=['Age'])

    if 'Role' not in train_data.columns or 'Role' not in valid_data.columns:
        raise ValueError("The input DataFrame does not contain a column'Role'")

    train_data = train_data[train_data['Age']>=16]
    valid_data = valid_data[valid_data['Age']>=16]

    train_data = train_data[~((train_data['Age'] ==16) & (train_data['Role'] == 'Mid'))]
    valid_data = valid_data[~((valid_data['Age'] ==16) & (valid_data['Role'] == 'Mid'))]

    train_data = train_data[~((train_data['Age'].isin ([16,17,18])) & (train_data['Role'] == 'Senior'))]
    valid_data = valid_data[~((valid_data['Age'].isin ([16,17,18])) & (valid_data['Role'] == 'Senior'))]

    return train_data.reset_index(drop=True), valid_data.reset_index(drop=True)

train_data, valid_data = clean_data(train_data, valid_data)

print(train_data['Age'].sort_values().head(9))

# %%
target = "Salary"
features = [col for col in train_data.columns if col !=target]
num_features = ['Experience', 'Age']
cat_features = list(set(features) - set(num_features))

# %%
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
     ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor(n_neighbors=5))
    ])
# %%
X_train = train_data[features]
y_train = train_data[target]
model.fit(X_train, y_train)

# %%
X_valid = valid_data[features]
y_valid = valid_data[target]
y_pred = model.predict(X_valid)

# %%
mape = mean_absolute_percentage_error(y_valid, y_pred)
print('First variant')
print(f'Validation MAPE: {mape:.2%}')
print()

# %%
num_imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_train_num = scaler.fit_transform(num_imputer.fit_transform(train_data[num_features]))
X_valid_num = scaler.transform(num_imputer.transform(valid_data[num_features]))

# %%
encoder = ce.TargetEncoder()
X_train_cat = encoder.fit_transform(train_data[cat_features], train_data[target])
X_valid_cat = encoder.transform(valid_data[cat_features])

# %%
X_train = np.hstack([X_train_num, X_train_cat])
X_valid = np.hstack([X_valid_num, X_valid_cat])
y_train = train_data[target]
y_valid = valid_data[target]

# %%
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_valid)
mape = mean_absolute_percentage_error(y_valid, y_pred)
print('Second variant')
print(f"Validation MAPE: {mape:.2%}")
print()

# %%
y_train_log = np.log1p(y_train)
y_valid_true = y_valid.copy()

# %%
model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.8,
    random_state=37)

# %%
model.fit(X_train, np.log1p(y_train))

# %%
y_pred_log = model.predict(X_valid)
y_pred = np.expm1(model.predict(X_valid))

# %%
sns.histplot(train_data['Salary'], bins=50)
plt.title('Distribution Salary before log')
plt.show()

sns.histplot(np.log1p(train_data['Salary']), bins=50)
plt.title('Distribution Salary after log1p')
plt.show()

# %%
mape = mean_absolute_percentage_error(y_valid_true, y_pred)
print('Third variant with log transformation')
print(f"Validation MAPE: {mape:.2%}")
print()

# %%
rf_model = RandomForestRegressor(
    n_estimators=200,       # Quantity of trees
    max_depth=10,           # Maximum depth of tree
    random_state=37,        # Record randomness
    n_jobs=-1               # Parallelization on all cores
)

# %%
rf_model.fit(X_train, y_train_log)

# %%
y_pred_log = rf_model.predict(X_valid)
y_pred = np.expm1(y_pred_log)

# %%
mape_rf = mean_absolute_percentage_error(y_valid, y_pred)
print('Fourth variant Random Forest with log transformation')
print(f"Validation MAPE: {mape_rf:.2%}")
print()

# %%
qualifications = ['Junior', 'Mid', 'Senior']
kf = KFold(n_splits=5, shuffle=True, random_state=37)

all_y_val_true = []
all_y_val_pred = []
print()
print('Fifth variant with cross validation')
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=37)
for qual in qualifications:
    print(f"\n=== Role: {qual} ===")
    q_data = train_data[train_data['Role'] == qual].copy()
    val_q_data = val_data[val_data['Role'] == qual].copy()

    if len(q_data) < 10:
        print("Insufficient data for cross-validation.")
        continue

    X = q_data[features]
    y = np.log1p(q_data[target])

    mape_scores = []

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_valid_enc = encoder.transform(X_valid)

        model = GradientBoostingRegressor(n_estimators=1000, max_depth=6, learning_rate=0.01, random_state=37)
        model.fit(X_train_enc, y_train)

        y_pred_log = model.predict(X_valid_enc)
        y_pred = np.expm1(y_pred_log)

        mape = mean_absolute_percentage_error(np.expm1(y_valid), y_pred)
        mape_scores.append(mape)

    mean_mape = np.mean(mape_scores)
    std_mape = np.std(mape_scores)
    print(f"CV MAPE: {mean_mape * 100:.2f}% ± {std_mape * 100:.2f}%")

    if not val_q_data.empty:
        X_val_final = val_q_data[features]
        y_val_final = val_q_data[target]

        full_X_enc = encoder.fit_transform(X, y)
        model.fit(full_X_enc, y)

        X_val_enc = encoder.transform(X_val_final)
        y_val_pred_log = model.predict(X_val_enc)
        y_val_pred = np.expm1(y_val_pred_log)

        val_mape = mean_absolute_percentage_error(y_val_final, y_val_pred)
        print(f"Validation MAPE: {val_mape * 100:.2f}%")

        # Додаємо для глобального обрахунку
        all_y_val_true.extend(y_val_final)
        all_y_val_pred.extend(y_val_pred)
    else:
        print("There are no matching examples in the validation set..")

if all_y_val_true and all_y_val_pred:
    overall_val_mape = mean_absolute_percentage_error(all_y_val_true, all_y_val_pred)
    print(f"\n=== Total Validation MAPE for all Roles:===")
    print(f'Validation MAPE:{overall_val_mape * 100:.2f}%')
else:
    print("\nInsufficient data to calculate total Validation MAPE.")



