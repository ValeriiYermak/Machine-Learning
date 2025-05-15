# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import zscore
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %% Download data for analise
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()

# %% some information about data
df.head(10)

# %% some more information about data
df.info()

# %%
df_cleaned = df.copy()

# %%
# clean from the anomalies
columns_to_check = ["AveRooms", "AveBedrms", "AveOccup", "Population"]

# %% calculate z-scores
z_scores = df_cleaned[columns_to_check].apply(zscore)

# %% Identify anomalies: where z_score >3 or <-3
outliers_mask = (np.abs(z_scores) > 3).any(axis=1)

# %% remote lines from the anomalies
df_cleaned = df_cleaned[~outliers_mask].reset_index(drop=True)

# %%
corr_matrix = df_cleaned.corr()
print(corr_matrix)

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix of cleaned data")
plt.show

# %% We scale the signs
X_features = df_cleaned.drop(columns=["MedHouseVal"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# %% Count VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X_features.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])
]
print(f"VIF:{vif_data}")

# %% Recalculate VIF without 'Longitude'
X_features_reduced = df_cleaned.drop(columns=["MedHouseVal", "Longitude"])
X_scaled_reduced = scaler.fit_transform(X_features_reduced)

vif_data_reduced = pd.DataFrame()
vif_data_reduced["feature"] = X_features_reduced.columns
vif_data_reduced["VIF"] = [
    variance_inflation_factor(X_scaled_reduced, i)
    for i in range(X_scaled_reduced.shape[1])
]
print("\nVIF after removing 'Longitude':")
print(vif_data_reduced)

# %%
X = df_cleaned.drop(columns=["MedHouseVal"])
y = df_cleaned["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
# normalization of features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# %%
# build the model of linear regression

model_lin = LinearRegression()
model_lin.fit(X_train_scaled, y_train)

# Prediction
y_pred_lin = model_lin.predict(X_test_scaled)

# %% evaluation of the model
r_sq_upd_lin = model_lin.score(X_train_scaled, y_train)
mae_upd_lin = mean_absolute_error(y_test, y_pred_lin)
mape_upd_lin = mean_absolute_percentage_error(y_test, y_pred_lin)
print()
print("Linear Regression")
print(f"R²: {r_sq_upd_lin:.4f} | MAE: {mae_upd_lin:.4f} | MAPE: {mape_upd_lin:.2f}%")

# %%
# Polynomial Regression
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

y_pred_poly = model_poly.predict(X_test_poly)

r2_poly = model_poly.score(X_train_poly, y_train)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
mape_poly = mean_absolute_percentage_error(y_test, y_pred_poly)
print()
print("Polynomial Regression (level 2):")
print(f"R2: {r2_poly:.2f} | MAE: {mae_poly:.2f} | MAPE: {mape_poly:.2f}")

# %% Plot of actual vs predicted values for each model
plt.figure(figsize=(14, 6))

# Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lin, alpha=0.5, label="Prediction vs Real")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
    label="Perfect Prediction",
)
plt.title("Linear Regression: Real vs Predicted")
plt.xlabel("Real Features")
plt.ylabel("Predicted Features")
plt.legend(loc="upper right")

# %%
# Polynomial Regression
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_poly, alpha=0.5, label="Prediction vs Real")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
    label="Perfect Prediction",
)
plt.title("Polynomial Regression: Real vs Predicted")
plt.xlabel("Real Features")
plt.ylabel("Predicted Features")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# %%
# Residual Plot
plt.figure(figsize=(10, 5))

# Linear Regression
plt.subplot(1, 2, 1)
residuals_lin = y_test - y_pred_lin
sns.scatterplot(x=y_pred_lin, y=residuals_lin, alpha=0.5, label="Rating – Real")
plt.hlines(
    0,
    xmin=y_pred_lin.min(),
    xmax=y_pred_lin.max(),
    colors="r",
    linestyles="--",
    label="Zero error",
)
plt.title("Linear Regression: Residual errors")
plt.xlabel("Predicted Features")
plt.ylabel("Residual errors")
plt.legend(loc="upper right")

# %%
# Polynomial Regression
plt.subplot(1, 2, 2)
residuals_poly = y_test - y_pred_poly
sns.scatterplot(x=y_pred_poly, y=residuals_poly, alpha=0.5, label="Rating – Real")
plt.hlines(
    0,
    xmin=y_pred_poly.min(),
    xmax=y_pred_poly.max(),
    colors="r",
    linestyles="--",
    label="Zero error",
)
plt.title("Polynomial Regression: Residual errors")
plt.xlabel("Predicted Features")
plt.ylabel("Residual errors")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# %%
# Comparison plot of R2, MAE, MAPE for both models
metrics = pd.DataFrame(
    {
        "Model": ["Linear Regression", "Polynomial Regression"],
        "R2": [r_sq_upd_lin, r2_poly],
        "MAE": [mae_upd_lin, mae_poly],
        "MAPE": [mape_upd_lin, mape_poly],
    }
)

# Prepare of plot
metrics.set_index("Model", inplace=True)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# R2
metrics["R2"].plot(kind="bar", ax=ax[0], color=["blue", "green"])
ax[0].set_title("R2: Comparison of models")
ax[0].set_ylabel("R2")

# MAE
metrics["MAE"].plot(kind="bar", ax=ax[1], color=["blue", "green"])
ax[1].set_title("MAE: Comparison of models")
ax[1].set_ylabel("MAE")

# MAPE
metrics["MAPE"].plot(kind="bar", ax=ax[2], color=["blue", "green"])
ax[2].set_title("MAPE: Comparison of models")
ax[2].set_ylabel("MAPE")

plt.tight_layout()
plt.show()
