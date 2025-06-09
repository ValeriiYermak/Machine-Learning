import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import urllib.request
import missingno as msno
import joblib
joblib.parallel_backend('loky', n_jobs=-1)
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

# %%
url = 'https://raw.githubusercontent.com/goitacademy/MACHINE-LEARNING-NEO/main/datasets/mod_05_topic_10_various_data.pkl'
with urllib.request.urlopen(url) as fl:
    datasets = pickle.load(fl)
    
# %%
autos = datasets['autos']

# %%
auto = autos.copy()

# %%
print(type(auto))
if isinstance(auto, dict):
    for k in auto:
        print(f'{k}: {type(auto[k])}')
    autos = auto['Autos']
else:
    autos = auto
    
# %%
print("\nDataset 'autos' preview:")
print(autos.head(10))
print()

# %%
print()
print(autos.info())

# %%
msno.matrix(autos)

# %%
print()
print(autos.isnull().sum())

# %% 
# Automatic identification of categorical features
categorical = autos.select_dtypes(include=['object', 'category']).columns.tolist()

# Additionally - discrete numerical features
discrete_numerical = [col for col in autos.select_dtypes(include=['int64','float64']).columns
                      if autos[col].nunique() < 20 and col != 'price']

print('Categorical features:', categorical)
print('Discrete numerical features:', discrete_numerical)

# %% 
# Label Encoding of categorical/discrete features
autos_encoded = autos.copy()

le_dict = {}
for col in categorical:
    le = LabelEncoder()
    autos_encoded[col] = le.fit_transform(autos_encoded[col].astype(str))
    le_dict[col] = le
print()
print(autos_encoded.head(10))

# %% 
# Target variable
y = autos_encoded['price']
print()
print(y.sort_values())

# %% 
# All impute variables except price
X = autos_encoded.drop(columns=['price'])

# %%
discrete_features_mask = [col in (categorical + discrete_numerical) for col in X.columns]

# %% 
# Mutual inframation
mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features_mask)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print()
print('Mutual Inforamtion:\n', mi_series)

# %% 
# Distribute on train\test
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)

# %% 
# The Model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print()
print('The importance of sings from RandomForest:\n', importances)

# %% 
# Reduction to rank percentages
mi_rank = mi_series.rank(pct=True)
importance_rank = importances.rank(pct=True)

# %% 
# Unite
ranks_autos = pd.concat([mi_rank, importance_rank], axis=1)
ranks_autos.columns = ['MI Rank', 'Model Importance Rank']
ranks_autos.head(10)

# %% 
# Melt for seaborn
ranks_melted = ranks_autos.reset_index().melt(id_vars='index',
                                              var_name = 'Metric',
                                              value_name = 'Rank')
ranks_melted.rename(columns={'index': 'Feature'}, inplace=True)
ranks_melted.head(10)

# %% 
# Draw of a schedule
feature_order = ranks_autos.mean(axis=1).sort_values(ascending=False).index
sns.set(style='whitegrid')

g = sns.catplot(data=ranks_melted,
            kind='bar',
                x='Rank',
                y='Feature',
                hue='Metric',
                order=feature_order,
                height=8,
                aspect=1.2,
                palette='muted',
                legend_out=False)
g.set_titles('Comparison of MI rank values and feature importance')
g.set_xlabels("Rank (Normalized)")
g.set_ylabels("Feature")
plt.tight_layout()
plt.show()