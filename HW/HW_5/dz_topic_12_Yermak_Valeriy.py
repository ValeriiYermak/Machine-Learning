import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import urllib.request
import missingno as msno

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
# %%
url = 'https://raw.github.com/goitacademy/MACHINE-LEARNING-NEO/main/datasets/mod_05_topic_10_various_data.pkl'
with urllib.request.urlopen(url) as file:
    dataset = pickle.load(file)

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# %%
concrete = dataset['concrete']
concrete.info()

# %%
concrete.head(10)

# %%
feature_cols = concrete.columns.drop('CompressiveStrength')
concrete['Components'] = (concrete[feature_cols] > 0).sum(axis=1)
concrete.head(10)

# %%
X = concrete.drop(columns=['CompressiveStrength'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
model = KMeans(random_state=37)
visualizer = KElbowVisualizer(model, k=(2,10))
visualizer.fit(X_scaled)
visualizer.show()

# %%
optimal_k = visualizer.elbow_value_
kmeans = KMeans(n_clusters=optimal_k, random_state=37)
clusters = kmeans.fit_predict(X_scaled)
concrete['Cluster'] = clusters

# %%
median_report = concrete.groupby('Cluster').median(numeric_only=True)

median_report.info()

# %%
counts = concrete['Cluster'].value_counts().sort_index()
median_report['Recipes_count']=counts

print(median_report)

# %%