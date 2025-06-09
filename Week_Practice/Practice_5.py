import pandas as pd
import kagglehub
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# %%
data_path = kagglehub.dataset_download('arjunbhasin2013/ccdata')
data = pd.read_csv(data_path + '/CC GENERAL.csv')
data.head()

# %%
data.drop('CUST_ID', axis=1, inplace=True)
# %%
data.isna().sum()

# %%
data.describe()

# %%
data.dropna(subset=['CREDIT_LIMIT'], inplace=True)
data['MINIMUM_PAYMENTS'] - data['MINIMUM_PAYMENTS'].fillna(0)             
# %% 
data_scaled = StandardScaler().fit_transform(data)

pca = PCA(n_components=2)
pca.fit(data_scaled)
pca_components = pca.component_
pca_components

# %%
loadings = pd.DataFrame(pca_components.T,
columns=['PC1', 'PC2'])



# %%
sns.set_style('white')
plt.figure(figsize=(9,5))

sns.scatterplot(x='PC1', y='PC2', data=loadings.reset_index(),
                hue='index', style='index', s=100)
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
plt.axvline(0, color='gray', linewidth=0.8, linestyle='--')
plt.title('PCA Loading Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# %%
loadings = loadings.reset_index().rename(columns={'index': 'Feature'})

fig = px.scatter(
    loadings,
    x='PC1',
    y='PC2',
    color='Feature',
    symbol='Feature',
    title='PCA Loading Plot (Interactive)',
    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
    width=600, height=600)

fig.update_traces(marker=dict(size=12, opacity=0.7), textposition='top center')
fig.update_layout(
    xaxis=dict(title='Principal Component 1', zeroline=False),
    yaxis=dict(title='Principal Component 2', zeroline=False),
    showlegend=False)

# %%
features_classifier = KMean(n_cluster=5, random_state=37)
features_classifier.fit(loadings)

# %%
loadings['cluster'] = features_clussifier.fit_predict(loadings)


