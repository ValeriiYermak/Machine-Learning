import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import urllib.request
import missingno as msno

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
# %%
url = 'https://raw.github.com/goitacademy/MACHINE-LEARNING-NEO/main/datasets/mod_06_topic_12_nci_data.pkl' 
with urllib.request.urlopen(url) as file:
    dataset = pickle.load(file)

# %%
data = dataset['data']
target = dataset['labels']
data.shape

# %%
target['label'].value_counts().sort_index()

# %%
X = StandardScaler().fit_transform(data)

pca = PCA(random_state=37).fit(X)
pve = pca.explained_variance_ratio_

# %%
sns.set_theme()

kneedle = KneeLocator(
    x=range(1, len(pve) + 1),
    y=pve,
    curve='convex',
    direction='decreasing')

kneedle.plot_knee()

plt.title(f'Knee Point at {kneedle.elbow + 1}')
plt.show()

# %%
n_components = kneedle.elbow

ax = sns.lineplot(np.cumsum(pve))
ax.axvline(x=n_components,
           c='black',
           linestyle='--',
           linewidth = 0.75)
ax.axhline(y=np.cumsum(pve)[n_components],
           c='black',
           linestyle='--',
           linewidth=0.75)
ax.set(xlabel='number of components',
       ylabel='cumulative explained variance')
plt.show()

# %%
X = pca.transform(X)[:, :n_components]

# %%
model_kmn = KMeans(random_state=37)

visualizer = KElbowVisualizer(
    model_kmn,
    k=(2, 10),
    timings=False)

visualizer.fit(X)
visualizer.show()

# %%
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1 #left node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
        ).astype(float)
    dendrogram(linkage_matrix, **kwargs)

# %%   
model_agg = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model_agg = model_agg.fit(X)

plot_dendrogram(model_agg, truncate_mode='level', p=3)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Number of points in node (or index of point if no parenthesis)')
plt.show()

# %%
k_best = visualizer.elbow_value_

model_kmn = KMeans(n_clusters=k_best, random_state=37).fit(X)
model_agg = AgglomerativeClustering(n_clusters=k_best).fit(X)

labels_kmn = pd.Series(model_kmn.labels_, name='k-means')
labels_agg = pd.Series(model_agg.labels_, name='h-clust')

# %%
pd.crosstab(labels_agg, labels_kmn)

# %%
pd.crosstab(target['label'], labels_kmn)

#%%
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5), sharey=True)
for i, s in enumerate([target['label'], labels_kmn]):
    ax = axes[i]
    sns.scatterplot(x=X[:, 0],
                    y=X[:, 1],
                    hue=s,
                    style=s,
                    edgecolor='black',
                    linewidth=0.5,
                    s=60,
                    palette='tab20',
                    legend=False,
                    ax=ax)
    ax.set(title=s.name)
plt.show()