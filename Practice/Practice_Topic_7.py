import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import category_encoders as ce
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# %%
df = pd.read_csv('C:/Projects/MasterSc/ML/Practice/Bank/bank-additional-full.csv', sep=';')
data=df.copy()
data.head(10)
# %%
data.drop('duration', axis=1, inplace=True)

data.describe()

# %%
data.skew(numeric_only=True)

# %%
data = data[np.abs(zscore(data['campaign'])) < 2]

# %%
mtx = data.drop('y', axis=1).corr(numeric_only=True).abs()

fig, ax = plt.subplots(figsize=(8, 8))

sns.heatmap(mtx,
       cmap='crest',
       annot=True,
       fmt=".2f",
       linewidth=.5,
       mask=np.triu(np.ones_like(mtx, dtype=bool)),
       square=True,
       cbar=False,
       ax=ax)

plt.show()

# %%
data.drop(
     ['emp.var.rate',
      'cons.price.idx',
      'nr.employed'],
     axis=1,
     inplace=True)
# %%
data.select_dtypes(include='object').nunique()

# %%
data['y'] = data['y'].replace({'no': 0, 'yes': 1})
# %%
X_train, X_test, y_train, y_test = (train_test_split(data.drop('y', axis=1),data['y'],test_size=0.2,random_state=42))

# %%
cat_cols = X_train.select_dtypes(include='object').columns

encoder = ce.WOEEncoder(cols=cat_cols)
X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)
# %%
power_transform = PowerTransformer().set_output(transform='pandas')

X_train = power_transform.fit_transform(X_train)
X_test = power_transform.transform(X_test)
# %%
X_train.skew()
# %%
y_train.value_counts(normalize=True)
# %%
sm = SMOTE(random_state=42, k_neighbors=50)
X_res, y_res = sm.fit_resample(X_train, y_train)
# %%
knn_mod = KNeighborsClassifier(n_neighbors=7, n_jobs=-1).fit(X_res, y_res)
knn_preds = knn_mod.predict(X_test)
knn_score = balanced_accuracy_score(y_test, knn_preds)
print(f'KNN model accuracy: {knn_score:.1%}')
# %%
gnb_mod = GaussianNB().fit(X_res, y_res)
gnb_preds = gnb_mod.predict(X_test)
gnb_score = balanced_accuracy_score(y_test, gnb_preds)
print(f'GNB model accuracy: {gnb_score:.1%}')

# %%
confusion_matrix(y_test, gnb_preds)
