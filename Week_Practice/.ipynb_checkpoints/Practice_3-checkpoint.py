import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import category_encoders as ce
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
# %%
data = pd.read_csv('https://github.com/Limwail/machine-learning-practice/raw/refs/heads/master/datasets/startups_in_turkey.csv')

# %%
fig, ax = plt.subplots(figsize = (8, 5))
sns.countplot(data, x = 'category', hue = "success", ax=ax)
ax.tick_params(axis='x', labelrotation=45)

# %%
describe_categical = data.describe(include='object').T

# %%
describe_numerical = data.describe().T

# %%
data.drop(columns=['project_name', 'project_owner_name', 'location',
                   'project_start_date', 'project_end_date', 
                   'description', 'id', 'Unnamed: 0'], inplace=True)

# %%
data.isna().mean()

# %%
X_train, X_test, y_train, y_test = train_test_split(data.drop('success', axis=1),
                                                    data["success"], train_size=0.7)

# %%
num_features = data.select_dtypes(include=np.number).columns
cat_features = data.select_dtypes(include='object').columns

# %%
# X = data.drop('success', axis=1)
# X_train, X_test=...
# scaler.fit(X_train)
# X = scaler.transform(X)

# %%
scaler = pp.StandardScaler()

X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# %%
X_train['region'] = X_train['region'].fillna('uncertain')
X_test['region'] = X_test['region'].fillna('uncertain')
# %%
encoder = ce.OneHotEncoder(use_cat_names = True)

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
# %%
log_model = LogisticRegression(class_weight = "balanced"
                               ).fit(X_train, y_train)
prediction = log_model.predict(X_test)

print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

# # %%
# sm_model = sm.Logit(y_train.mao({'yes':1, 'no':0}).values), X_train.values.fit()

# print(sm_model.summary())

# %%

probabilities = log_model.predict_proba(X_test)

# %%

tree_model = DecisionTreeClassifier(class_weight = "balanced")
tree_model.fit(X_train, y_train)
prediction = tree_model.predict(X_test)

print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

# %%
features = pd.Series(tree_model.feature_importances_,
                    index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=features, y=features.index, orient='h')
plt.title('Feature Importance')
plt.show()

np.sum(features)


# %%

plot_tree(tree_model, feature_names=X_train.columns)