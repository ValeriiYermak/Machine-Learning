import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer 
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, kbins, scaler, encoder

# %%
df = pd.read_csv('C:/Projects/MasterSc/ML/Practice/mod_04_topic_08_petfinder_data.csv.gz')
# %%
data=df.copy()
# %%
data.info()
# %%
data.nunique()
# %%
data['Description'].head()
# %%
data.drop('Description', axis=1, inplace=True)
# %%
data['AdoptionSpeed'].value_counts().sort_index()
# %%
data['AdoptionSpeed'] = np.where(data['AdoptionSpeed'] == 4, 0, 1)
data['AdoptionSpeed'].value_counts()
# %%
data['Fee'] = data['Fee'].astype(bool).astype(int).astype(str)
# %%
X_train, X_test, y_train, y_test = (train_test_split(data.drop('AdoptionSpeed', axis=1), 
                                                    data['AdoptionSpeed'],test_size=0.2,random_state=42))
# %%
num_cols = X_train.select_dtypes(exclude='object').columns
kbins = KBinsDiscretizer(encode='ordinal').fit(X_train[num_cols])
X_train[num_cols] = (kbins.transform(X_train[num_cols]).astype(int).astype(str))
X_test[num_cols] = (kbins.transform(X_test[num_cols]).astype(int).astype(str))
# %%
encoder = ce.TargetEncoder()
X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)
X_train.head(10)
# %%
scaler = StandardScaler().set_output(transform='pandas')
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %%
clf = SVC(class_weight='balanced',kernel='poly',probability=True,random_state=42)
clf.fit(X_train, y_train)
# %%
preds = clf.predict(X_test)
confusion_matrix(y_test, preds)
# %%
print(f'Model accuracy is: {accuracy_score(y_test, preds):.1%}')

# %%
pet = pd.DataFrame(
    data={
         'Type': 'Cat',
         'Age': 3,
         'Breed1': 'Tabby',
         'Gender': 'Male',
         'Color1': 'Black',
         'Color2': 'White',
         'MaturitySize': 'Small',
         'FurLength': 'Short',
         'Vaccinated': 'No',
         'Sterilized': 'No',
         'Health': 'Healthy',
         'Fee': True,
         'PhotoAmt': 2,
     },
     index=[0])
# %%
pet[num_cols] = kbins.transform(pet[num_cols]).astype(int).astype(str)
prob = (clf
         .predict_proba(
             scaler
             .transform(
                 encoder
                 .transform(
                     pet)))
         .flatten())
# %%
print(f'This pet has a {prob[1]:.1%} probability "of getting adopted"')

# %%
