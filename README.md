# concepte useful de stiut (very messy for now :((( 🐻 with me)

-----
-----

### IMPORTS + SETUP

```python

### basic stuff

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json


### validare 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold


### metrics

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


### preprocesare

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif



### modele

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import IsolationForest


### unsupervised stuff + anomaly detection
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans

```

```python
### NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
```

```python

### deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

### vision
import torchvision
from torchvision import transforms, models
from PIL import Image

### device + seeds
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
print('Device:', DEVICE)

```

-----
-----
-----

### DATA READING

* CSV:
```python
train_data = pd.read_csv('/content/train_data.csv')
test_data = pd.read_csv('/content/test_data.csv')
print(train_data.shape, test_data.shape)
```

* JSON:
```python
with open('/content/file.json') as f:
  data = json.load(f)
print(type(data), len(data))
print(str(data)[:500])
```

* JSONL:
```python
data =  pd.read_json(path, lines=True)
```

* imagini: 
```python
from PIL import Image

paths = []
labels = []
for label in os.listdir('/content/train'):
  for file in os.listdir(f'/content/train/{label}'):
    paths.append(f'/content/train/{label}/{file}')
    labels.append(label)
train_data = pd.DataFrame({'path': paths, 'label':labels})

img = Image.open(paths[0])
print(len(train_data), img.size, img.mode)


```

-----
-----
-----

### DATA EXPLORATION

### First look: 
```python
print(train_data.shape, test_data.shape)
print(train_data.head())
train_data.info()                                     
print(set(train_data.columns) - set(test_data.columns))
```
!! coloane care sunt numerical, dar sunt de type object => possible "dirty" missing values

### Checking target values: 
```python
target = 'Price' #exemplu
print(train_data[target].describe())    
print(train_data[target].value_counts().head(10))
train_data[target].hist(bins=50)  
```
verificam value counts pentru ca poate exista un fel de clustering (de ex aia cu 9.99 si 14.99 la prices)

putem sa variam hist bins!! de obicei se pun 10 la inceput, dar asa s-ar putea sa ratam spikeurile (de ex observam usor 14.99 si 9.99 la vaoues 0-70 cu 50 bins pt ca ar veni cam 1.4 pentru fiecare)

### Missing values
```python
print(train_data.isna().sum()) #cate missing values avem pt fiecare coloana (ca sa identificam coloanele cu missing values)
print(test_data.isna().sum().sum()) #cate missing values avem overall pe tot tabelul
```
Gapsurile:
- le lasam NaN pt HistGradientBoosting
- le dai adress manual / cu SimpleImputer
!! daca o coloana are FOARTE multe missing values, de cele mai multe ori the best option e sa ii dai drop si gata (inainte de a da build la X si Y)

#### Dirty/undercover missing values 
```python
### varianta 1, restrans, whole table:
for col in train_data.select_dtypes(exclude='number').columns:
  as_numbers = pd.to_numeric(train_data[col], errors='coerce')
  bad = train_data[col][as_numbers.isna() & train_data[col].notna()]
  print(col, '| not numbers:', len(bad), 'of', len(train_data), '|', bad.value_counts().head(5).to_dict())

### varianta 2, explicit, o singura coloana):
bad = []
for v in train_data['Age']:
  try:
    float(v)
  except:
    bad.append(v)
print(pd.Series(bad).value_counts().head(10))
# avem pd.series pt ca value_counts().head(10) da return la a pd Series (pe care il transformam in dict sa fie mai citibil pe ecran, asa ca il transformam inapi dupa in pd Series)
```
Daca:
- aproape toate val sunt bad (non-numerical) => pb chiar e a real text column (ex Occupation)
- putin val bad => pb avem dirt (ex "23_" sau "_")

other relevant / fixing stuffsies:
```python
### dupa ce dai suspect la un character (ex _ in 23_):
col = train_data['Age']
print(col[col.astype(str).str.contains('_', regex=False)].value_counts().head(10))
# regex fals ca sa ia acel caracter asa cum e, pt ca unele caractere sunt speciale idk

### useful, tipuri:
print(col.apply(type).value_counts())

### patching stuff up & more
data['Occupation'] = data['Occupation'].replace('_______', np.nan)
data['Credit_Mix'] = data['Credit_Mix'].replace(['_', '!@9#%8'], np.nan)
data['Age'] = pd.to_numeric(data['Age'].astype(str).str.strip('_'), errors='coerce')
data.loc[(data['Age'] < 0) | (data['Age'] > 100), 'Age'] = np.nan
data['col'] = data['col'].replace(-1, np.nan)

### flagging values care erau missing!!
data['Age_missing'] = data['Age'].isna().astype(int)

### de obicei astea sunt among the most common alea care apar, ca sa vezi valorile care apar de cele mai multe ori:
for col in train_data.columns:
  print(col, train_data[col].value_counts().head(5).to_dict())

### sa dai compare before&after:
print(train_data['Age'].isna().sum(), data['Age'].isna().sum())  # (daca ai modificat in data)
print(data['Age'].dtype)  # ar tb sa fie numerical / nu object sa zicem idk
print(data['Age'].describe())
print((data['Occupation'] == '_______').sum())  # un check mai specific
```


### Unique values, duplicates, etc
```python
print(train_data.nunique().sort_values()) #alea mai constante vor fi mai sus

constant = []
for col in train_data.columns:
  if train_data[col].nunique() == 1:
    constant.append(col)
print('constant:', constant)

print('duplicate rows:', train_data.duplicated().sum())
```

### Ranges, weird/impossible vals
```python
print(train_data.describe().T[['min', 'max', 'mean', 'std']])
```
!! ai grija la std


### Train vs Test data
```python
for col in test_data.select_dtypes(exclude='number').columns:
  unseen = (~test_data[col].isin(train_data[col])).mean()
  print(col, '| test rows with a value never seen in train:', round(unseen, 3))

## varianta explicita pentru unseen
for col in test_data.select_dtypes(exclude='number').columns:
  train_values = set(train_data[col])
  count = 0
  for v in test_data[col]:
    if v not in train_values:
      count += 1
  unseen = count / len(test_data)
  print(col, '| test rows with a value never seen in train:', round(unseen, 3))

# alta varianta (interiorul loopului, pentru known, de ex):
known = set(train_data['Publishers'])
answers = []
for v in test_data['Publishers']:
  answers.append(v in known)
##^^^ astea sunt pe row, deci nu individual/unique valori

##---------##

### acum si alea cu numere
for col in test_data.select_dtypes('number').columns:
  outside = ((test_data[col] > train_data[col].max()) | (test_data[col] < train_data[col].min())).mean()
  if outside > 0:
    print(col, '| test rows outside train range:', round(outside, 3))
```
!! e good practice sa dai run la acest check dupa data cleaning
<img width="917" height="686" alt="Screenshot 2026-10-16 at 14 26 53" src="https://github.com/user-attachments/assets/df5e8127-50cc-4e42-bdd3-743faaf003db" />
^^^ pt aia cu ce exista in train/test

!! ai grija cand apar valori noi si faceai cu ordinalencoder:
<img width="627" height="865" alt="Screenshot 2026-10-16 at 14 28 18" src="https://github.com/user-attachments/assets/74e4f26b-eb44-4e61-af7b-6fe109ceca5c" />


### Groups 
(nu vor fi revealed de duplicates pt ca row-urile nu sunt fully identice, au doar componente asa, in rest pb sunt very similar, de ex ac CostumerID)
```python
print(train_data['Customer_ID'].value_counts().head())
print(test_data['Customer_ID'].isin(train_data['Customer_ID']).mean()) #share of test with known groups
```
!! daca a doua linie iese:
- close to 0 => test are grupuri noi most likely => folosim GroupKFold
- close to 1 => test are grupuri cunoscute => folosim stratified folds (si incercam sa dam consider la features din each other group's row)


### Column - target relation
* Pentru number cols:
```python
print(train_data.select_dtypes('number').corr(method='spearman')[target].sort_values())
```
Fiecare primeste val intre -1 (opposite direction) si +1 (going up together), iar 0 e inconclusive (for example, da miss la U-shapes)

* Pentru categ cols:
```python
print(train_data.groupby('Occupation')[target].agg(['count', 'median']))
```
^^^ ia fiecare valoare din coloana respectiva si afiseaza statisticile de count si median de la targetul rowurilor cu acea valoare pt coloana selectata
!! daca medienele difera mult intre valori => coloana este relevanta!! (tho ai grija ca o mediana de la 2 rows e useless, ai grija si la countul fiecarei valori)


```python
exploded = train_data.assign(Genre=train_data['Genres'].str.split(',')).explode('Genre')
print(exploded.groupby('Genre')[target].agg(['count', 'median']))
```
!! NU SE ANTRENEAZA PE EXPLODED !! asta e doar asa mai mult de explorare

### Choosing the validation split
```python
print(train_data[target].value_counts(normalize=True))  # class balance??                 
print(test_data['Customer_ID'].isin(train_data['Customer_ID']).mean())    # cate grupuri sunt shared intre test si train
```
!! rows in grupuri => GroupKFold / StratifiedGroupKFold
!! clasificare => StratifiedKFold

usually se fac 5 folds (maybe 3 daca training is slow)

IMPORTANT: "A model trained on unbalanced data tends to ignore the rare class too. Two fixes: class_weight='balanced', which makes mistakes on the rare class count more, and tuning the decision threshold, like the cutoff in the anomaly task."

### Plots
* Hist bins:
```python
train_data[target].hist(bins=50)  
```
* 2 feateruri, in functie de clasa:
```python
plt.scatter(train_data['col_a'], train_data['col_b'], c=train_data[target], s=5)
plt.xlabel('col_a'); plt.ylabel('col_b')
plt.show()
```
* c=train_data[target] - coloreaza fiecare pct in functie de cls (ex 0/rosu, 1/verde)
* s=5 face punctele mai mici
* pozitie across col_a, inaltime across col_b
<img width="718" height="163" alt="Screenshot 2026-10-16 at 04 16 43" src="https://github.com/user-attachments/assets/91a5b3e2-8466-473b-80b2-95fd4386c8b8" />



### Checkul dinainte de model, dupa ce am dat clean!!
```python
print(list(X_clean.select_dtypes(exclude='number').columns))  # ar tb sa fie []
print(list(X_clean.columns) == list(test_clean.columns))  # ar tb sa fie true
```
!! NaN e ok: coloanele raman float64 daca erau numerice (doar text/other stuff transforma column typeul in object)

--------

### NLP exploration
* te uiti peste date pt chestii ca scrambled words (sau scrambled letters), special tokens ca $NE$ etc
* lungime
```python
train_data['text'].str.split().str.len().describe() 
```
^^^ informatii related de numarul de cuvinte al textului
!! alea mai lungi poate au nevoie de chunking !!!!!!!!!!!YOOOOOOO DE VAZUT CUM FACEM ASTA!!!!

-------- 

### CV exploration

#### Images / class
```python
print(train_data['label'].value_counts())
```

#### random ahh imgs
```python
classes = train_data['label'].unique()
fig, axes = plt.subplots(len(classes), 5, figsize=(15, 3 * len(classes)))
for r in range(len(classes)):
  rows = train_data[train_data['label'] == classes[r]].sample(5, random_state=0)
  for k in range(5):
    axes[r, k].imshow(Image.open(rows['path'].iloc[k]))
    axes[r, k].set_title(classes[r])
    axes[r, k].axis('off')
plt.show()
```


DE TERMINAT CV SECTION DE AICI GANG!!!!!!!



-------- 
-------- 
-------- 












































SELECTARE DATE IDKK??
.loc and .iloc
.loc selects by label, or by a condition:
df.loc[5, 'Price'] is the row whose label is 5.
df.loc[df['Price'] > 10] is the rows where the condition holds.
.iloc selects by position: df.iloc[5] is the sixth row, and df.iloc[fit_rows] takes the rows at those positions.
What to be careful with:
After shuffling, filtering or dropping rows, labels no longer match positions, so [i] and .loc[i] can pick the wrong row. reset_index(drop=True) renumbers them 0, 1, 2… again.
df[col][i] = x doesn't save. Use .loc, or build a list and assign the whole column.







value_counts() counts how often each distinct value appears, most frequent first. value_counts(normalize=True) gives shares instead of counts.










ERORI
MAE
MSE
F1 (weighted + binary)
Accuracy



Data reading

Aici pd read csv



Data exploration
* Missing values
* !! disguised missing values!! (cum???)
Number of unique values: nunique()=
Duplicates
de verif daca exista vals in test care nu apar in train
Constant values
Date in test cat si in train
Describe (min, max, mean - pe coloane usually)
De vazut ce data types sunt incluse acolo?? Ceva ceva dtypes???
= sa nu uitisa verifici la final ca toate sunt numbers, perhaps?? 
8 maybe sectiune separata pt CV?
Tot asa pt NLP

Checking for groups, and choosing what to group by
Duplicates won't show it: a customer's monthly rows aren't identical, just alike. Instead:
Find columns that repeat. For each ID-like column, compare nunique() with the number of rows; fewer distinct values means repeats. value_counts().head() shows rows per group.
Check whether test shares those groups: test_data[col].isin(train_data[col]).mean().
Close to 0: test has new groups, so use GroupKFold.
Close to 1: test has known groups, so use plain (stratified) KF






DATA CLEANING
Functie de OH encoder, maybe??

OrdinalEncoder (Cleaning section)
It turns each category into a number from 0 to n−1. By default the order is alphabetical, not by meaning: Bad, Good, Standard become 0, 1, 2, which is wrong for Credit_Mix. To set the order yourself: OrdinalEncoder(categories=[['Bad', 'Standard', 'Good']]).
A category that appears only in test crashes it, unless you pass handle_unknown='use_encoded_value', unknown_value=-1.

Data visualization
Plt work ig?? (not really necessary)

SimpleImputer (Cleaning section)
It fills each column's missing values with one value per column, learned from train.
Setting	Options
strategy	'mean' (the default), 'median' (safer with long tails), 'most_frequent' (works for text categories too), 'constant' (with fill_value=...)
missing_values	What counts as missing: np.nan by default, or for example 0
add_indicator=True	Also adds a 0/1 "was missing" column for each column that had gaps



CROSS VALIDATION

Stratified k fold, group k fol, k??? Idk de obicei 5
Cand facem, cand nu facem



MODELE 
ML
Histgradboost thingie

NLP



## SUBTASKS + SUBMISSION
SUBTASKS
Pd data frames, aia cu index

