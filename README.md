# concepte useful de stiut (very messy for now :((( 🐻 with me)

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

### DATA EXPLORATION



























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

