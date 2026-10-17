# concepte useful de stiut (very messy for now :((( 🐻 with me)

-----
-----

## IMPORTS + SETUP

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

## DATA READING

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

## DATA EXPLORATION

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
(desi poti sa faci schema cu flag column mai bine)

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
  except ValueError:
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

## sau pt fiecare coloana:
numbers = train_data.select_dtypes('number')
for col in numbers.columns:
  print(col, round(numbers[col].corr(numbers[target], method='spearman'), 2))
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
* c=train_data[target] - coloreaza fiecare pct in functie de cls (ex 0/rosu, 1/verde) - vezi ca are nevoie de numere!! asa ca daca e text tb mapping mai intai
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
* labels / samples
```python
print(train_data['label'].value_counts(normalize=True))

sample = train_data.sample(5, random_state=0)
for i in range(len(sample)):
  print(sample['label'].iloc[i], '|', sample['text'].iloc[i])
```
* lungime
```python
train_data['text'].str.split().str.len().describe()
```
^^^ informatii related de numarul de cuvinte al textului
!! alea mai lungi poate au nevoie de chunking!!
```python
lengths = []
for t in train_data['text']:
  lengths.append(len(str(t).split()))
print(pd.Series(lengths).describe())
```
- daca apare 0 => exista empty texts
- peste 200 + transformer => avem nevoie de chunking


-------- 

### CV exploration

* clase, dim, modes, broken files, train vs test
```python
print(train_data['label'].value_counts())

def scan(paths):
  sizes = {}
  broken = 0
  for p in paths:
    try:
      img = Image.open(p)
      key = (img.size, img.mode)
      if key not in sizes:
        sizes[key] = 0
      sizes[key] += 1
    except Exception:
      broken += 1
  return sizes, broken

print('train:', scan(train_data['path']))
print('test: ', scan(test_data['path']))
```
- daca avem more sizes => dam resize la tot la una
- mixed modes => .convert('RGB')
- dam drop la broken files

* random ahh imgs
```python
## train:
classes = train_data['label'].unique()
fig, axes = plt.subplots(len(classes), 5, figsize=(15, 3 * len(classes)))
for r in range(len(classes)):
  rows = train_data[train_data['label'] == classes[r]].sample(5, random_state=0)
  for k in range(5):
    axes[r, k].imshow(Image.open(rows['path'].iloc[k]))
    axes[r, k].set_title(classes[r])
    axes[r, k].axis('off')
plt.show()

## test:
sample = test_data.sample(5, random_state=0)
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for k in range(5):
  axes[k].imshow(Image.open(sample['path'].iloc[k]))
  axes[k].axis('off')
plt.show()
```

-------- 
-------- 
-------- 

## DATA CLEANING

!! 2 tipuri de cleaning:
- row by row (converting, fixing, dropping columns): the same on train and test, inside clean_input
- learned from data (category lists, counts, medians): computed from train only, before or after the function, never from test

!! compute subtask answers on the raw data, before any cleaning

### inainte de data cleaning function
```python
### constant columns
constant = []
for col in X.columns:
  if X[col].nunique() == 1:
    constant.append(col)

### categories for one hot, without the placeholder
occupations = []
for v in X['Occupation'].dropna().unique():
  if v != '_______':
    occupations.append(v)

### common items in a list column
genre_counts = {}
for e in X['Genres'].dropna():
  for g in e.split(','):
    if g not in genre_counts:
      genre_counts[g] = 1
    else:
      genre_counts[g] += 1

common_genres = []
for g in genre_counts:
  if genre_counts[g] >= 10:
    common_genres.append(g)

### rows per category, for columns with many categories
pub_counts = X['Publishers'].value_counts()
```
### functie OH
```python
def onehot(data, col, categs):
  for c in categs:
    column = []
    for v in data[col]:
      if v == c:
        column.append(1)
      else:
        column.append(0)
    data[col+'_'+str(c)] = column
  data = data.drop(columns=col)
  return data
```

### functie cleaning
```python
def damclean(data):
  data = data.copy()

  ## daca avem dat drop la anumite coloane
  data = data.drop(columns=['ID', 'Name'])
  data = data.drop(columns=constant)

  ## daca avem de curatat anumite valori ca sa le facem numerice
  for f in ['Age', 'Annual_Income']:
    data[f] = pd.to_numeric(data[f].astype(str).str.strip('_'), errors='coerce')

  ## daca avem de dat replace la chestii cu NaN
  data['Occupation'] = data['Occupation'].replace('_______', np.nan)
  data.loc[(data['Age'] < 0) | (data['Age'] > 100), 'Age'] = np.nan

  ## din range in numere (ex 20000-50000 -> 35000) 
  mids = []
  for e in data['Estimated owners']: 
    mini, maxi = e.split('-')
    mids.append((int(mini) + int(maxi)) / 2)
  data['Estimated owners'] = mids

  ## lucru cu calendar data de exemplu
  years = []
  for e in data['Release date']:                
    years.append(int(e[-4:]))
  data['Year'] = years
  data = data.drop(columns='Release date')

  ## sa dai order la text stuff!! (daca nu faci cu ordine si faci cu ordinal encoder clasic, se pun in ordine lexicografica)
  order = {'Bad': 0, 'Standard': 1, 'Good': 2}
  mix = []
  for v in data['Credit_Mix']:
    if v in order:
      mix.append(order[v])
    else:
      mix.append(np.nan)                     
  data['Credit_Mix'] = mix

  ## unordered categories -> 0/1 columns
  data = one_hot(data, 'Occupation', occupations)

  ## list columns -> one 0/1 column per common item (daca apar alea multe, altfel daca sunt other sau nu, ca s anu bagam mai mult de like 10 coloane)
  for g in common_genres:
    column = []
    for e in data['Genres']:
      if g in str(e).split(','):
        column.append(1)
      else:
        column.append(0)
    data['Genre_' + g] = column
  data = data.drop(columns='Genres')

  ## un fel de ordinal encoder, dar sa luam dupa value_counts maybe??
  counts = []
  for v in data['Publishers']:
    if v in pub_counts.index:
      counts.append(pub_counts[v])
    else:
      counts.append(0)                       
  data['Publishers'] = counts

  return data
```

### aplica cleaningul
```python
X_clean = clean_input(X)
test_clean = clean_input(test_data)
```

### missing values
* Option 1: leave the NaN. HistGradientBoosting handles them.
* Option 2: for models that crash on NaN: fill with the train median and add a flag column
```python
for col in list(X_clean.columns):
  if X_clean[col].isna().sum()>0 or test_clean[col].isna().sum()>0:
    median = X_clean[col].median()                          
    X_clean[col + '_missing'] = X_clean[col].isna().astype(int)
    test_clean[col + '_missing'] = test_clean[col].isna().astype(int)
    X_clean[col] = X_clean[col].fillna(median)
    test_clean[col] = test_clean[col].fillna(median)
```
^^^ useful functia de fillna!!!!!

### final checks
```python
print(list(X_clean.select_dtypes(exclude='number').columns))   
print(list(X_clean.columns) == list(test_clean.columns))       
print(X_clean.isna().sum().sum(), test_clean.isna().sum().sum())
```

### traps
* data[col][i] = x doesn't save. Build a list, then assign the whole column.
* Strip before coercing, or "23_" becomes NaN instead of 23.
* The first version can be simple: drop the columns that need hard work, submit, then add them back one at a time.

---------

### NLP CLEANING

!! TF-IDF's default tokenizer already ignores punctuation and one-letter words. Stripping changes nothing there unless you change token_pattern.

Exemplu de fct de cleaning:
```python
def clean_text(t, keep='!?'):
  t = str(t).lower()
  chars = []
  for ch in t:
    if ch.isalnum() or ch == ' ' or ch in keep:
      chars.append(ch)
    else:
      chars.append(' ')  # punctuation becomes a space
  t = ''.join(chars)
  t = ' '.join(t.split()) # collapses repeated spaces
  return t

clean = []
for t in train_data['text']:
  clean.append(clean_text(t))
train_data['clean'] = clean
```
Observatii:
- isalnum() is True for letters and digits, including letters with accents
- str(t) turns a missing text into 'nan', so fill missing texts with '' first.

#### chunking:
```python
lengths = []
for t in train_data['text']:
  lengths.append(len(tok(str(t))['input_ids']))
lengths = pd.Series(lengths)
print(lengths.describe())
print('share above 512 tokens:', (lengths > 512).mean())
```
```python
def chunk_words(text, size=300, overlap=50):
  words = str(text).split()
  chunks = []
  start = 0
  while start < len(words):
    chunks.append(' '.join(words[start:start + size]))
    if start + size >= len(words):
      break
    start += size - overlap
  return chunks
```
Observatii:
- size is in words, so keep it safely under the token limit. About 300 words fits in 512 tokens.
- overlap repeats the last 50 words of each chunk at the start of the next. A sentence cut at a boundary then appears whole in at least one chunk.
```python
doc_vecs = []
for text in docs['text']:
  chunks = chunk_words(text, size=80, overlap=20)  
  chunk_vecs = model.encode(chunks)  # one vector per piece
  doc_vec = chunk_vecs.mean(axis=0)  # average the pieces into one vector
  doc_vecs.append(doc_vec)
doc_vecs = np.array(doc_vecs)        # one row per document

## make every vector length 1, so @ gives cosine similarity
lengths = np.sqrt((doc_vecs ** 2).sum(axis=1, keepdims=True))
doc_vecs = doc_vecs / lengths
```
<img width="613" height="292" alt="Screenshot 2026-10-16 at 23 29 02" src="https://github.com/user-attachments/assets/f8b56004-9604-4853-8886-773c6366ac32" />

----------

### CV CLEANING

#### (target) Labels to numbers (pastram classes ca sa convertim dupa aceea inapoi):
```python
classes = sorted(train_data['label'].unique())
class_to_id = {}
for i in range(len(classes)):
  class_to_id[classes[i]] = i

ids = []
for v in train_data['label']:
  ids.append(class_to_id[v])
train_data['y'] = ids
```

#### Preprocessing
```python
IMG_SIZE = 128
mean = [0.485, 0.456, 0.406]   # ImageNet values, for pretrained models
std = [0.229, 0.224, 0.225]    # from scratch: use the values from exploration

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),        # augmentation: only if a flipped image keeps its label
    transforms.RandomRotation(10),            # augmentation: random tilt up to 10 degrees
    transforms.ToTensor(),                    # pixels 0–255 -> 0–1, shape (channels, height, width)
    transforms.Normalize(mean, std),
])

test_tf = transforms.Compose([               
  transforms.Resize((IMG_SIZE, IMG_SIZE)),
  transforms.ToTensor(),
  transforms.Normalize(mean, std),
])
```
^^^ e hepful sa ask ourselves daca a human would still give this image the same label

<img width="630" height="407" alt="Screenshot 2026-10-16 at 23 30 06" src="https://github.com/user-attachments/assets/f5b941e4-95c3-480b-b8d4-581028589167" />



#### Dataset stuff
```python
class ImageDataset(Dataset):
  def __init__(self, paths, labels, tf):
    self.paths = list(paths)
    self.labels = list(labels)
    self.tf = tf

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, i):
    img = Image.open(self.paths[i]).convert('RGB')
    return self.tf(img), self.labels[i]
```
```python
fit_ds = ImageDataset(fit_data['path'], fit_data['y'], train_tf)
val_ds = ImageDataset(val_data['path'], val_data['y'], test_tf) # ai grija ca aici e test_tf, nu train_tf!!
test_ds = ImageDataset(test_data['path'], [0] * len(test_data), test_tf)   # dummy label
```

-------------
-------------
-------------

### SCALING + FEATURE ENGINEERING

* General feature engineering stuff, exemple:
```python
### totals and ratios (smoothed, so no division by 0)
data['Total'] = data['Positive'] + data['Negative']
data['Share'] = (data['Positive']+1) / (data['Total']+2)

### number of items in a list column
# daca avem chestii cu NaN, tb sa le bagam cu '' mai intai idk
data['n_genres'] = data['Genres'].str.count(',') + 1
```

* miau :3 sorry gang its late uhh scaling idk
```python
### fit on train only, then apply the same scaling to both
scaler = StandardScaler()
scaler.fit(X_clean)
X_scaled = pd.DataFrame(scaler.transform(X_clean), columns=X_clean.columns, index=X_clean.index)
test_scaled = pd.DataFrame(scaler.transform(test_clean), columns=test_clean.columns, index=test_clean.index)
```
<img width="830" height="490" alt="Screenshot 2026-10-17 at 00 15 00" src="https://github.com/user-attachments/assets/f5148a50-c213-4cce-ba5f-7c329e60232a" />

!! in cross-validation, daca faci cu pipeline, tb sa bagi asta:
```python
model = Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(max_iter=1000))])
```

-------------
-------------
-------------

### VALIDATION

<img width="789" height="518" alt="Screenshot 2026-10-17 at 00 21 11" src="https://github.com/user-attachments/assets/b5f8e668-122d-4ff6-a24d-f18cd47d4c3a" />

* functia!!
```python
def evaluate(model, X_clean, y, kf, metric, groups=None):
  scores = []
  for fit_rows, val_rows in kf.split(X_clean, y, groups):
    X_fit = X_clean.iloc[fit_rows]
    y_fit = y.iloc[fit_rows]
    X_val = X_clean.iloc[val_rows]
    y_val = y.iloc[val_rows]

    model.fit(X_fit, y_fit)                  
    preds = model.predict(X_val)
    scores.append(metric(y_val, preds))

  print('folds:', [round(s, 3) for s in scores])
  print('mean:', round(np.mean(scores), 4), ' spread:', round(np.std(scores), 4))
  return scores
```
(vezi ca stratifiedkfold are nevoie de clase, nu putem baga pe numerical values)
```python
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

scores0 = evaluate(DummyClassifier(strategy='most_frequent'), X_clean, y, kf, accuracy_score)
scores1 = evaluate(HistGradientBoostingClassifier(random_state=0), X_clean, y, kf, accuracy_score)
```
!! functii mai fancy de evaluare:
```python
f1_weighted = lambda a, b: f1_score(a, b, average='weighted')
evaluate(model, X_clean, y, kf, f1_weighted)
```
!! exemplu pt groups:
```python
evaluate(model, X_clean, y, GroupKFold(n_splits=5), accuracy_score, groups=train_data['Customer_ID'])
```


#### measuring winuri:
```python
wins = 0
for i in range(len(scores1)):
  if scores1[i] > scores0[i]:      # for errors (MAE, MSE) use <
    wins += 1
print('wins:', wins, 'of', len(scores1))
```

#### METRICS!!
<img width="819" height="537" alt="Screenshot 2026-10-17 at 01 39 34" src="https://github.com/user-attachments/assets/5385feee-14be-4f7a-abed-1ef6f8cac4a3" />


#### simple train-test split
```python
fit_data, val_data = train_test_split(train_data, test_size=0.2, stratify=train_data['y'], random_state=0)
```
^^^ fit_data si val_data care apar in ImageDataset, de exemplu


#### Special case: NO LABELS (anomaly detection)
```python
cut = int(len(X) * 0.8)
X_fit = X.iloc[:cut]
X_val = X.iloc[cut:]

## sau cu train_test_split si shuffle=False
X_fit, X_val = train_test_split(X, test_size=0.2, shuffle=False)
```

------------
-----------
-----------

## MODEL

### Tabular data

(setup)!!
```python
kf = KFold(n_splits=5, shuffle=True, random_state=0)               
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  
```

#### Dummy
- Ignores the features, so it works on any data.
- Regression: strategy='median' for MAE, 'mean' for MSE.
- Classification: strategy='most_frequent'
```python
s0 = evaluate(DummyRegressor(strategy='median'), X_clean, y, kf, mean_absolute_error)
```

#### HistGradientBoosting
- Needs: nothing extra (NaN is fine, no scaling)
- PARAMETRII:
  * loss: default 'squared_error';  'absolute_error' pt MAE. 'log_loss' pt classifier
  * learning_rate: how big each correction step is. Default 0.1; try 0.05 and 0.2.
  * max_iter: number of trees. Default 100; try 300 or 500, especially with a smaller learning rate.
  * max_leaf_nodes: how complex each tree is. Default 31; try 15 and 63.
  * class_weight: classifier only. 'balanced' makes rare classes count more.
```python
s1 = evaluate(HistGradientBoostingRegressor(loss='absolute_error', random_state=0), X_clean, y, kf, mean_absolute_error)
```


#### RandomForest
- Needs: X_filled, no NaN. No scaling.
- Parameters to change:
  * n_estimators: number of trees; more is steadier but slower. Default 100; try 300.
  * max_depth: how deep each tree goes. Default unlimited; try 10 and 20.
  * class_weight: classifier only. 'balanced'.
  * Always add n_jobs=-1.
```python
s2 = evaluate(RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0), X_filled, y, kf, mean_absolute_error)
```


#### Linear models
- Needs:
  * X_filled, no NaN.
  * Scaling, through the Pipeline.
- LinearRegression: nothing to tune. The data preparation is what matters.
- LogisticRegression parameters:
  * C: how strictly it avoids overfitting; lower means simpler. Default 1.0; try 0.1 and 10.
  * class_weight: 'balanced' for rare classes.
  * max_iter: training steps. Use 1000; raise it if it warns that it didn't finish.
```python
## log on long-tailed columns, a separate copy just for linear models
X_lin = X_filled.copy()
test_lin = test_filled.copy()
for col in ['Positive', 'Negative', 'Recommendations']:
  X_lin[col] = np.log1p(X_lin[col])
  test_lin[col] = np.log1p(test_lin[col])

s2 = evaluate(Pipeline([('scale', StandardScaler()), ('model', LinearRegression())]))
```


### Tuning!!
* 1 parameter:
```python
results = {}
for lr in [0.05, 0.1, 0.2]:
  model = HistGradientBoostingRegressor(loss='absolute_error', learning_rate=lr, random_state=0)
  scores = evaluate(model, X_clean, y, kf, mean_absolute_error)
  results[lr] = np.mean(scores)

best = None
for v in results:
  if best is None or results[v] < results[best]:  # < for errors (MAE), > for accuracy or F1
    best = v
print('best learning_rate:', best, round(results[best], 4))
```

* 2 parameters (nested loops):
```python
results = {}
for lr in [0.05, 0.1]:
  for leaf in [20, 50]:
    model = HistGradientBoostingRegressor(loss='absolute_error', learning_rate=lr, min_samples_leaf=leaf, random_state=0)
    scores = evaluate(model, X_clean, y, kf, mean_absolute_error)
    results[(lr, leaf)] = np.mean(scores)
```
!! Watch the cost: 2 × 2 values × 5 folds = 20 trainings. With 3 values each it's 45, so nested loops only make sense for fast models and few values.

#### Reguli generale:
* ordering:
  * HistGradientBoosting: loss → learning_rate with max_iter → max_leaf_nodes → min_samples_leaf
  * For random forest: min_samples_leaf → max_features → n_estimators.
* Overfitting: if a model scores much better on its training rows than in validation, move toward simpler settings: higher min_samples_leaf, lower max_depth or max_leaf_nodes, lower C.


### Exemple mai complete:
* regresie:
```python
kf = KFold(n_splits=5, shuffle=True, random_state=0)

s0 = evaluate(DummyRegressor(strategy='median'), X_clean, y, kf, mean_absolute_error)
s1 = evaluate(HistGradientBoostingRegressor(loss='absolute_error', random_state=0), X_clean, y, kf, mean_absolute_error)
s2 = evaluate(RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0), X_filled, y, kf, mean_absolute_error)
s3 = evaluate(Pipeline([('scale', StandardScaler()), ('model', LinearRegression())]), X_lin, y, kf, mean_absolute_error)

## final: the best model, with its best settings, on all training rows
model = HistGradientBoostingRegressor(loss='absolute_error', random_state=0)
model.fit(X_clean, y)
preds = model.predict(test_clean)
```

* classification:
```python
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
f1_weighted = lambda a, b: f1_score(a, b, average='weighted')

s0 = evaluate(DummyClassifier(strategy='most_frequent'), X_clean, y, kf, f1_weighted)
s1 = evaluate(HistGradientBoostingClassifier(random_state=0), X_clean, y, kf, f1_weighted)
s2 = evaluate(RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0), X_filled, y, kf, f1_weighted)
s3 = evaluate(Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(max_iter=1000))]), X_lin, y, kf, f1_weighted)

## unbalanced classes: try class_weight='balanced'
s4 = evaluate(HistGradientBoostingClassifier(class_weight='balanced', random_state=0), X_clean, y, kf, f1_weighted)

## final
model = HistGradientBoostingClassifier(random_state=0)
model.fit(X_clean, y)
preds = model.predict(test_clean)
```






























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

