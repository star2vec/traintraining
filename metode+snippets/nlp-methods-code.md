# NLP-methods-code.md — code blocks for every approach in NLP-methods.md

Each block says what goes in, what comes out (type and shape), then the code. No Pipelines: every step is visible so it can be rewritten from memory. Section numbers match NLP-methods.md.

Imports used everywhere:
```python
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
```

-----

## 0. Load and look

In: csv paths. Out: `train`, `test` DataFrames with a clean `text` column (string, no NaN).
```python
train = pd.read_csv('train.csv'); test = pd.read_csv('test.csv')
train['text'] = train['text'].fillna('').astype(str)
test['text'] = test['text'].fillna('').astype(str)
print(train.shape, test.shape); print(train.columns.tolist())

print(train['label'].value_counts())                       # class balance
lengths = [len(t.split()) for t in train['text']]
print(pd.Series(lengths).describe())                        # word counts
print(train['text'].duplicated().sum())                     # duplicates

for label in train['label'].unique():                       # 3 texts per class
  rows = train[train['label'] == label]
  for k in range(3):
    print(label, '|', rows['text'].iloc[k][:200])

# share of test words seen in train
train_words = set()
for t in train['text']:
  for w in t.split():
    train_words.add(w)
seen = 0; total = 0
for t in test['text']:
  for w in t.split():
    total += 1
    if w in train_words: seen += 1
print('test words seen in train:', seen / total)
```

Labels as numbers and back:
```python
classes = sorted(train['label'].unique())                   # list of names, fixed order
class_to_id = {}
for i in range(len(classes)): class_to_id[classes[i]] = i
y = np.array([class_to_id[c] for c in train['label']])      # (N,) ints
# back: classes[pred_id]
```

-----

## A. TF-IDF

In: list or Series of strings. Out: a sparse matrix `(N, V)` with floats in [0, 1]; V = vocabulary size.
```python
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train = vec.fit_transform(train['text'])      # fit = build vocabulary; transform = count
X_test = vec.transform(test['text'])            # same vocabulary; new words are ignored
print(X_train.shape, X_test.shape, type(X_train))   # (N, V) (M, V) scipy.sparse csr_matrix

X_train[0]                                      # one row, still sparse
X_train[0].toarray()                            # (1, V) dense numpy
X_train[:5].toarray()[0][:20]                   # first 20 numbers of the first text, mostly 0
words = vec.get_feature_names_out()             # (V,) array of strings, column j = words[j]
```
Style version:
```python
vec_c = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True)
```
Fitting the vocabulary on all text (allowed, no labels used):
```python
vec.fit(list(train['text']) + list(test['text']))
X_train = vec.transform(train['text']); X_test = vec.transform(test['text'])
```
Most important words per class (after a LogisticRegression `clf` on `X_train`):
```python
for i in range(len(classes)):
  order = np.argsort(-clf.coef_[i])             # coef_ is (K, V); for binary it is (1, V)
  print(classes[i], [words[j] for j in order[:15]])
```

-----

## B. LogisticRegression on TF-IDF + validation by hand

In: `X` sparse `(N, V)`, `y` `(N,)` ints. Out: `pred` `(M,)` ints, `proba` `(M, K)` floats, rows sum to 1.
```python
clf = LogisticRegression(C=10, max_iter=2000)
clf.fit(X_train, y)
pred = clf.predict(X_test)                      # (M,)
proba = clf.predict_proba(X_test)               # (M, K); column k = clf.classes_[k]
print(clf.classes_)                             # the order of the columns in proba
pred_from_proba = clf.classes_[proba.argmax(axis=1)]   # same as pred
```
Validation without Pipeline (the vectorizer is fit inside each fold):
```python
texts = list(train['text'])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = []
for fit_idx, val_idx in skf.split(texts, y):
  fit_texts = [texts[i] for i in fit_idx]; val_texts = [texts[i] for i in val_idx]
  vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
  Xf = vec.fit_transform(fit_texts); Xv = vec.transform(val_texts)
  clf = LogisticRegression(C=10, max_iter=2000).fit(Xf, y[fit_idx])
  scores.append(f1_score(y[val_idx], clf.predict(Xv), average='macro'))
print(np.mean(scores), np.std(scores))
```
Same loop with `KFold` and `mean_absolute_error` for regression, `GroupKFold(5).split(texts, y, groups)` when texts share an author or document.

Trying C values: wrap the loop in `for C in [1, 10, 100]:`.

-----

## C. Hand features and stacking

In: list of strings. Out: `(N, F)` numpy of floats, one row per text.
```python
def text_features(t):
  words = t.split()
  n_words = len(words)
  n_chars = len(t)
  avg_word = 0
  if n_words > 0: avg_word = n_chars / n_words
  n_caps = 0; n_punct = 0; n_digits = 0
  for ch in t:
    if ch.isupper(): n_caps += 1
    if ch in '!?.,;:': n_punct += 1
    if ch.isdigit(): n_digits += 1
  n_unique = len(set(words))
  return [n_words, n_chars, avg_word, n_caps, n_punct, n_digits, n_unique, t.count('$NE$')]

F_train = np.array([text_features(t) for t in train['text']])   # (N, 8)
F_test = np.array([text_features(t) for t in test['text']])
```
Scale, then glue to TF-IDF (sparse stays sparse):
```python
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
sc = StandardScaler().fit(F_train)
X_all_train = hstack([X_train, csr_matrix(sc.transform(F_train))]).tocsr()   # (N, V + 8)
X_all_test = hstack([X_test, csr_matrix(sc.transform(F_test))]).tocsr()
```
Or feed `F_train` alone to `HistGradientBoostingClassifier` as a second, different model.

-----

## D. Combining two models

In: two `proba` arrays `(M, K)` with the same column order. Out: one `pred`.
```python
p = (proba_words + proba_chars) / 2
pred = p.argmax(axis=1)
```
Both classifiers must have the same `classes_`; true when both saw the same `y`. Weighted: `0.7 * p1 + 0.3 * p2`, weights chosen on validation.

-----

## E. Word2Vec

In: list of lists of tokens (one inner list per text). Out: `model.wv[word]` is `(100,)`; a text becomes the mean `(100,)`; all texts `(N, 100)` numpy.
```python
from gensim.models import Word2Vec
sentences = []
for t in list(train['text']) + list(test['text']) + list(unlabeled['text']):
  sentences.append(t.split())

w2v = Word2Vec(sentences, vector_size=100, window=3, min_count=2, sg=1, epochs=50, seed=0, workers=1)
print(len(w2v.wv))                          # vocabulary size
print(w2v.wv['xqzt'].shape)                 # (100,)
print(w2v.wv.most_similar('xqzt')[:5])     # sanity check: neighbors should make sense

def text_vector(t):
  total = np.zeros(100); count = 0
  for w in t.split():
    if w in w2v.wv:
      total = total + w2v.wv[w]; count += 1
  if count == 0: return total               # all unknown -> zeros
  return total / count

E_train = np.array([text_vector(t) for t in train['text']])   # (N, 100)
E_test = np.array([text_vector(t) for t in test['text']])
clf = LogisticRegression(C=1, max_iter=2000).fit(E_train, y)
```
Count how many texts came out all zeros: `(np.abs(E_test).sum(axis=1) == 0).sum()`.

-----

## F. Sentence embeddings (SentenceTransformer)

In: list of strings. Out: `(N, d)` numpy floats, each row length 1 when `normalize_embeddings=True`.
```python
from sentence_transformers import SentenceTransformer
st = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
E_train = st.encode(list(train['text']), normalize_embeddings=True, batch_size=128, show_progress_bar=True)
E_test = st.encode(list(test['text']), normalize_embeddings=True, batch_size=128)
print(E_train.shape, E_train.dtype)         # (N, 384) float32
print(np.linalg.norm(E_train[0]))           # 1.0

clf = LogisticRegression(C=1, max_iter=2000).fit(E_train, y)
```
Cosine between every test text and every train text:
```python
S = E_test @ E_train.T                      # (M, N); S[i, j] = cosine(test i, train j)
nearest = S.argmax(axis=1)                  # (M,) index into train
pred = y[nearest]                           # 1-nearest-neighbor label
top5 = np.argsort(-S, axis=1)[:, :5]        # (M, 5) indices of the 5 most similar
```

-----

## G. Embeddings with AutoModel (no SentenceTransformer)

In: list of strings. Out: `(N, d)` numpy, normalized.
```python
import torch
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModel.from_pretrained(NAME).to(DEVICE).eval()

def embed(texts, batch_size=64, max_length=256):
  pieces = []
  for start in range(0, len(texts), batch_size):
    batch = texts[start:start + batch_size]
    enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
      out = model(**enc).last_hidden_state                # (B, L, d)
    mask = enc['attention_mask'].unsqueeze(-1).float()    # (B, L, 1): 1 real token, 0 padding
    summed = (out * mask).sum(dim=1)                      # (B, d): zeros padding then adds over L
    counts = mask.sum(dim=1)                              # (B, 1): number of real tokens
    mean = summed / counts                                # (B, d)
    pieces.append(mean.cpu().numpy())
  E = np.concatenate(pieces)                              # (N, d)
  norms = np.linalg.norm(E, axis=1, keepdims=True)        # (N, 1)
  return E / norms

E_train = embed(list(train['text']))
```
What the tokenizer returns:
```python
enc = tok(['hello world', 'a much longer sentence here'], padding=True, return_tensors='pt')
print(enc['input_ids'].shape)         # (2, L): token ids, padded to the longest
print(enc['attention_mask'])          # (2, L): 1 1 1 0 0 for the short one
print(tok.convert_ids_to_tokens(enc['input_ids'][0]))    # ['[CLS]', 'hello', 'world', '[SEP]', '[PAD]', ...]
```
Choosing max_length:
```python
n_tokens = [len(tok(t)['input_ids']) for t in list(train['text'])[:2000]]
print(np.percentile(n_tokens, [50, 90, 95, 99]))
```
[CLS] vector instead of the mean: `out[:, 0]` -> `(B, d)`.

-----

## H. Clustering texts

In: list of strings. Out: `labels` `(N,)` cluster ids 0..K-1.
```python
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

vec = TfidfVectorizer(min_df=2, max_df=0.5, sublinear_tf=True)
V = vec.fit_transform(texts)                        # (N, Vocab) sparse
svd = TruncatedSVD(n_components=100, random_state=0)
S = svd.fit_transform(V)                            # (N, 100) dense numpy
S = Normalizer().fit_transform(S)                   # each row length 1

for K in range(2, 13):                              # choosing K
  km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(S)
  print(K, round(silhouette_score(S, km.labels_), 3), np.bincount(km.labels_))

km = KMeans(n_clusters=K_BEST, n_init=10, random_state=0).fit(S)
labels = km.labels_                                 # (N,)
```
Read and name the clusters:
```python
words = vec.get_feature_names_out()
centers = svd.inverse_transform(km.cluster_centers_)   # (K, Vocab): back to word space
for k in range(K_BEST):
  order = np.argsort(-centers[k])
  print('cluster', k, [words[j] for j in order[:12]])
  idx = np.where(labels == k)[0]
  for i in idx[:4]: print('   ', texts[i][:150])
```
Map clusters to categories by hand, then predict:
```python
cluster_to_cat = {0: 'sport', 1: 'politics', 2: 'sport', 3: 'tech'}   # from reading
pred = [cluster_to_cat[c] for c in labels]
```
Confidence per text (small margin = unsure):
```python
D = km.transform(S)                                 # (N, K) distance to each center
D_sorted = np.sort(D, axis=1)                       # each row sorted
margin = D_sorted[:, 1] - D_sorted[:, 0]            # (N,)
```
With sentence embeddings: skip TF-IDF and SVD, `KMeans(...).fit(E)` directly.

-----

## I. Zero-shot with embeddings and with keywords

In: category names + one description each. Out: `pred` `(N,)` category ids.
```python
descriptions = ['an article about sports and matches', 'an article about politics and government', 'an article about technology']
C = st.encode(descriptions, normalize_embeddings=True)   # (K, d)
E = st.encode(texts, normalize_embeddings=True)          # (N, d)
scores = E @ C.T                                         # (N, K)
pred = scores.argmax(axis=1)
```
Keyword counting:
```python
keywords = {0: ['match', 'team', 'goal'], 1: ['minister', 'election', 'law'], 2: ['software', 'device', 'app']}
pred = []
for t in texts:
  low = t.lower()
  best = -1; best_count = -1
  for k in keywords:
    count = 0
    for w in keywords[k]:
      count += low.count(w)
    if count > best_count: best = k; best_count = count
  pred.append(best)
```
Combine: use the keyword answer when the embedding margin is small.

-----

## J. Matching pairs, search, deduplication

In: `Q` `(M, d)` queries, `D` `(N, d)` documents, both normalized. Out: index of the best document per query.
```python
S = Q @ D.T                                 # (M, N)
best = S.argmax(axis=1)                     # (M,)
topk = np.argsort(-S, axis=1)[:, :10]       # (M, 10)
```
Mutual best match (reliable anchors):
```python
best_q_to_d = S.argmax(axis=1)              # (M,)
best_d_to_q = S.argmax(axis=0)              # (N,)
pairs = []
for i in range(len(best_q_to_d)):
  j = best_q_to_d[i]
  if best_d_to_q[j] == i: pairs.append((i, j))
```
Pair classifier (is this pair a duplicate?). In: two embeddings per pair. Out: features `(P, 2d + 1)`.
```python
feats = []
for (i, j) in pair_indices:
  a = E[i]; b = E[j]
  row = [float(a @ b)] + list(np.abs(a - b)) + list(a * b)
  feats.append(row)
F = np.array(feats)                         # (P, 1 + d + d)
clf = LogisticRegression(C=1, max_iter=2000).fit(F, pair_labels)
```
TF-IDF version of search: `vec.fit(documents)`, `S = vec.transform(queries) @ vec.transform(documents).T`, then `S = S.toarray()` before argmax (or `S.argmax(axis=1)` works on sparse too, returns a matrix: `np.asarray(...).ravel()`).

Relative representation (two embedding spaces, shared anchors): `R_a = E_a @ E_a[anchor_ids_a].T`, `R_b = E_b @ E_b[anchor_ids_b].T`, normalize rows, then match `R_a` against `R_b` as above.

-----

## K. Labels per token

In: sentences as lists of tokens, labels as lists of the same length. Out: one row per token, `X` `(T, F)`, `y` `(T,)`.
```python
def token_features(tokens, i):
  w = tokens[i]
  prev_w = '<s>'
  if i > 0: prev_w = tokens[i - 1]
  next_w = '</s>'
  if i < len(tokens) - 1: next_w = tokens[i + 1]
  return {'w': w.lower(), 'prev': prev_w.lower(), 'next': next_w.lower(),
          'suf2': w[-2:], 'suf3': w[-3:], 'cap': w[0].isupper(), 'digit': w.isdigit(),
          'first': i == 0, 'len': len(w)}

rows = []; y_tok = []
for sent, labs in zip(sentences, sentence_labels):
  for i in range(len(sent)):
    rows.append(token_features(sent, i)); y_tok.append(labs[i])

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer()
X_tok = dv.fit_transform(rows)              # (T, F) sparse; strings become one-hot columns
clf = LogisticRegression(C=1, max_iter=2000).fit(X_tok, y_tok)
```
Test: build rows the same way, `dv.transform(rows_test)`, predict, then split predictions back per sentence by counting tokens.

Dictionary baseline:
```python
counts = {}                                 # word -> {label: count}
for sent, labs in zip(sentences, sentence_labels):
  for w, l in zip(sent, labs):
    if w not in counts: counts[w] = {}
    counts[w][l] = counts[w].get(l, 0) + 1
most_common_label = pd.Series(y_tok).value_counts().index[0]
def predict_token(w):
  if w not in counts: return most_common_label
  best = None; best_c = -1
  for l in counts[w]:
    if counts[w][l] > best_c: best = l; best_c = counts[w][l]
  return best
```
Transformer token classification, the alignment part:
```python
enc = tok(tokens, is_split_into_words=True, truncation=True)
word_ids = enc.word_ids()                   # one entry per subtoken: word index or None
labels = []
prev = None
for wid in word_ids:
  if wid is None: labels.append(-100)       # special tokens, ignored by the loss
  elif wid != prev: labels.append(label_ids[wid])   # first subtoken of a word gets the label
  else: labels.append(-100)                 # other subtokens ignored
  prev = wid
```

-----

## L. Regression on text

In: TF-IDF `(N, V)` or embeddings `(N, d)`, `y` floats `(N,)`. Out: `pred` floats `(M,)`.
```python
reg = Ridge(alpha=1.0).fit(X_train, y)      # try alpha 0.3, 1, 3, 10
pred = reg.predict(X_test)
```
Target mean per word without leakage (word level rows, grouped by document):
```python
gkf = GroupKFold(n_splits=5)
for fit_idx, val_idx in gkf.split(df, df['target'], df['doc_id']):
  fit = df.iloc[fit_idx]
  word_mean = fit.groupby('word')['target'].mean()           # Series: word -> mean on fit rows only
  overall = fit['target'].mean()
  val_feat = df.iloc[val_idx]['word'].map(word_mean).fillna(overall)   # unknown word -> overall mean
```

-----

## M. Seq2seq: generate, fine tune, safety rule

In: list of strings. Out: list of strings.
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained('google/flan-t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base').to(DEVICE)

def generate(texts, prefix='grammar: ', batch_size=32):
  outputs = []
  model.eval()
  for start in range(0, len(texts), batch_size):
    batch = [prefix + t for t in texts[start:start + batch_size]]
    enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
      ids = model.generate(**enc, max_new_tokens=128, num_beams=4)   # (B, L_out) token ids
    outputs += tok.batch_decode(ids, skip_special_tokens=True)       # list of strings
  return outputs
```
Fine tune loop (minimal):
```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
for epoch in range(3):
  model.train()
  perm = np.random.permutation(len(src_texts))
  for start in range(0, len(perm), 16):
    idx = perm[start:start + 16]
    src = [prefix + src_texts[i] for i in idx]; tgt = [tgt_texts[i] for i in idx]
    enc = tok(src, padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
    lab = tok(tgt, padding=True, truncation=True, max_length=128, return_tensors='pt')['input_ids'].to(DEVICE)
    lab[lab == tok.pad_token_id] = -100                   # padding is not a target
    loss = model(**enc, labels=lab).loss
    opt.zero_grad(); loss.backward(); opt.step()
  # validate with generate(...) + the task metric, keep best state
```
Safety rule (keep the original when the change is too large):
```python
from difflib import SequenceMatcher
final = []
for original, fixed in zip(test_texts, generated):
  ratio = SequenceMatcher(None, original, fixed).ratio()   # 1.0 = identical
  if ratio < 0.6: final.append(original)                    # threshold chosen on val
  else: final.append(fixed)
```

-----

## N. Given a model: probe, score, embedding arithmetic

Hidden states of a chosen layer. In: strings. Out: `(N, d)` per layer.
```python
model = AutoModel.from_pretrained(NAME, output_hidden_states=True).to(DEVICE).eval()
enc = tok(batch, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
with torch.no_grad():
  out = model(**enc)
hs = out.hidden_states                     # tuple of (n_layers + 1) tensors, each (B, L, d); hs[0] = embeddings
layer = len(hs) // 2                       # middle layer
mask = enc['attention_mask'].unsqueeze(-1).float()
feat = (hs[layer] * mask).sum(1) / mask.sum(1)    # (B, d) masked mean, as in G
```
Then `LogisticRegression` on the stacked features = a probe. Compare a few layers on validation.

Score a candidate continuation with a causal model (higher = more likely). In: prompt and candidate strings. Out: one float per candidate.
```python
from transformers import AutoModelForCausalLM
lm = AutoModelForCausalLM.from_pretrained(NAME).to(DEVICE).eval()
def score(prompt, candidate):
  p_ids = tok(prompt, return_tensors='pt')['input_ids'][0]
  c_ids = tok(candidate, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
  ids = torch.cat([p_ids, c_ids]).unsqueeze(0).to(DEVICE)      # (1, Lp + Lc)
  with torch.no_grad():
    logits = lm(ids).logits[0]                                  # (Lp + Lc, Vocab)
  logprobs = torch.log_softmax(logits, dim=-1)
  total = 0.0
  for k in range(len(c_ids)):
    pos = len(p_ids) + k - 1                                    # the position that predicts token k
    total += logprobs[pos, c_ids[k]].item()
  return total
```
Nearest centroid and a direction. In: `E` `(N, d)`, `y` `(N,)`. Out: `pred` `(M,)`.
```python
centroids = np.array([E[y == k].mean(axis=0) for k in range(K)])      # (K, d)
D = np.zeros((len(E_test), K))
for k in range(K):
  D[:, k] = np.linalg.norm(E_test - centroids[k], axis=1)
pred = D.argmin(axis=1)

direction = centroids[1] - centroids[0]                        # e.g. a gender direction
direction = direction / np.linalg.norm(direction)
proj = E @ direction                                           # (N,) how far along the direction
E_removed = E - np.outer(proj, direction)                      # the direction projected out
```

-----

## O. Audio

In: wav paths. Out: `(N, d)` embeddings.
```python
import torchaudio
from transformers import AutoProcessor, AutoModel
proc = AutoProcessor.from_pretrained('facebook/hubert-base-ls960')
enc_model = AutoModel.from_pretrained('facebook/hubert-base-ls960').to(DEVICE).eval()

def load_16k(path):
  wave, sr = torchaudio.load(path)                 # (channels, samples)
  if wave.shape[0] > 1: wave = wave.mean(dim=0)    # mono
  else: wave = wave[0]
  if sr != 16000: wave = torchaudio.functional.resample(wave, sr, 16000)
  return wave.numpy()                              # (samples,)

vectors = []
for path in paths:
  inputs = proc(load_16k(path), sampling_rate=16000, return_tensors='pt').to(DEVICE)
  with torch.no_grad():
    h = enc_model(**inputs).last_hidden_state      # (1, T, d)
  vectors.append(h.mean(dim=1)[0].cpu().numpy())   # (d,)
E = np.array(vectors)                              # (N, d)
```
Transcribe: `from transformers import pipeline; asr = pipeline('automatic-speech-recognition', model='openai/whisper-small'); asr(path)['text']`.

-----

## P. Fine tune a transformer for classification (minimal loop)

In: strings + `y` ints. Out: `(M, K)` logits.
```python
from transformers import AutoModelForSequenceClassification
import copy
clf_model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=K).to(DEVICE)
opt = torch.optim.AdamW(clf_model.parameters(), lr=2e-5, weight_decay=0.01)

def predict_logits(texts):
  clf_model.eval(); pieces = []
  for start in range(0, len(texts), 64):
    enc = tok(texts[start:start + 64], padding=True, truncation=True, max_length=MAXLEN, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
      pieces.append(clf_model(**enc).logits.cpu().numpy())
  return np.concatenate(pieces)                    # (N, K)

best_score = -1; best_state = None
for epoch in range(3):
  clf_model.train()
  perm = np.random.permutation(len(fit_texts))
  for start in range(0, len(perm), 16):
    idx = perm[start:start + 16]
    enc = tok([fit_texts[i] for i in idx], padding=True, truncation=True, max_length=MAXLEN, return_tensors='pt').to(DEVICE)
    labels = torch.tensor(y_fit[idx]).to(DEVICE)
    loss = clf_model(**enc, labels=labels).loss
    opt.zero_grad(); loss.backward(); opt.step()
  val_pred = predict_logits(val_texts).argmax(axis=1)
  s = f1_score(y_val, val_pred, average='macro'); print(epoch, s)
  if s > best_score: best_score = s; best_state = copy.deepcopy(clf_model.state_dict())
clf_model.load_state_dict(best_state)
```

-----

## Q. Submission

```python
sub = pd.DataFrame({'id': test['id'], 'label': [classes[p] for p in pred]})
print(sub.shape, sub.head()); print(sub['label'].value_counts(normalize=True))
sub.to_csv('submission.csv', index=False)
```
For several subtasks in one file: build each part as its own DataFrame with the exact columns from the statement, then `pd.concat([part1, part2], ignore_index=True)`.
