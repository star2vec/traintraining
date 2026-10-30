# nlp-methods-code.md — code blocks for NLP-methods.md

One block per approach, numbered like NLP-methods.md. Every block says what goes in and what comes out.
Convention: `texts` is a python list of strings, `y` is a numpy array of shape (N,), `test_texts` the same for test.

-----

## 0. Load, look, validate

```python
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score

train = pd.read_csv('train.csv'); test = pd.read_csv('test.csv')
train['text'] = train['text'].fillna('').astype(str)
test['text'] = test['text'].fillna('').astype(str)

texts = list(train['text']); y = train['label'].values
test_texts = list(test['text'])

# look
print(train['label'].value_counts())
lengths = [len(t.split()) for t in texts]
print(pd.Series(lengths).describe())
for label in sorted(set(y)):
  print('=====', label)
  rows = train[train['label'] == label]
  for t in rows['text'].iloc[:3]: print(t[:300]); print('-')

# how many test words were seen in train
train_words = set()
for t in texts:
  for w in t.split(): train_words.add(w)
seen = 0; total = 0
for t in test_texts:
  for w in t.split():
    total += 1
    if w in train_words: seen += 1
print('test words seen in train:', seen / total)

# text labels -> numbers and back
classes = sorted(set(y))
class_to_id = {}
for i in range(len(classes)): class_to_id[classes[i]] = i
y_id = np.array([class_to_id[v] for v in y])
# back: classes[pred]

# one evaluate function, reused everywhere; model = anything with fit / predict
def evaluate(model, X, y, metric='f1'):
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
  scores = []
  for fit_rows, val_rows in skf.split(np.zeros(len(y)), y):
    X_fit = [X[i] for i in fit_rows] if isinstance(X, list) else X[fit_rows]
    X_val = [X[i] for i in val_rows] if isinstance(X, list) else X[val_rows]
    model.fit(X_fit, y[fit_rows])
    pred = model.predict(X_val)
    if metric == 'f1': scores.append(f1_score(y[val_rows], pred, average='macro'))
    else: scores.append(accuracy_score(y[val_rows], pred))
  print(np.round(scores, 4), 'mean', round(float(np.mean(scores)), 4))
  return float(np.mean(scores))
```
`X` can be a list of strings (then the model is a Pipeline with a vectorizer) or a numpy / sparse matrix.

-----

## 1a. Content classification: TF-IDF + LogisticRegression

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
  ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
  ('clf', LogisticRegression(C=10, max_iter=2000))])
evaluate(model, texts, y)

model.fit(texts, y)
pred = model.predict(test_texts)               # (N_test,) labels
proba = model.predict_proba(test_texts)        # (N_test, K), columns in model.classes_ order
```
TF-IDF alone, when the matrix is needed:
```python
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X = vec.fit_transform(texts)                   # in: list of str -> out: sparse (N, V)
X_test = vec.transform(test_texts)             # same vocabulary, never fit again
print(X.shape, len(vec.get_feature_names_out()))
```
Scan C:
```python
for C in [1, 10, 100]:
  print('C =', C)
  evaluate(Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                     ('clf', LogisticRegression(C=C, max_iter=2000))]), texts, y)
```

-----

## 1b. Style classification: char n-grams, stacking, hand features

```python
model_char = Pipeline([
  ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
  ('clf', LogisticRegression(C=10, max_iter=2000))])
evaluate(model_char, texts, y)
```
Words + chars in one matrix:
```python
from scipy.sparse import hstack, csr_matrix
vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
vec_c = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True)
X = hstack([vec_w.fit_transform(texts), vec_c.fit_transform(texts)]).tocsr()
X_test = hstack([vec_w.transform(test_texts), vec_c.transform(test_texts)]).tocsr()
evaluate(LogisticRegression(C=10, max_iter=2000), X, y)
```
Hand features (in: list of str -> out: DataFrame (N, k)):
```python
def hand_features(texts):
  rows = []
  for t in texts:
    words = t.split()
    n_words = len(words)
    n_chars = len(t)
    avg_word = n_chars / max(n_words, 1)
    n_upper = 0; n_digit = 0; n_punct = 0
    for ch in t:
      if ch.isupper(): n_upper += 1
      if ch.isdigit(): n_digit += 1
      if ch in '.,;:!?': n_punct += 1
    n_unique = len(set(words))
    rows.append([n_words, n_chars, avg_word, n_upper / max(n_chars, 1), n_digit, n_punct,
                 t.count('!'), t.count('?'), n_unique / max(n_words, 1), t.count('$NE$')])
  return pd.DataFrame(rows, columns=['n_words', 'n_chars', 'avg_word', 'upper_share', 'n_digit',
                                     'n_punct', 'n_excl', 'n_quest', 'unique_share', 'n_ne'])

H = hand_features(texts); H_test = hand_features(test_texts)
```
Hand features alone with HGB, or stacked onto TF-IDF:
```python
from sklearn.ensemble import HistGradientBoostingClassifier
evaluate(HistGradientBoostingClassifier(random_state=0), H.values, y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_all = hstack([X, csr_matrix(scaler.fit_transform(H))]).tocsr()
X_all_test = hstack([X_test, csr_matrix(scaler.transform(H_test))]).tocsr()
evaluate(LogisticRegression(C=10, max_iter=2000), X_all, y)
```

-----

## 1c. Fake words: Word2Vec

```python
from gensim.models import Word2Vec

# in: list of token lists (train + test + unlabeled, no labels needed)
sentences = []
for t in texts + test_texts + unlabeled_texts:
  sentences.append(t.split())

w2v = Word2Vec(sentences, vector_size=100, window=3, min_count=2, sg=1, epochs=50, seed=0, workers=1)
# out: w2v.wv[word] -> vector (100,); word in w2v.wv -> known or not

def text_vector(t):
  total = np.zeros(100); count = 0
  for w in t.split():
    if w in w2v.wv:
      total = total + w2v.wv[w]; count += 1
  if count == 0: return total
  return total / count

X = np.array([text_vector(t) for t in texts])             # (N, 100)
X_test = np.array([text_vector(t) for t in test_texts])
evaluate(LogisticRegression(C=1, max_iter=2000), X, y)
```
Counting subtasks on tokens:
```python
counts = {}
for t in texts:
  for w in t.split():
    counts[w] = counts.get(w, 0) + 1
print('vocab size:', len(counts))
most = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
print(most)
```
Word2Vec extras: `w2v.wv.most_similar('token', topn=10)`; a word that behaves like a known one.

-----

## 1d. Clustering texts

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

vec = TfidfVectorizer(min_df=2, max_df=0.5, sublinear_tf=True)   # + stop_words='english' if English
V = vec.fit_transform(all_texts)                                 # sparse (N, V)
svd = TruncatedSVD(n_components=100, random_state=0)
S = svd.fit_transform(V)                                         # dense (N, 100)
S = Normalizer().fit_transform(S)                                # rows of length 1

for K in range(2, 13):
  labels = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(S)
  print(K, round(silhouette_score(S, labels), 4), np.bincount(labels))

K = 6
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(S)
labels = km.labels_                                              # (N,) ints 0..K-1

# read clusters
for c in range(K):
  print('===== cluster', c, 'size', (labels == c).sum())
  idx = np.where(labels == c)[0][:4]
  for i in idx: print(all_texts[i][:200]); print('-')

# top words per cluster
terms = vec.get_feature_names_out()
centers = svd.inverse_transform(km.cluster_centers_)             # (K, V)
for c in range(K):
  top = np.argsort(-centers[c])[:15]
  print(c, [terms[i] for i in top])

# confidence per text: gap between closest and second closest center
d = np.sort(km.transform(S), axis=1)
margin = d[:, 1] - d[:, 0]

# cluster id -> category name, by hand after reading
cluster_to_cat = {0: 'sports', 1: 'politics', 2: 'mixed', 3: 'tech', 4: 'sports', 5: 'economy'}
pred = [cluster_to_cat[c] for c in labels]
```
With sentence embeddings instead of TF-IDF: skip `vec`, `svd`, `Normalizer`; `S = E` from block 2a.

Zero-shot with category descriptions:
```python
descriptions = {'sports': 'a news article about sports, matches, players and teams',
                'politics': 'a news article about government, elections and politicians'}
cats = list(descriptions.keys())
D = st.encode([descriptions[c] for c in cats], normalize_embeddings=True)   # (K, d)
sims = E @ D.T                                                              # (N, K)
pred = [cats[i] for i in sims.argmax(axis=1)]
```
Keyword counts per category:
```python
keywords = {'sports': ['match', 'team', 'goal', 'player'], 'politics': ['minister', 'election', 'party']}
def keyword_category(t):
  t = t.lower(); best = None; best_count = 0
  for cat in keywords:
    count = 0
    for k in keywords[cat]: count += t.count(k)
    if count > best_count: best_count = count; best = cat
  return best            # None when nothing matched
```

-----

## 1e. Matching, search, duplicates

```python
# in: Q (n_q, d) and D (n_d, d), both rows of length 1
sims = Q @ D.T                        # (n_q, n_d) cosine
best = sims.argmax(axis=1)            # (n_q,) index of the best document per query
topk = np.argsort(-sims, axis=1)[:, :5]

# mutual best match
best_q_to_d = sims.argmax(axis=1)
best_d_to_q = sims.argmax(axis=0)
pairs = []
for q in range(sims.shape[0]):
  d = best_q_to_d[q]
  if best_d_to_q[d] == q: pairs.append((q, d))
print('mutual pairs:', len(pairs))

# top-1 accuracy on known pairs: known[q] = correct d
correct = 0
for q in range(len(known)):
  if best[q] == known[q]: correct += 1
print(correct / len(known))
```
TF-IDF cosine instead of embeddings:
```python
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(docs)
D = vec.transform(docs); Q = vec.transform(queries)
sims = (Q @ D.T).toarray()            # TF-IDF rows are already length 1
```
Pair classification (is this pair a duplicate?):
```python
# in: pairs as two lists a_texts, b_texts with labels y (1 duplicate, 0 not)
A = st.encode(a_texts, normalize_embeddings=True); B = st.encode(b_texts, normalize_embeddings=True)
cos = (A * B).sum(axis=1)                             # (N,)
F = np.concatenate([cos[:, None], np.abs(A - B), A * B], axis=1)   # (N, 1 + 2d)
evaluate(LogisticRegression(C=1, max_iter=2000), F, y)
```
Negatives when only positive pairs are given: pair each text with a random other text.
```python
rng = np.random.default_rng(0)
neg_a = []; neg_b = []
for i in range(len(a_texts)):
  j = rng.integers(len(b_texts))
  while j == i: j = rng.integers(len(b_texts))
  neg_a.append(a_texts[i]); neg_b.append(b_texts[j])
```
Relative representation (two embedding spaces that do not share axes):
```python
# anchors: indices of items known to correspond in both spaces
RA = A @ A[anchors].T                 # (n_a, n_anchors) similarity of each item to the anchors
RB = B @ B[anchors].T
RA = RA / np.linalg.norm(RA, axis=1, keepdims=True); RB = RB / np.linalg.norm(RB, axis=1, keepdims=True)
sims = RA @ RB.T                      # now comparable
```

-----

## 1f. Labels per token

```python
# in: sentences = list of token lists, tags = list of tag lists (same shapes)
rows = []; labels = []
for s in range(len(sentences)):
  words = sentences[s]
  for i in range(len(words)):
    w = words[i]
    prev = words[i - 1] if i > 0 else '<s>'
    nxt = words[i + 1] if i < len(words) - 1 else '</s>'
    rows.append({'w': w.lower(), 'prev': prev.lower(), 'next': nxt.lower(),
                 'suf2': w[-2:], 'suf3': w[-3:], 'cap': int(w[0].isupper()),
                 'digit': int(w.isdigit()), 'pos': i, 'len': len(w)})
    labels.append(tags[s][i])
F = pd.DataFrame(rows); y_tok = np.array(labels)

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer()
X = dv.fit_transform(F.to_dict('records'))     # one-hot of all string columns, sparse
evaluate(LogisticRegression(C=1, max_iter=2000), X, y_tok)
```
Dictionary baseline:
```python
tag_counts = {}
for s in range(len(sentences)):
  for i in range(len(sentences[s])):
    w = sentences[s][i].lower(); t = tags[s][i]
    if w not in tag_counts: tag_counts[w] = {}
    tag_counts[w][t] = tag_counts[w].get(t, 0) + 1
default_tag = pd.Series(y_tok).value_counts().index[0]
def predict_token(w):
  w = w.lower()
  if w not in tag_counts: return default_tag
  return max(tag_counts[w], key=tag_counts[w].get)
```
Transformer tokens, label alignment (first subword gets the label, the rest -100):
```python
enc = tok(words, is_split_into_words=True, truncation=True, max_length=128)
word_ids = enc.word_ids()
aligned = []; prev_word = None
for wid in word_ids:
  if wid is None or wid == prev_word: aligned.append(-100)
  else: aligned.append(tag_to_id[tags[wid]])
  prev_word = wid
```
Masked word position with a masked LM:
```python
from transformers import AutoModelForMaskedLM
mlm = AutoModelForMaskedLM.from_pretrained('bert-base-uncased').eval()
def score_position(words, pos, target):
  tmp = words[:pos] + ['[MASK]'] + words[pos:]
  enc = tok(' '.join(tmp), return_tensors='pt')
  mask_idx = (enc['input_ids'][0] == tok.mask_token_id).nonzero()[0, 0]
  with torch.no_grad(): logits = mlm(**enc).logits[0, mask_idx]
  return logits[tok.convert_tokens_to_ids(target)].item()
best_pos = max(range(len(words) + 1), key=lambda p: score_position(words, p, target))
```

-----

## 1g. Regression on text

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

model = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                  ('reg', Ridge(alpha=1.0))])          # alpha 0.3 / 1 / 3 / 10
kf = KFold(n_splits=5, shuffle=True, random_state=0)
scores = []
for fit_rows, val_rows in kf.split(texts):
  model.fit([texts[i] for i in fit_rows], y[fit_rows])
  pred = model.predict([texts[i] for i in val_rows])
  scores.append(mean_absolute_error(y[val_rows], pred))
print(np.round(scores, 4), np.mean(scores))
```
Target mean per word computed inside the fold (word level tasks):
```python
def add_word_mean(df_fit, df_other, col='word', target='time'):
  means = df_fit.groupby(col)[target].mean()
  global_mean = df_fit[target].mean()
  df_other = df_other.copy()
  df_other['word_mean'] = df_other[col].map(means).fillna(global_mean)
  return df_other
# for df_fit itself: leave one document out, or use the fold split so a row never sees its own target
```
GroupKFold when rows share a document:
```python
gkf = GroupKFold(n_splits=5)
for fit_rows, val_rows in gkf.split(X, y, groups=train['doc_id']): ...
```

-----

## 1h. Generate or correct text

Pretrained as is:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
NAME = 'google/flan-t5-base'
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(NAME).to(DEVICE).eval()

def generate(batch_texts, prefix='grammar: ', max_new_tokens=64):
  enc = tok([prefix + t for t in batch_texts], padding=True, truncation=True, max_length=128,
            return_tensors='pt').to(DEVICE)
  with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=4)
  return tok.batch_decode(out, skip_special_tokens=True)    # list of str

outputs = []
for start in range(0, len(test_texts), 32):
  outputs += generate(test_texts[start:start + 32])
```
Baseline: `outputs = list(test_texts)` (unchanged). Score both with the official metric on val.

Fine tune loop:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
model.train()
for epoch in range(3):
  perm = np.random.permutation(len(src)); total = 0
  for start in range(0, len(src), 16):
    idx = perm[start:start + 16]
    enc = tok(['grammar: ' + src[i] for i in idx], padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
    lab = tok([tgt[i] for i in idx], padding=True, truncation=True, max_length=128, return_tensors='pt')['input_ids'].to(DEVICE)
    lab[lab == tok.pad_token_id] = -100
    loss = model(**enc, labels=lab).loss
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    total += loss.item()
  print(epoch, total)
  # evaluate on val here, keep best_state
model.eval()
```
Safety rule after generation (change too big -> keep original):
```python
def word_diff(a, b):
  wa = a.split(); wb = b.split(); n = 0
  for i in range(max(len(wa), len(wb))):
    if i >= len(wa) or i >= len(wb) or wa[i] != wb[i]: n += 1
  return n
final = []
for i in range(len(test_texts)):
  final.append(outputs[i] if word_diff(test_texts[i], outputs[i]) <= 3 else test_texts[i])
```
Edit distance when the metric is character based: `import editdistance; editdistance.eval(a, b)` or a plain DP loop.

-----

## 1i. Given a model: probes, scoring, embedding arithmetic

Hidden states of a layer as features:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained(NAME, output_hidden_states=True).to(DEVICE).eval()

def layer_features(texts, layer):
  feats = []
  for start in range(0, len(texts), 32):
    enc = tok(texts[start:start + 32], padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
      hs = model(**enc).hidden_states          # tuple of (n_layers + 1) tensors, each (B, L, d)
    h = hs[layer]
    mask = enc['attention_mask'].unsqueeze(-1)
    mean = (h * mask).sum(1) / mask.sum(1)
    feats.append(mean.cpu().numpy())
  return np.concatenate(feats)

n_layers = len(model(**tok(['x'], return_tensors='pt').to(DEVICE)).hidden_states)
for layer in [n_layers // 2, n_layers - 1]:
  X = layer_features(texts, layer)
  print('layer', layer); evaluate(LogisticRegression(C=1, max_iter=2000), X, y)
```
Last token instead of mean (causal models): `h[torch.arange(h.shape[0]), enc['attention_mask'].sum(1) - 1]`.

Score a candidate continuation with a causal model:
```python
from transformers import AutoModelForCausalLM
lm = AutoModelForCausalLM.from_pretrained('gpt2').to(DEVICE).eval()
def logprob(prompt, candidate):
  ids_p = tok(prompt, return_tensors='pt')['input_ids'][0]
  ids_c = tok(' ' + candidate, return_tensors='pt')['input_ids'][0]
  ids = torch.cat([ids_p, ids_c]).unsqueeze(0).to(DEVICE)
  with torch.no_grad(): logits = lm(ids).logits[0]           # (L, vocab)
  logp = torch.log_softmax(logits, dim=-1)
  total = 0.0
  for i in range(len(ids_c)):
    pos = len(ids_p) + i - 1                                   # token at pos predicts token pos+1
    total += logp[pos, ids_c[i]].item()
  return total
best = max(candidates, key=lambda c: logprob(prompt, c))
```
Embedding arithmetic:
```python
mean0 = E[y == 0].mean(axis=0); mean1 = E[y == 1].mean(axis=0)
direction = mean1 - mean0; direction = direction / np.linalg.norm(direction)
proj = E @ direction                                 # (N,) position along the direction
E_debiased = E - np.outer(proj, direction)           # remove that direction from every row
pred = (np.linalg.norm(E - mean1, axis=1) < np.linalg.norm(E - mean0, axis=1)).astype(int)
```

-----

## 1j. Audio

```python
import torchaudio
def load_16k(path):
  wave, sr = torchaudio.load(path)                   # (channels, samples)
  if wave.shape[0] > 1: wave = wave.mean(dim=0, keepdim=True)
  if sr != 16000: wave = torchaudio.functional.resample(wave, sr, 16000)
  return wave[0]                                     # (samples,)

from transformers import AutoProcessor, AutoModel
proc = AutoProcessor.from_pretrained('facebook/hubert-base-ls960')
enc_model = AutoModel.from_pretrained('facebook/hubert-base-ls960').to(DEVICE).eval()
feats = []
for path in paths:
  wave = load_16k(path)
  inp = proc(wave.numpy(), sampling_rate=16000, return_tensors='pt').to(DEVICE)
  with torch.no_grad(): h = enc_model(**inp).last_hidden_state      # (1, T, d)
  feats.append(h.mean(dim=1)[0].cpu().numpy())
X = np.array(feats)                                  # (N, d)
evaluate(LogisticRegression(C=1, max_iter=2000), X, y)
```
Transcribe then treat as text:
```python
from transformers import pipeline
asr = pipeline('automatic-speech-recognition', model='openai/whisper-small', device=0)
transcripts = [asr(p)['text'] for p in paths]
```
Spectrogram as an image for a small CNN:
```python
mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=64)
spec = torch.log(mel(wave) + 1e-6)                   # (64, T); pad or crop T to a fixed length, then (1, 64, T)
```

-----

## 2a. Sentence embeddings

```python
from sentence_transformers import SentenceTransformer
st = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
E = st.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)   # in: list of str -> out: (N, 384)
E_test = st.encode(test_texts, normalize_embeddings=True, batch_size=128)
evaluate(LogisticRegression(C=1, max_iter=2000), E, y)
```
Without SentenceTransformer: block in NLP-methods.md section 2 (tokenizer + AutoModel + masked mean).

Chunked long texts:
```python
def chunks(t, size=300, overlap=50):
  words = t.split(); out = []
  start = 0
  while start < len(words):
    out.append(' '.join(words[start:start + size]))
    if start + size >= len(words): break
    start += size - overlap
  return out
E = []
for t in texts:
  parts = chunks(t)
  V = st.encode(parts, normalize_embeddings=True)
  v = V.mean(axis=0); E.append(v / np.linalg.norm(v))
E = np.array(E)
```

-----

## 2b. Fine tune a transformer for classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
NAME = 'distilbert-base-uncased'
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=len(classes)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

def predict_logits(texts_list):
  model.eval(); outs = []
  for start in range(0, len(texts_list), 64):
    enc = tok(texts_list[start:start + 64], padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
    with torch.no_grad(): outs.append(model(**enc).logits.cpu())
  return torch.cat(outs)                               # (N, K)

best_score = -1; best_state = None
for epoch in range(3):
  model.train(); perm = np.random.permutation(len(fit_texts))
  for start in range(0, len(fit_texts), 16):
    idx = perm[start:start + 16]
    enc = tok([fit_texts[i] for i in idx], padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
    labels = torch.tensor(y_fit[idx]).long().to(DEVICE)
    loss = model(**enc, labels=labels).loss
    optimizer.zero_grad(); loss.backward(); optimizer.step()
  pred = predict_logits(val_texts).argmax(1).numpy()
  score = f1_score(y_val, pred, average='macro'); print(epoch, round(score, 4))
  if score > best_score: best_score = score; best_state = copy.deepcopy(model.state_dict())
model.load_state_dict(best_state)
proba_test = torch.softmax(predict_logits(test_texts), dim=1).numpy()
```

-----

## 3. Combine, decide, submit

Average probabilities of two models (columns must be in the same class order):
```python
p1 = model_words.predict_proba(test_texts); p2 = model_chars.predict_proba(test_texts)
assert list(model_words.classes_) == list(model_chars.classes_)
p = (p1 + p2) / 2
pred = model_words.classes_[p.argmax(axis=1)]
```
Out-of-fold probabilities, to check the combination honestly:
```python
def oof_proba(model, texts, y):
  P = np.zeros((len(y), len(set(y))))
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
  for fit_rows, val_rows in skf.split(np.zeros(len(y)), y):
    model.fit([texts[i] for i in fit_rows], y[fit_rows])
    P[val_rows] = model.predict_proba([texts[i] for i in val_rows])
  return P
P1 = oof_proba(model_words, texts, y); P2 = oof_proba(model_chars, texts, y)
for w in [0, 0.3, 0.5, 0.7, 1]:
  pred = classes[(w * P1 + (1 - w) * P2).argmax(1)] if isinstance(classes, np.ndarray) else np.array(classes)[(w * P1 + (1 - w) * P2).argmax(1)]
  print(w, round(f1_score(y, pred, average='macro'), 4))
```
Threshold for a rare positive class:
```python
P = oof_proba(model, texts, y)[:, 1]
for th in [0.2, 0.3, 0.4, 0.5, 0.6]:
  print(th, round(f1_score(y, (P > th).astype(int)), 4))
```
Submission:
```python
sub = pd.DataFrame({'id': test['id'], 'label': pred})
print(sub.shape, sub['label'].value_counts(normalize=True).round(3))
sub.to_csv('submission.csv', index=False)
```
