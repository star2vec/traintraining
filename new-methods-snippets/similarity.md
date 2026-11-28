# Similarities

## pipeline

1. VECTORIZARE (de la imag, text, audio etc la vector)
=> Q(M, d) pt queries, C(N, d) pt candidates

2. SCORE TABLE (every query is compared with every candidate)

3. ANSWER (best, top-k, one-to-one pairs, nothing matches)

### COD PIPELINE

```python
import numpy as np
from sentence_transformers import SentenceTransformers

queries = list(test['query'])
cands = list(docs['text'])

st = SentenceTransformer(MODEL_PATH)
Q = st.encode(queries, normalize_embeddings=True)
C = st.encode(cands, normalize_embeddings=False)

S = Q @ C.T

best = S.argmax(axis=1)
```

#### Validare:

```python
nrc = 0
for i in range(len(truematch)):
  if best[i] == truematch[i]:
    nrc += 1
print('top 1 accurcay:', nrc/len(truematch))
```

--------

## STEP 1: vectorizare

### 1.A: word counts
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
vec.fit(queries+cands)
Q = vec.transform(queries)
C = vec.transform(cands)
S = (Q @ C.T).toarray() # aici e toarray pt ca tfidfvec da return la a sparse matrix
```
### 1.B: sentence model
```python
st = SentenceTransformers(MODEL_PATH)

Q = st.encode(queries, normalize_embeddings=True)
C = st.encode(cands, normalize_embeddings=True)

S = Q @ C.T
best = S.argmax(axis=1)
```

### 1.C: text cu any encoder

```python
import numpy as np
import torch
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE).eval()

def embed(texts):
    vectors = []
    for t in texts:
        enc = tok(t, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)
        # enc['input_ids']       (1, L)  the token numbers
        # enc['attention_mask']  (1, L)  all 1 here
        with torch.no_grad():
            out = model(**enc).last_hidden_state    # (1, L, d)  one vector per token
        vec = out[0].mean(dim=0)                     # (L, d) -> (d,)  one vector per text
        vectors.append(vec.cpu().numpy())            # (d,) as numpy
    E = np.array(vectors)                            # (N, d)  one row per text
    return normalize(E)                              # (N, d)  every row has length 1

E = embed(texts)
print(E.shape)                                       # check: (number of texts, d)
```

### 1.D: cu word2vec
```python
from gensim.models import Word2Vec
from sklearn.preprocesing import normalize

sentences = [t.split() for t in queries + candidates]
w2v = Word2Vec(sentences, vector_size=100, window=3, min_count=2, sg=1, epochs=50, seed=0, workers=1)

def texvec(t):
    vecs = []
    for w in t.split():
        if w in w2v.wv:
            vecs.append(w2v.wv[w])
    if len(vecs) == 0:
        return np.zeros(w2v.vector_size)
    return np.mean(vecs, axis=0)

Q = unit(np.array([texvec(t) for t in queries]))
C = unit(np.array([texvec(t) for t in candidates]))
```

### 1.E: image-text (CLIP)
```python
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import normalize

clip = CLIPModel.from_pretrained(MODEL_PATH).to(DEVICE).eval()
proc = CLIPProcessor.from_pretrained(MODEL_PATH)

def embimg(paths):
  vecs = []
  for p in paths:
    img = Image.open(p).convert('RGB')
    pix = proc(images=img, return_tensors='pt')['pixel_values'].to(DEVICE)
    with torch.no_grad():
      emb = clip.get_image_features(pixel_values=pix).pooler_output
    vecs.append(emb[0].cpu().numpy())
  return normalize(np.array(vecs))

def embtext(words):
  texts = ['a photo of a ' + w for w in words]
  t = proc.tokenizer(texts, padding=True, return_tensors='pt').to(DEVICE)
  with torch.no_grad():
    emb = clip.get_text_features(**t).pooler_output
  return normalize(emb.cpu().numpy())

Q = embimg(query_paths)
C = embimg(cand_paths)
# sau C = embtext(class_words)
```

### 1.F: small images
```python
from sklearn.preprocessing import normalize
Q = normalize(queryimgs.reshape(len(queryimgs), -1).astype(np.float32))      # (M, H*W)
C = normalize(templateimgs.reshape(len(templateimgs), -1).astype(np.float32))
```

* query_imgs has shape (M, H, W): M grayscale images, each H pixel rows by W pixel columns. For example, 5 images of 28×28 give (5, 28, 28). template_imgs is the same with N images: (N, H, W).

* len(query_imgs) is M, the number of images. -1 means "work this number out yourself": it becomes H×W, because the total number of values (M×H×W) can't change. So (5, 28, 28) becomes (5, 784). Each image is now one row of 784 numbers: its first pixel row, then its second, and so on.

* Pixels are usually stored as uint8, whole numbers from 0 to 255. Squaring them inside uint8 overflows (200² = 40000 doesn't fit, so it wraps around to a wrong number), and normalize squares the values. Converting to floats avoids that.

* TRAPS:
  -  White background. If the background is 255 and the ink is dark, almost every pixel is large in every image. All cosines then come out near 1 and the matching is mostly noise. Flip the image first, so ink is high and background is 0 (to_ink in bizantin does 1 - img / 255). Then only the ink counts.
  - Color images. They have shape (M, H, W, 3). The same reshape gives (M, H*W*3) and everything else works unchanged.

### 1.G: audio (one vector per clip)
```python
import torch, torchaudio
import numpy as np
from torchaudio.transforms import Spectrogram
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

spectf = Spectogram(n_fft=512, hop_length=256, power=1)
def clipvec(x):
  S = spectf(torch.tensor(x, dtype=torch.float32))
  logspec = torch.log(S + 1e-6).numpy()
  return np.concatenate([logspec.mean(axis=1), logspec.std(axis=1)])

F_all = np.array([clipvec(x) for x in querywaves + candwaves])
F_all = StandardScaler().fit_transform(F_all)
Q = normalize(F_all[:len(querywaves)])
C = normalize(F_all[len(querywaves):])
```

--------

## STEP 2: cum contruim S[i, j]

### 2.1: default
```python
S = Q @ C.T
```

### 2.2: pixel difference between images of the same size
```python
S = np.zeros((len(imgs), len(refs)))
for i in range(len(imgs)):
  for j in range(len(refs)):
    S[i, j] = -np.abs(imgs[i]-refs[j]).mean()
```

### 2.3: edit distance la strings (levenshtein)
```python
def levenshtein(a, b):
  d = [[0]*(len(b)+1) for _ in range(len(a)+1)]
  for i in range(len(a) + 1):
    d[i][0] = i          
  for j in range(len(b) + 1):
    d[0][j] = j    
  for i in range(1, len(a)+1):
    for j in range(1, len(b)+1):
      if a[i-1]==b[j-1]:
        cost = 0
      else:
        cost = 1
      d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
  return d[len(a)][len(b)]

S = np.zeros((len(words), len(vocab)))
for i in range(len(words)):
  for j in range(len(vocab)):
    S[i, j] = 1 - levenshtein(words[i], vocab[j]) / max(len(words[i]), len(vocab[j]), 1)
```

### 2.4: comparing vectors from 2 different models

```python
def relrepr(M, anchors):          # M: (n, d), one model's vectors, rows of length 1
                                  # anchors: list of k row numbers
    A = M[anchors]                # (k, d)
    R = []
    for i in range(len(M)):
        row = A @ M[i]            # (k,): cosine of item i to each anchor
        row = row - row.mean()    # (k,)
        row = row / np.sqrt(np.sum(row ** 2))   # (k,), length 1
        R.append(row)
    return np.array(R)            # (n, k)

Ra = relrepr(Ma, anchors_a)       # Ma: (n_a, d_a) -> Ra: (n_a, k)
Rb = relrepr(Mb, anchors_b)       # Mb: (n_b, d_b) -> Rb: (n_b, k); anchors_b[k] is the pair of anchors_a[k]
S = Ra @ Rb.T                     # (n_a, n_b)

```

Vectors from two different models can't be compared directly, but the relations between items carry over. An anchor is an item whose pair is already known, and each item is described by its similarity to every anchor on its own side. The anchors are the same items in the same order on both sides, so these descriptions line up and can be compared with cosine.

------

## STEP 3: CINE E THE WINNER

### 3.1: default (best, top-k)
```python
best = S.argmax(axis=1)

order = np.argsort(-S, axis=1)
top5 = order[:, :5]
```

### 3.2: multiple queries vote for one answer
```python
H = hint vectors of this question      # (h, d), rows of length 1
O = option vectors of this question    # (n, d), rows of length 1
S = H @ O.T                            # (h, n): one row per hint, one column per option

total = np.zeros(S.shape[1])           # one slot per option, all start at 0
for i in range(S.shape[0]):            # go through the hints
    total += S[i]                      # each hint adds its score to every option
ranked = np.argsort(-total)            # options from best to worst
```

### 3.3: class centers as candidates
```python
centers = []
for k in range(K):
  centers.append(Elab[ylab==k].mean(axis=0))
C = normalize(np.array(centers))
pred = (Q @ C.T).argmax(axis=1)
```

### 3.4 each candidate can be used once => pair

```python
pairs = []
for i in range(S.shape[0]):
  for j in range(S.shape[1]):
    pairs.append(S[i, j], i, j)

match = [-1] * S.shape[0]
used = [False] * S.shape[1]
for sc, i, j in pairs:
  if match[i]==-1:
    if not used[j]:
      match[i] = j
      used[j] = True
```
### 3.5 reliable pairs?? 

A pair is kept when each side picks the other. Such pairs can still be wrong, but far less often than plain argmax.

```python
bestc = S.argmax(axis=1) # for each query, its best candidate
bestq = S.argmax(axis=0) # for each candidate, its best query
moots = []
for i in range(S.shape[0]):
  j = bestc[i]
  if bestq[j] == i:
    moots.append((i, j))
```

### 3.6 (nu full metoda, ceva auxiliar, dar!!) sa verificam marginul

e useful sa vedem cat de useful suntem ca asta e cel mai bun (poate sunt 2/+ ft close idk)

```python
srt = np.sorted(S, axis=1)
margin = str[:-1] - str[:-2]
```



------

## TRAPS:

* Forgetting to normalize before @: long vectors win regardless of direction.
* A distance in S without the minus sign: argmax then returns the least similar item.
argmax giving one candidate to two queries when the statement says pairs are one to one (block 3e).
* A query that is also in the candidate list matches itself: set S[i, i] = -1 first (roboti-vorbareti).
* Vectors from two different models compared directly (block 2e).
* Averaging token vectors without the mask (block 1c).
* TF-IDF fit on queries only: candidate words missing from the vocabulary. Fit on both.
* A threshold chosen on the test scores alone, with no known valid items to compare against.
