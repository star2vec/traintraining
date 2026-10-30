# setari-model.md — which model, when, with what settings

Built from the notebooks in this repo (what actually worked) plus a few baselines worth knowing.
Rule: first model from the table, submit, only then the second one.

-----

## 0. Order of work (any problem)

1. Read the whole statement. Note: exact submission format (column names), metric, points per subtask.
2. "Counting" subtasks (means, nunique, how many rows with...) are computed on the RAW data and saved right away.
3. Dummy -> submit. A valid row on the leaderboard confirms the format.
4. First model from the table -> submit.
5. Only now: tuning, second model, combinations.

!! an easy subtask done well > a hard subtask done halfway

-----

## 1. Quick table

| data | first model | second | validation | typical metric |
|---|---|---|---|---|
| table, regression | HistGradientBoostingRegressor | CatBoostRegressor | KFold(5, shuffle) | MAE / MSE / R2 |
| table, classification | HistGradientBoostingClassifier | CatBoostClassifier / RF | StratifiedKFold(5) | accuracy / f1 |
| table with groups (Customer_ID, text id) | HGB | CatBoost | GroupKFold(5) | same |
| text, classify by content | TF-IDF words (1,2) + LogisticRegression | sentence embeddings + LR | StratifiedKFold | f1 macro / weighted |
| text, classify by style / author / dialect / AI vs human | TF-IDF char_wb (2,5) + LR | + words, combined | StratifiedKFold | f1 |
| text, "secret" or scrambled words / lots of unlabeled text | Word2Vec on all text + LR | TF-IDF | StratifiedKFold | f1 |
| text, grouping without labels | TF-IDF -> TruncatedSVD -> Normalizer -> KMeans | embeddings -> KMeans | silhouette + reading clusters | - |
| text, search / matching | sentence embeddings, cosine | TF-IDF cosine | top-1 accuracy on known pairs | - |
| small images (28x28, 48x48), few | flatten + KNN / LogisticRegression | small CNN | StratifiedKFold / split | accuracy |
| small images, many, parameter budget | small CNN from scratch | - | 80/20 stratified split | accuracy |
| real photos | pretrained resnet, fine tune | resnet features + LR | 80/20 stratified split | accuracy / f1 |
| images, "is something odd in the picture?" no labels | autoencoder / distance to reference + threshold | IsolationForest on features | histogram of scores | - |
| synthetic images with rules (staff lines, characters, symbols) | classic processing: threshold, measure.label, template | KNN on crops | look at the output | - |
| embeddings given directly (N, d) | nearest centroid / LR | KMeans | KFold | accuracy |
| MDP / RL with dynamics from logs | value iteration | tabular Q-learning | - | - |

-----

## 2. Tabular

### Dummy (always first)
```python
DummyRegressor(strategy='median')        # for MAE
DummyRegressor(strategy='mean')          # for MSE
DummyClassifier(strategy='most_frequent')
```

### HistGradientBoosting (default for tables)
- NaN is fine, no scaling needed, text columns NOT (make them numeric first).
- Regression: `loss='absolute_error'` when the metric is MAE, otherwise `squared_error`.
```python
HistGradientBoostingRegressor(loss='absolute_error', random_state=0)
HistGradientBoostingClassifier(random_state=0)
HistGradientBoostingClassifier(class_weight='balanced', random_state=0)   # rare classes
```
Settings, in the order to touch them:
| parameter | default | try | what it does |
|---|---|---|---|
| learning_rate | 0.1 | 0.05, 0.2 | step of each tree; smaller => more trees needed |
| max_iter | 100 | 300, 500 | number of trees; raise it when lowering learning_rate |
| max_leaf_nodes | 31 | 15, 63 | complexity of one tree |
| min_samples_leaf | 20 | 5, 50 | larger => simpler, against overfit |
| l2_regularization | 0 | 1.0 | against overfit |

Combinations worth trying: `(lr=0.05, max_iter=300)`, `(lr=0.05, max_iter=500, max_leaf_nodes=15)`.

### CatBoost (second for tables; first with many categorical columns)
- Takes text columns directly through `cat_features`, NaN fine.
- Worked on eye-tracking (regression with `cat_features=['genre']`) and box-avalanche (classification on box features).
```python
CatBoostRegressor(iterations=300, verbose=0, random_seed=0, cat_features=['genre'])
CatBoostClassifier(iterations=300, verbose=0, random_seed=0)
```
| parameter | default | try |
|---|---|---|
| iterations | 1000 | 300 (fast), 1000 |
| learning_rate | auto | 0.03, 0.1 |
| depth | 6 | 4, 8 |
| loss_function | RMSE / Logloss | 'MAE' for MAE |

!! `cat_features` takes column names; values in them must be strings, not floats with NaN (`astype(str)`).

### RandomForest
- No NaN (fill with median), no scaling. More stable than HGB on very little data.
```python
RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=0)
```
Try: `max_depth` 10 / 20, `min_samples_leaf` 1 / 5, `max_features` 'sqrt' / 0.5.

### LogisticRegression / LinearRegression (on tables)
- No NaN + scaling (Pipeline). Long-tailed columns (income, like counts): `np.log1p` first.
```python
Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(max_iter=1000))])
Pipeline([('scale', StandardScaler()), ('model', LinearRegression())])
```
- `C`: 1 default; 0.1 (simpler) and 10 (freer). `class_weight='balanced'` for rare classes.
- On tables these are baselines, rarely win. On text and on embeddings they are usually the final model.

### KNN
- On pixels of small images or on normalized embeddings. Scaling required on tables.
```python
KNeighborsClassifier(n_neighbors=3)              # try 1, 3, 5, 11
KNeighborsClassifier(n_neighbors=5, weights='distance')
```

### How much tuning fits in the time
- 1 parameter: 3 values x 5 folds = 15 trainings. Fine for anything.
- 2 parameters: 2x2 or 3x2. Fast models only.
- Order for HGB: loss -> learning_rate + max_iter -> max_leaf_nodes -> min_samples_leaf.
- Gain smaller than the fold spread = no change, keep the default.

-----

## 3. Text

### TF-IDF + LogisticRegression (default)
```python
Pipeline([
  ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
  ('clf', LogisticRegression(C=10, max_iter=2000))])
```
TfidfVectorizer settings:
| parameter | what it does | when |
|---|---|---|
| ngram_range=(1,2) | words + word pairs | almost always a small plus |
| sublinear_tf=True | 1+log(count) instead of count | almost always a plus |
| min_df=2 | drop words that appear in a single text | many texts |
| max_df=0.5 / 0.9 | drop words that appear everywhere | clustering, topics |
| analyzer='char_wb', ngram_range=(2,5) | letter chunks inside words | style, author, dialect, AI vs human, spelling errors |
| lowercase=False | keep capitals | when capitals carry signal (style) |
| stop_words='english' | drop common words | English only, clustering only |
| max_features=50000 | cap the vocabulary | if slow / memory |

LogisticRegression on TF-IDF: `C=10` is usually best (mickey_donald also gave 30, 100). Try `[1, 10, 100]`. `class_weight='balanced'` for unbalanced classes.

Words or letters? Content (topic, category) -> words. Style (who wrote it, which dialect, human or AI) -> char_wb. When unclear, evaluate both: 2 trainings.

### TF-IDF + LinearSVC
- Same vectorizer, usually slightly better than LR with many classes, but no `predict_proba`.
```python
from sklearn.svm import LinearSVC
LinearSVC(C=1.0)                         # try 0.3, 1, 3
```

### Sentence embeddings + LogisticRegression / cosine
- When meaning matters more than exact words; few texts; similarity is needed.
- Models that worked: `all-MiniLM-L6-v2` (fast, en), `BAAI/bge-small-en-v1.5`. In the contest: only the approved list.
```python
st = SentenceTransformer(MODEL_NAME, device=DEVICE)
E = st.encode(list(texts), normalize_embeddings=True, batch_size=128)   # (N, d), length 1
LogisticRegression(C=1, max_iter=2000).fit(E, y)                          # C 0.1 / 1 / 10
sims = Q @ E.T                                                            # cosine, for search
```
- Without SentenceTransformer, with AutoModel: the [CLS] token vector `out[0][:, 0]`, or the masked mean over tokens. Then normalize to length 1.
- Long texts (over ~300 words): chunking + mean of the chunks (see README).

### Word2Vec (gensim)
- When the words are not real words (codes, scrambled) or there is a lot of unlabeled text: train on train + test + unlabeled, then mean vector per text + LR.
```python
Word2Vec(sentences, vector_size=100, window=3, min_count=2, sg=1, epochs=50, seed=0, workers=1)
```
- `sg=1` (skip-gram) is better on little data. `window` 3 for syntax, 5+ for topic. `workers=1` for reproducibility.

### Text clustering (no labels)
```python
vec = TfidfVectorizer(min_df=2, max_df=0.5, sublinear_tf=True)      # (+ stop_words for English)
V = vec.fit_transform(texts)
S = TruncatedSVD(n_components=100, random_state=0).fit_transform(V) # 50..200
S = Normalizer().fit_transform(S)                                   # length 1 => KMeans becomes cosine
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(S)
```
- Choosing K: `silhouette_score(S, labels)` for K in 2..12, plus read 4 texts per cluster. A K 1..2 above the number of required categories leaves room for "mixed" clusters (mapped to categories by hand).
- Naming clusters: `centers = svd.inverse_transform(km.cluster_centers_)`, top 15 words via `vec.get_feature_names_out()`.
- On SentenceTransformer embeddings: KMeans directly (already normalized), no SVD.
- Confidence of a point: `d = np.sort(km.transform(S), axis=1); margin = d[:, 1] - d[:, 0]` (small margin = unsure).

### Zero-shot with embeddings (no labeled row at all)
- One descriptive sentence per category, encode it, argmax on cosine. Worked on algorel.
- Combined with keyword counts per category: keywords decide when the cluster is unsure.

### CLIP (image + text)
```python
img_emb = model.get_image_features(pixel_values=pix).pooler_output
txt_emb = model.get_text_features(**tok).pooler_output
# normalize both, then score = img_emb @ txt_emb.T
```
- Prompts like "a photo of a {word}" beat the bare word. Horizontal flip + average = free TTA.
- CLIP fine tune: `lr=1e-6`, 3..5 epochs, keep the state with the best val score (zero-shot is the score to beat).

### Transformer fine tune for classification (GPU and over ~2000 texts)
- Small model (`distilbert`, `bert-base`, or whatever is on the list), `max_length=128/256`, `batch_size=16/32`, `lr=2e-5`, 2..3 epochs, AdamW, `weight_decay=0.01`.
- Validate after each epoch, keep the best state. If TF-IDF+LR is within 1..2 points, fine tuning rarely pays off in contest time.

### Seq2seq (text correction, small translation)
- `flan-t5-small/base` with a task prefix ("grammar: "), `lr=3e-4`, 3..4 epochs, `batch_size=16`, `max_length=128`.
- Pad in labels -> `-100`. After generation, keep the original text when the change is too large (threshold chosen on val).

-----

## 4. Images

### Flatten + KNN / LogisticRegression
- Small, centered images, same background (symbols, digits). Pixels in [0, 1], `X.shape = (N, H*W)`.
- Center and resize every crop to the same size (square, then resize to 28..48) before flattening.
```python
KNeighborsClassifier(n_neighbors=3)
LogisticRegression(C=0.1, max_iter=2000)      # on pixels a small C is better
```

### Small CNN from scratch
- Small images (28..64 px), many (over ~2000), or a parameter budget in the score.
```python
class SmallCNN(nn.Module):
  def __init__(self, in_ch, n_classes, c1=16, c2=32, c3=64):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(in_ch, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
      nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(), nn.MaxPool2d(2),      # 14 -> 7
      nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(),
      nn.AdaptiveAvgPool2d(1),      # (N, c3, 7, 7) -> (N, c3, 1, 1), works for any input size
      nn.Flatten(),                 # (N, c3)
      nn.Dropout(0.2),
      nn.Linear(c3, n_classes))
  def forward(self, x):
    return self.net(x)
```
What the layers mean:
- `Conv2d(in, out, 3, padding=1)`: `out` filters of 3x3; padding=1 keeps H and W. Parameters = in*out*9 + out.
- `MaxPool2d(2)`: halves H and W. Two pools: 28 -> 7.
- `BatchNorm2d`: stabilizes training; allows a higher lr. Under a strict parameter budget drop it (costs 2*out).
- `AdaptiveAvgPool2d(1)` + `Linear(c3, n_classes)`: few parameters. Alternative `Flatten` + `Linear(c3*7*7, n)`: more parameters, sometimes more accurate on little data.
- Channels: 8/16/32 for a small budget (under ~10k parameters), 16/32/64 standard, 32/64/128 with lots of data.
- Parameter count: `sum(p.numel() for p in model.parameters())`.
- Last layer: `n_classes` outputs + `CrossEntropyLoss` (no softmax in the model). Binary: 1 output + `BCEWithLogitsLoss`.

Training: Adam `lr=1e-3`, `batch_size=64`, 10..30 epochs, augmentation = 2 px shift (`torch.roll`) and flip only if the label stays valid. See section 7.

### Pretrained resnet, fine tune
- Real photos, any size. `resnet18` (fast) or `resnet34`; `SIZE` 128 (fast) .. 224 (standard).
```python
model = torchvision.models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, n_classes)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
```
- `lr=1e-4` for the whole model; 5..15 epochs; `batch_size` 32; ImageNet normalization; augmentation flip + small rotation + `ColorJitter(0.2, 0.2)`.
- Several outputs (class + box + count) = one `fc` with several outputs and a weighted sum of losses (ducks: `bce + 5*l1_box + giou + 10*l1_fill`). The box is predicted normalized to [0, 1] through sigmoid.
- `ReduceLROnPlateau(mode='max', factor=0.3, patience=2)` on the val score + early stopping with `patience=4`.
- TTA: prediction on the image + on its flip, averaged.

### Features from pretrained + LogisticRegression (fastest win on photos)
- Remove `fc` (`model.fc = nn.Identity()`), pass all images through without grad -> `(N, 512)`, then `LogisticRegression(C=1, max_iter=2000)` or KNN. Two minutes, usually 90% of the fine tune score.

### Image anomalies (no labels)
- Autoencoder given or trained: reconstruction error `((rec - img)**2).mean()` per image -> histogram with `bins=500` + percentiles -> threshold at the "elbow" between the two modes.
- Distance to references (templates): `np.abs(refs - img).mean(axis=(1,2)).min()`; threshold chosen by comparing the histogram on known valid examples vs suspects.
- On tables/embeddings: `IsolationForest(contamination=0.05, random_state=0)`, score = `score_samples`.

### Classic processing (synthetic images with rules)
- Threshold: `mask = gray < 128` (black ink) or `> 0.5` on [0,1].
- Components: `measure.label(mask)` + `measure.regionprops(labels)` -> `r.area`, `r.bbox`, `r.centroid`. Filter `area >= min_area`.
- Projections: `mask.any(axis=0)` = columns with ink (splits symbols horizontally); `mask.mean(axis=1) > 0.5` = long horizontal lines (staff).
- Rotation: `scipy.ndimage.rotate(img, angle, reshape=False)`; search the angle with minimum error over `range(-30, 31)`.
- Other tools: `filters.threshold_otsu(gray)`, `ndimage.binary_opening/closing` for noise, `cv2.findContours`.

### Detection: cleaning boxes
- Hand-written NMS (IoU + sort by score). The IoU threshold is chosen by evaluating on train with the official metric.
- Second step: a classifier (CatBoost) on per-box features (score, rank, size, how many boxes support it, overlap with better boxes) -> keep boxes with `proba >= threshold`; `GroupKFold` by image.

-----

## 5. Clustering, anomalies, no labels (general)

### KMeans
```python
KMeans(n_clusters=K, n_init=10, random_state=0)    # n_init 10..50; on little data 50
```
- Scale first on tables (`StandardScaler`); on text/embeddings normalize to length 1.
- K: given by the problem > silhouette > inertia elbow. Check `np.bincount(labels)`: one huge cluster + many tiny ones = K too large or unscaled data.
- Cluster labels have no fixed meaning: map them to categories by hand, by reading or looking at examples.
- `km.transform(X)` = distances to centers; `km.cluster_centers_`.

### DBSCAN (clusters not round, or "noise" is wanted)
```python
from sklearn.cluster import DBSCAN
DBSCAN(eps=0.5, min_samples=5).fit_predict(X)      # -1 = noise
```
- `min_samples` = 2 * number of dimensions (at least 5). `eps`: distance to the `min_samples`-th neighbor (`NearestNeighbors`), sorted and plotted; `eps` at the elbow. On normalized embeddings `metric='cosine'`, eps 0.1..0.3.

### Nearest centroid (labeled embeddings, baseline in 3 lines)
```python
mean0 = X[y == 0].mean(axis=0); mean1 = X[y == 1].mean(axis=0)
pred = (np.linalg.norm(E - mean1, axis=1) < np.linalg.norm(E - mean0, axis=1)).astype(int)
```

### Dimensionality reduction
- `PCA(n_components=2)` for plotting; `PCA(50)` before KNN/KMeans on pixels. On a sparse matrix (TF-IDF): `TruncatedSVD`.

### Thresholds for anomalies / decisions
- Chosen on a histogram of the scores, or on validation by maximizing the metric, never "0.5 because". Rare classes: scan thresholds `[0.2 .. 0.6]` on probabilities.

-----

## 6. RL / MDP

### Value iteration (dynamics known or recovered from logs)
```python
V = np.zeros(N)
while True:
  Q = rew + GAMMA * V[nxt]      # (N, A): immediate reward + value of the next state
  new_V = Q.max(axis=1)
  new_V[end] = 0                # terminal states are worth 0
  if np.abs(new_V - V).max() < 1e-6: break
  V = new_V
policy = Q.argmax(axis=1)
```
- From logs `(s, a, r, s2)`: count `(a, s2 - s)` to learn what each action does; random transitions = weighted average of what was observed.
- `GAMMA` 0.9..0.99 from the statement.

### Tabular Q-learning (no dynamics)
```python
Q = np.zeros((N, A))
for ep in range(EPISODES):
  s = start
  while not done:
    a = rng.integers(A) if rng.random() < eps else Q[s].argmax()
    s2, r, done = step(s, a)
    Q[s, a] += alpha * (r + GAMMA * Q[s2].max() * (not done) - Q[s, a])
    s = s2
```
- `alpha=0.1`, `eps` 0.1 (or decaying from 1 to 0.05), 1000..10000 episodes.

### Bandits
- Epsilon greedy (`eps=0.1`) or UCB: `mean + c * sqrt(log(t) / count)`, `c=1..2`. Pull each arm once at the start.

-----

## 7. Training settings for networks

| situation | optimizer | lr | epochs | batch | weight_decay |
|---|---|---|---|---|---|
| small CNN from scratch | Adam | 1e-3 | 10..30 | 64 | 0 or 1e-4 |
| MLP on table / embeddings | Adam | 1e-3 | 20..50 | 64..256 | 1e-4 |
| pretrained resnet, fine tune all | AdamW | 1e-4 | 5..15 | 32 | 1e-4 |
| pretrained resnet, `fc` only | Adam | 1e-3 | 5..10 | 64 | 0 |
| text transformer, fine tune | AdamW | 2e-5 | 2..3 | 16..32 | 0.01 |
| t5 small / base, seq2seq | AdamW | 3e-4 | 3..4 | 16 | 0 |
| CLIP fine tune | AdamW | 1e-6 | 3..5 | 1..8 | 0 |

Reading the training run:
- loss does not drop at all in 2..3 epochs -> lr 10x smaller or larger; check `optimizer.zero_grad()` is in the loop, labels are `long` for CrossEntropy, input is normalized.
- loss jumps / becomes NaN -> lr too high; `clip_grad_norm_(model.parameters(), 1.0)`.
- fit acc >> val acc -> overfit: augmentation, dropout 0.2..0.5, weight_decay, smaller model, fewer epochs.
- fit acc low and val acc low -> model too small or too few epochs.
- Always: `best_state = copy.deepcopy(model.state_dict())` at the best val score, `patience` 3..5 epochs, at the end `load_state_dict(best_state)`.
- Scheduler, simple and enough: `ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=2)`, or `CosineAnnealingLR(optimizer, T_max=EPOCHS)`.
- After choosing the epoch count on val, retrain on all of train with the same number of epochs (ducks: 16).
- Seed at the top: `torch.manual_seed(0); np.random.seed(0)`.

Loss by target type:
| target | last layer | loss | prediction |
|---|---|---|---|
| one class of K | `Linear(.., K)` | `CrossEntropyLoss` | `out.argmax(1)` |
| yes / no | `Linear(.., 1)` | `BCEWithLogitsLoss` | `sigmoid(out) > threshold` |
| several labels per row | `Linear(.., K)` | `BCEWithLogitsLoss` | `sigmoid(out) > 0.5` |
| real number | `Linear(.., 1)` | `L1Loss` (MAE) / `MSELoss` | `out` |
| number in [0, 1] (box, fraction) | `Linear(.., k)` | `L1Loss` on `sigmoid(out)` | `sigmoid(out)` |
| unbalanced classes | same | `CrossEntropyLoss(weight=w)`, w = 1/frequency | threshold on proba |

-----

## 8. Combining models

- Average of probabilities (safest): `p = (p1 + p2) / 2`, then argmax. Models need `predict_proba`. Works well across families (TF-IDF words + TF-IDF chars, HGB + CatBoost, resnet + features+LR).
- Majority vote on 3+ models with `predict`: `np.apply_along_axis(lambda r: np.bincount(r).argmax(), 1, P)`; pointless with 2 models.
- Weighted: weights proportional to val score, or scan `w in [0.3, 0.5, 0.7]` on val.
- Per-model confidence: when models disagree, take the prediction of the model with the larger margin (algorel: rank of the margin within its own method). A cluster known to be "mixed" gets confidence 0.
- Rule: combine only if on val the average beats each model alone; compare on the same folds (out-of-fold).
- TTA on images (flip) and averaging 2..3 seeds are free combinations.
- Stacking (an LR on top of the models' out-of-fold probabilities) only with 30 minutes to spare and solid validation.

-----

## 9. Final checks (before submit)

- Same `clean_input` on train and test; `list(X.columns) == list(test.columns)`.
- Final model trained on ALL of train, with the settings chosen on val.
- `pd.Series(preds).value_counts(normalize=True)` looks like the train distribution (or like what is expected)?
- Submission row count = required count, per subtask, columns exactly as in the statement; `index=False`.
- One `print(submission.head())` and `submission.shape`.
