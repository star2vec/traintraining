# unsupervised.md — clustering, dimensionality reduction, anomalies

No labels, or labels only for a few rows. Input is always a dense matrix `X` of shape `(N, d)` (or sparse TF-IDF, see section 2). Output is a cluster id per row, a lower dimensional `X`, or an anomaly score per row.

Imports:
```python
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
```

-----

## 0. Which tool

| task | first | second |
|---|---|---|
| group rows into K groups, K known | KMeans | GaussianMixture |
| group rows, K unknown | KMeans + silhouette over a range | DBSCAN |
| groups of odd shapes, or "noise" points wanted | DBSCAN | AgglomerativeClustering |
| few rows (under ~2000), want a tree of merges | AgglomerativeClustering | - |
| shrink columns before distance methods | PCA (dense) / TruncatedSVD (sparse) | - |
| plot in 2D | PCA(2) | TSNE (slow, plot only) |
| "which rows are unusual" | IsolationForest | distance to nearest neighbors |
| "which rows don't belong to this reference set" | distance to the reference's centroid / nearest reference | autoencoder error |
| find the soft structure in embeddings | PCA explained variance, then look at the components | - |

Before any of them: **scale** (tables) or **normalize to length 1** (embeddings, TF-IDF after SVD). Distance methods are ruined by one column with large values.

-----

## 1. Preparing X

Table: scale every column to mean 0, std 1.
```python
num = df.select_dtypes('number').fillna(df.median(numeric_only=True))
X = StandardScaler().fit_transform(num.values)      # (N, d) float, each column mean 0 std 1
```
Embeddings: each row to length 1, so Euclidean distance = cosine distance.
```python
X = E / np.linalg.norm(E, axis=1, keepdims=True)    # (N, d)
```
Images: flatten to `(N, H*W)`, values in [0, 1], then PCA(50) before clustering.
```python
X = images.reshape(len(images), -1) / 255.0
X = PCA(n_components=50, random_state=0).fit_transform(X)
```
TF-IDF: sparse `(N, V)` -> TruncatedSVD(100) -> Normalizer (section 2).

Long tailed columns (counts, prices): `np.log1p` before scaling. Categorical columns: one-hot (`pd.get_dummies`) or drop them; KMeans on raw category codes is meaningless.

-----

## 2. Dimensionality reduction

### PCA (dense input)
In: `(N, d)`. Out: `(N, k)`, columns ordered by how much variance they explain.
```python
pca = PCA(n_components=50, random_state=0)
Z = pca.fit_transform(X)                            # (N, 50)
print(pca.explained_variance_ratio_[:10])           # share of variance per component
print(np.cumsum(pca.explained_variance_ratio_))     # how many components reach 0.9 / 0.95
pca.components_                                     # (50, d): each component as a direction in the original columns
pca.inverse_transform(Z)                            # (N, d): back to the original space (lossy)
```
Choosing k: the number of components where cumulative variance reaches ~0.9, or simply 2 for plots, 50 for KNN/KMeans on pixels. `PCA(n_components=0.95)` picks k automatically for 95% variance.

Reading a component: `np.argsort(-np.abs(pca.components_[0]))[:10]` = the original columns that weigh most in the first component.

### TruncatedSVD (sparse input, TF-IDF)
Same idea, works without converting to dense. Output is not centered and not normalized.
```python
svd = TruncatedSVD(n_components=100, random_state=0)
Z = svd.fit_transform(V)                            # V sparse (N, Vocab) -> Z dense (N, 100)
Z = Normalizer().fit_transform(Z)                   # rows to length 1, needed before KMeans
svd.inverse_transform(centers)                      # (K, Vocab): cluster centers back in word space
```

### When reduction hurts
Linear models on sparse text (LogisticRegression, LinearSVC): no reduction, they handle the raw matrix. Tree models on tables: no reduction, they pick columns themselves. Reduction is for distances, plots, and networks with a huge input layer.

### 2D for plotting only
```python
P = PCA(n_components=2).fit_transform(X)
plt.scatter(P[:, 0], P[:, 1], c=labels, s=3, cmap='tab10'); plt.show()
```
`TSNE(n_components=2, random_state=0).fit_transform(X)` gives prettier clusters but is slow above ~10k rows and its coordinates mean nothing outside the plot; never cluster on t-SNE output for a submission.

-----

## 3. KMeans

In: `X` `(N, d)` scaled/normalized, K. Out: `labels` `(N,)` ints 0..K-1, `cluster_centers_` `(K, d)`.
```python
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
labels = km.labels_
print(np.bincount(labels))                          # cluster sizes
D = km.transform(X)                                 # (N, K) distance of each row to each center
new_labels = km.predict(X_test)                     # assign new rows to the nearest center
km.inertia_                                         # sum of squared distances to own center
```
Settings: `n_init=10` default (50 on small data for a stable result), `random_state` fixed. K is the only real decision.

### Choosing K
```python
for K in range(2, 13):
  km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
  print(K, round(silhouette_score(X, km.labels_), 3), round(km.inertia_), np.bincount(km.labels_))
```
- Silhouette in [-1, 1]: higher = rows closer to their own cluster than to the next one. Pick the K with the highest value, but a K given by the problem wins, and a K that produces one giant cluster plus crumbs is wrong even with a good score.
- Inertia always goes down with K; look for the elbow where it stops dropping fast.
- Silhouette on more than ~10k rows is slow: `silhouette_score(X, labels, sample_size=5000, random_state=0)`.

### What the labels mean
Nothing by themselves. Cluster 0 in one run is cluster 3 in another. After the final fit, read examples per cluster and build the mapping by hand:
```python
for k in range(K):
  idx = np.where(labels == k)[0]
  print('cluster', k, len(idx))
  for i in idx[:5]: print('   ', texts[i][:120])      # or show images, or print df.iloc[i]
cluster_to_cat = {0: 'A', 1: 'C', 2: 'A', 3: 'B'}
pred = [cluster_to_cat[c] for c in labels]
```
Tables: `df.groupby(labels).mean(numeric_only=True)` shows what each cluster is about.

### When a few labels exist
Map each cluster to the majority label among its labeled rows:
```python
cluster_to_cat = {}
for k in range(K):
  known = y_known[(labels_known == k)]            # labels of the labeled rows in cluster k
  if len(known) > 0: cluster_to_cat[k] = pd.Series(known).value_counts().index[0]
```
Alternatively: nearest centroid built from the labeled rows (section 6), often better than clustering when 20+ labels per class exist.

### Confidence per row
```python
D_sorted = np.sort(km.transform(X), axis=1)
margin = D_sorted[:, 1] - D_sorted[:, 0]            # small = between two clusters, unsure
```

-----

## 4. DBSCAN

Groups rows that sit in dense regions; rows in sparse regions become noise (label -1). No K; instead `eps` (neighborhood radius) and `min_samples` (how many neighbors make a dense point).
In: `X` `(N, d)` scaled. Out: `labels` `(N,)` with -1 for noise.
```python
db = DBSCAN(eps=0.5, min_samples=5).fit(X)
labels = db.labels_
print('clusters:', labels.max() + 1, 'noise:', (labels == -1).sum())
```
Choosing the parameters:
- `min_samples`: 2 * d, at least 5; on embeddings 5..10.
- `eps`: distance to the `min_samples`-th nearest neighbor for every row, sorted; `eps` at the elbow of that curve.
```python
nn = NearestNeighbors(n_neighbors=5).fit(X)
dist, _ = nn.kneighbors(X)                          # (N, 5) distances, column 0 is the row itself (0)
kth = np.sort(dist[:, -1])                          # distance to the 5th neighbor, sorted
plt.plot(kth); plt.show()                           # eps at the elbow
```
On normalized embeddings: `DBSCAN(eps=0.2, min_samples=5, metric='cosine')`, eps in 0.1..0.4.

When to prefer it over KMeans: clusters of different sizes or shapes (rings, lines), a "find the noise / outliers" subtask, or when nobody knows K. When not: clusters with very different densities (one eps can't fit both), or more than ~50k rows in high dimension (slow).

Rows with label -1 are a free anomaly detector.

-----

## 5. Other clusterers

### AgglomerativeClustering
Merges the two closest clusters repeatedly. Deterministic, no `random_state`, and `linkage` controls the shape.
```python
ag = AgglomerativeClustering(n_clusters=K, linkage='ward')        # ward = KMeans-like compact clusters
labels = ag.fit_predict(X)
ag2 = AgglomerativeClustering(n_clusters=None, distance_threshold=3.0, linkage='average')  # cut by distance instead of K
```
Slow above ~10k rows. Useful on small sets and when a threshold ("texts closer than 0.3 are the same") is more natural than K. `metric='cosine'` with `linkage='average'` for embeddings.

### GaussianMixture
KMeans with soft assignments and ellipse shaped clusters.
```python
gm = GaussianMixture(n_components=K, covariance_type='full', random_state=0).fit(X)
labels = gm.predict(X)
P = gm.predict_proba(X)                             # (N, K) membership probabilities
gm.bic(X)                                           # lower is better; compare across K
```
Use when clusters overlap and probabilities are wanted, or for density based anomaly scores (`gm.score_samples(X)`, low = unusual). Needs `d` small relative to `N` (PCA first).

-----

## 6. Few labels: nearest centroid and label propagation

Nearest centroid. In: labeled `E_lab` `(n, d)`, `y_lab`; unlabeled `E` `(N, d)`. Out: `pred` `(N,)`.
```python
centroids = np.array([E_lab[y_lab == k].mean(axis=0) for k in range(K)])   # (K, d)
D = np.zeros((len(E), K))
for k in range(K):
  D[:, k] = np.linalg.norm(E - centroids[k], axis=1)
pred = D.argmin(axis=1)
```
Self training: predict, take the most confident predictions (largest margin between best and second best distance), add them to the labeled set, recompute centroids, repeat 2..3 times. Stop when the predictions stop changing.

KNN with the labeled rows as the base: `KNeighborsClassifier(n_neighbors=5).fit(E_lab, y_lab).predict(E)`; simplest option when there are 50+ labels.

-----

## 7. Anomalies

Output is always a score per row; the threshold is a separate decision made on a histogram or on the known fraction of anomalies.

### IsolationForest (tables, embeddings)
```python
iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=0).fit(X)
score = -iso.score_samples(X)                       # higher = more unusual
flag = iso.predict(X) == -1                         # True for the contamination share
```
`contamination` = expected share of anomalies; if unknown, ignore `predict` and threshold `score` yourself.

### Distance to nearest neighbors
```python
nn = NearestNeighbors(n_neighbors=6).fit(X)
dist, _ = nn.kneighbors(X)
score = dist[:, 1:].mean(axis=1)                    # mean distance to 5 nearest others (skip column 0 = itself)
```
Rows far from everything are odd. Works on embeddings without any fitting assumption.

### Distance to a reference set (known-good examples exist)
```python
mu = X_good.mean(axis=0)
score = np.linalg.norm(X - mu, axis=1)              # distance to the center of the good ones
# or nearest good example:
nn = NearestNeighbors(n_neighbors=1).fit(X_good)
score = nn.kneighbors(X)[0][:, 0]
```

### Reconstruction error (autoencoder, PCA)
PCA version, no network needed: project to k components and back; rows that reconstruct badly are unusual.
```python
pca = PCA(n_components=20, random_state=0).fit(X_good)
R = pca.inverse_transform(pca.transform(X))
score = ((X - R) ** 2).mean(axis=1)                 # (N,)
```

### Choosing the threshold
```python
plt.hist(score, bins=200); plt.show()               # two humps -> threshold in the valley
thr = np.percentile(score, 95)                      # or from a known share of anomalies
flag = score > thr
```
If a few labeled anomalies exist: scan thresholds and pick the one with the best f1 on them.

-----

## 8. Checking unsupervised results without labels

- Cluster sizes: `np.bincount(labels)`. One huge + many tiny = wrong K, unscaled data, or outliers pulling a center.
- Read or look at 5 examples per cluster. Nothing replaces this.
- Silhouette for comparing settings on the same X; never for comparing across different X (different scaling, different reduction).
- Stability: run KMeans with two seeds and compare labels with `adjusted_rand_score(l1, l2)` (1 = identical grouping). Under 0.8 means the structure is weak.
- If a handful of labels exist, `adjusted_rand_score(y_known, labels_known)` or the mapped accuracy on those rows is the real check.

-----

## 9. Traps

- Not scaling a table before KMeans/DBSCAN: the column with the largest numbers decides everything.
- Not normalizing embeddings or SVD output: KMeans then clusters by vector length instead of direction.
- Clustering TF-IDF directly (sparse, 50k columns) is slow and noisy; SVD first.
- Cluster ids are arbitrary: fix `random_state`, build the mapping after the final fit, never reuse a mapping after refitting.
- `silhouette_score` on 50k rows takes minutes: use `sample_size`.
- DBSCAN marking almost everything as noise: eps too small. Everything one cluster: eps too large.
- PCA on data with NaN crashes: fill first.
- Using t-SNE coordinates for anything but the picture.
- Choosing an anomaly threshold "at 0.5" or at the default `contamination` without looking at the histogram.
- `fit_transform` on train and `fit_transform` again on test: the second call refits and changes the axes. Fit once, `transform` both.
