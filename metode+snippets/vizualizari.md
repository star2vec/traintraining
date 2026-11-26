# vizualizari.md — plots that pay off in a contest

A plot is worth drawing when it changes a decision: which threshold, which K, whether the data is dirty, whether the model is learning. Everything else is a print.

Setup: `import matplotlib.pyplot as plt`. Every plot ends with `plt.show()`.

-----

## 1. Histogram — one column of numbers

```python
plt.hist(values, bins=50)
plt.show()
```
Use for: the target, prediction errors, anomaly scores, text lengths, distances.

What to look for:
- spikes at certain values (prices 9.99 / 14.99): rounding or categories hidden in a number -> round predictions to them
- two humps: two kinds of rows -> a threshold goes in the valley between them (anomaly scores, reconstruction errors)
- a long tail to the right: `np.log1p` before linear models
- `bins=10` hides spikes, `bins=50` shows them; for thresholds go to `bins=200..500`

Two histograms on top of each other (known good vs suspects, class 0 vs class 1):
```python
plt.hist(a, bins=50, alpha=0.5, density=True, label='known valid')
plt.hist(b, bins=50, alpha=0.5, density=True, label='suspects')
plt.legend(); plt.show()
```
`density=True` makes them comparable when the two groups have different sizes. The threshold goes where the two curves cross.

-----

## 2. Scatter — two columns, colored by class

```python
plt.scatter(df['a'], df['b'], c=df['label'], s=5, cmap='coolwarm')
plt.xlabel('a'); plt.ylabel('b')
plt.show()
```
`c=` needs numbers (map text labels to 0..K first). `s=5` for many points.

What to look for:
- classes in separate blobs: a simple model is enough (even a rectangle rule, as in points_2024)
- a shape no straight line separates (ring, U): trees or KNN, not logistic regression
- one class inside a region of the other: the boundary is spatial, use both columns together

-----

## 3. Embeddings or many columns in 2D

```python
P = PCA(n_components=2).fit_transform(X)          # dense: embeddings, pixels, scaled table
P = TruncatedSVD(n_components=2, random_state=0).fit_transform(V)   # sparse: TF-IDF
plt.scatter(P[:, 0], P[:, 1], c=labels, s=3, cmap='tab10')
plt.colorbar(); plt.show()
```
Color by: known labels (is the signal there?) or by cluster id (do the clusters look like groups?).

What to look for:
- colored groups separate: embeddings + a linear model will work
- everything one cloud: the 2D view lost the signal, not necessarily the data; check with a quick LogisticRegression before giving up
- clusters: one giant blob with tiny islands around it means K is wrong or the data is unscaled
- a few points far from everything: outliers or broken rows, print them

-----

## 4. Images — always look at some

Per class, a few examples (train):
```python
classes = sorted(df['label'].unique())
fig, axes = plt.subplots(len(classes), 5, figsize=(15, 3 * len(classes)))
for r in range(len(classes)):
  rows = df[df['label'] == classes[r]].sample(5, random_state=0)
  for k in range(5):
    axes[r, k].imshow(Image.open(rows['path'].iloc[k]))
    axes[r, k].set_title(classes[r]); axes[r, k].axis('off')
plt.show()
```
Same grid for test, 1 row, no titles. Gray images: `cmap='gray'`.

Before / after a processing step (threshold, line removal, crop):
```python
plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('input')
plt.subplot(1, 2, 2); plt.imshow(result, cmap='gray'); plt.title('after')
plt.show()
```

Boxes on an image:
```python
import matplotlib.patches as patches
fig, ax = plt.subplots(); ax.imshow(img)
ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
plt.show()
```

One cluster of images (20 examples) to decide what the cluster is:
```python
fig, axes = plt.subplots(2, 10, figsize=(15, 3))
axes = axes.flatten()
for k in range(20):
  axes[k].axis('off')
  if k < len(rows): axes[k].imshow(images[rows[k]], cmap='gray')
plt.show()
```

What to look for:
- does a human still give the same label after flip / rotation? decides augmentation
- same size and color mode everywhere? mixed sizes -> resize all; mixed modes -> `convert('RGB')`
- is the object centered and alone? if yes, flatten + KNN works; if not, CNN or cropping first
- test images that look different from train (other background, inverted colors, scanner noise) -> fix in preprocessing, as in panda-mnist
- after a processing step: did it remove what it should and nothing else? one example is enough to catch a bug, five catch most

-----

## 5. Training curves — is the network learning?

Collect the numbers per epoch, then plot once at the end:
```python
plt.plot(fit_losses, label='fit loss'); plt.legend(); plt.show()
plt.plot(fit_acc, label='fit'); plt.plot(val_acc, label='val'); plt.legend(); plt.show()
```
What to look for:
- loss flat from epoch 1: lr wrong, labels wrong, `zero_grad` missing
- loss going down, val accuracy going up then flat: stop there (early stopping)
- fit accuracy climbing while val drops: overfit, use the best epoch and add augmentation or dropout
- val jumping around a lot: val set too small or lr too high

-----

## 6. Where the model is wrong

Confusion matrix, for classification:
```python
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_val, preds))
```
Rows = true class, columns = predicted. Big numbers off the diagonal = two classes the model mixes up; print a few texts/images from that pair.

Worst examples, for regression or any custom score:
```python
errors = np.abs(y_val - preds)
for i in np.argsort(-errors)[:10]:
  print(round(errors[i], 2), '|', X_val.iloc[i].to_dict())
```
What to look for: a pattern among the worst rows (same category, missing values, extreme values) is the next feature or the next cleaning rule.

-----

## 7. Quick bars

```python
df['label'].value_counts().plot(kind='bar'); plt.show()
```
Class balance in one look. A dominant class means accuracy is misleading -> f1, `class_weight='balanced'`, threshold on probabilities.

-----

## 8. Habits

- Plot early, before cleaning: the target, a few images, the class balance. Ten minutes that steer the whole session.
- Plot before choosing any threshold or K. Never pick them from a formula alone.
- Save the plot cell, rerun it after cleaning: the same histogram with the spike gone, the same grid with the noise removed.
- Do not plot in the loop that trains; collect numbers, plot after.
- When a plot looks surprising, print the rows behind it before changing the model.
