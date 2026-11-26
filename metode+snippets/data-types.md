# data-types.md — lists, numpy, pandas, torch: shapes, axes, conversions

Goal: no more time lost on "what shape is this" and "how does it become that".
Base reflex: `print(type(x), x.shape)` (or `len(x)` for lists) before any new operation.

-----

## 1. What is what

| object | looks like | used when | knows |
|---|---|---|---|
| `list` | `[1, 2, 3]` | building a column row by row, collecting results | `append`, `len`, indexing, slicing; no math on the whole thing |
| `dict` | `{'a': 1}` | counting occurrences, mapping category -> number | `in`, `.get`, `.items()` |
| `np.ndarray` | `array([[1, 2], [3, 4]])` | math on everything at once, images, embeddings | `shape`, `axis`, broadcasting, `@` |
| `pd.Series` | one column with an index | a column of a table | `value_counts`, `isna`, `map`, `str.` |
| `pd.DataFrame` | table with column names | tabular data | `loc/iloc`, `groupby`, `select_dtypes` |
| `torch.Tensor` | like ndarray, but on GPU and with gradients | anything entering a network | `shape`, `dim`, `.to(device)`, `.backward()` |
| sparse matrix (`scipy.sparse`) | `(N, V)` almost all zeros | TF-IDF output | `.shape`, `.toarray()`, `[rows]`; NOT `.iloc` |
| `PIL.Image` | a picture | reading / showing / resizing | `.size` = (W, H)!, `.mode`, `.convert('RGB')`, `.resize((W, H))` |

-----

## 2. Axes

`axis` = the direction along which the operation "collapses". The result loses that axis.

```python
X.shape == (N, d)
X.mean(axis=0)   # down the columns, across rows -> (d,)   one mean per column / per dimension
X.mean(axis=1)   # along the rows, across columns -> (N,)   one mean per row / per example
X.mean()         # over everything -> one number
X.mean(axis=1, keepdims=True)   # (N, 1): keeps the axis, so X can be divided by it
```
Mnemonic: `axis=0` answers "for each column", `axis=1` answers "for each row".

Examples from the notebooks:
- `X[y == 0].mean(axis=0)`: center of class 0, a vector of length d (gender-bias).
- `mask.any(axis=0)`: for each pixel column, is there ink? -> vector of length W (music).
- `mask.mean(axis=1) > 0.5`: for each pixel row, more than half black? -> staff lines.
- `np.abs(refs - img).mean(axis=(1, 2))`: `refs` is `(K, H, W)`, mean over H and W -> one distance per reference `(K,)`.
- `out.argmax(1)` on `(N, K)`: for each row, the position of the max -> predicted class `(N,)`. `argmax(0)` would give, for each class, the row with the highest score (almost never the wanted thing).
- `np.sort(d, axis=1)`: sorts each row separately; `d[:, 0]` smallest distance, `d[:, 1]` second smallest.
- `scores.argmax(axis=1)` on `(Q, D)`: for each query, the closest document.

Same in torch, the name is `dim`: `x.mean(dim=0)`, `x.argmax(dim=1)`, `F.softmax(out, dim=1)` (probabilities on a row sum to 1).

-----

## 3. Standard shapes

| data | shape | note |
|---|---|---|
| table | `(N, d)` | N rows, d columns |
| labels | `(N,)` | NOT `(N, 1)`; `y.ravel()` if it went wrong |
| embeddings | `(N, d)` | d = 384 (MiniLM), 512 (CLIP, resnet18), 768 (bert) |
| gray image, numpy / PIL | `(H, W)` | values 0..255 uint8 |
| color image, numpy / PIL | `(H, W, 3)` | what matplotlib `imshow` wants |
| image, torch | `(C, H, W)` | `ToTensor()` converts and scales to 0..1 |
| batch of images, torch | `(N, C, H, W)` | what `Conv2d` receives |
| after conv | `(N, channels, H', W')` | each `MaxPool2d(2)` halves H', W' |
| input to `Linear` | `(N, k)` | hence `Flatten` before it |
| tokenized text | `(N, L)` ids + `(N, L)` attention_mask | L = max_length, with padding |
| transformer output | `(N, L, d)` | `[:, 0]` = the [CLS] token -> `(N, d)` |
| classifier output | `(N, K)` logits | `argmax(1)` -> `(N,)` |
| binary output | `(N, 1)` or `(N,)` | `out[:, 0]` to match `y` of shape `(N,)` |
| TF-IDF | `(N, V)` sparse | V = vocabulary size |
| distances / similarities | `(N, M)` | row = one element of the first set |

Images: `img.size` in PIL is `(W, H)`, but `np.array(img).shape` is `(H, W, 3)`. The order flips.

-----

## 4. numpy

```python
a = np.array([1, 2, 3])                 # from a list
a = np.zeros((N, d)); np.ones(N); np.arange(N); np.linspace(0, 1, 100)
a.shape, a.dtype, a.ndim

# indexing
a[0], a[-1], a[2:5], M[i, j], M[i], M[:, j], M[:5, :2]
M[mask]            # mask is bool of shape (N,): keeps rows with True
M[idx]             # idx is a list/array of positions: takes rows in that order (repeats allowed)
np.where(cond)[0]  # positions where cond is True
np.where(cond, a, b)   # a where cond, b otherwise

# shapes
M.reshape(N, -1)   # -1 = work it out; (N, H, W) -> (N, H*W)
M.flatten()        # everything into one vector
M.T                # transpose (d, N)
M[:, None]         # (N,) -> (N, 1); M[None, :] -> (1, N)
np.concatenate([A, B])           # glue along axis 0: (N1, d) + (N2, d) -> (N1+N2, d)
np.concatenate([A, B], axis=1)   # side by side: (N, d1) + (N, d2) -> (N, d1+d2)
np.stack([a, b])                 # creates a new axis: two (d,) -> (2, d)

# math on everything
M.sum(axis=0), M.mean(axis=1), M.max(), M.argmax(axis=1), M.std()
np.sqrt(M), np.log1p(M), np.abs(M), np.clip(M, 0, 1), np.round(M, 2)
np.linalg.norm(M, axis=1)                # length of each row (N,)
M / np.linalg.norm(M, axis=1, keepdims=True)   # each row to length 1
A @ B.T                                  # (N, d) @ (d, M) -> (N, M) dot products; on normalized vectors = cosine
np.argsort(a)                            # positions in ascending order; np.argsort(-a) descending
np.argsort(-S, axis=1)[:, :10]           # top 10 per row
np.unique(y, return_counts=True); np.bincount(y)
np.percentile(err, [50, 90, 95, 99]); np.quantile(a, 0.3)
np.isin(labels, keep_list)               # mask: is each element in the list?

# random
rng = np.random.default_rng(0); rng.random(N); rng.integers(0, A, size=N); rng.permutation(N)
```
Broadcasting: `(N, d) - (d,)` subtracts the vector from every row. `(N, d) / (N, 1)` divides every row by its own number. `(N, d) / (N,)` errors -> `[:, None]`.

-----

## 5. pandas

```python
df = pd.read_csv(path); df.shape; df.columns; df.dtypes; df.info(); df.head()
df['col']                 # Series
df[['a', 'b']]            # DataFrame with 2 columns
df.loc[df['Price'] > 10]  # rows by condition
df.loc[i, 'col']          # by index LABEL
df.iloc[i]                # by POSITION; df.iloc[fit_rows] in cross validation
df['col'].iloc[i]         # i-th value of a column

df.select_dtypes('number'); df.select_dtypes(exclude='number')
df.isna().sum(); df.nunique(); df.duplicated().sum(); df.describe()
df['col'].value_counts(); df['col'].value_counts(normalize=True); df['col'].unique()
df.groupby('cat')['target'].agg(['count', 'median'])
df['col'].map({'Bad': 0, 'Good': 1})          # dict -> NaN for anything not in the dict
df['col'].isin(set_or_list)                   # mask
df['text'].str.split(); df['text'].str.len(); df['text'].str.contains('_', regex=False); df['x'].str.strip('_')
pd.to_numeric(df['col'], errors='coerce')     # non-numbers -> NaN
df['col'].fillna(0); df['col'].replace('_', np.nan); df.drop(columns=['a', 'b'])
df.sort_values('id').reset_index(drop=True)   # after sorting/filtering, renumber the index
pd.concat([df1, df2], ignore_index=True)      # one under the other
pd.concat([df1, df2], axis=1)                 # side by side
df.merge(other, on='id', how='left')

# building from lists (the usual way)
df = pd.DataFrame({'id': ids, 'path': paths})
X = pd.DataFrame(list_of_vectors)             # each vector one row
X = pd.DataFrame(arr, columns=cols, index=df.index)   # back from numpy after a scaler

# out to numpy / torch
df.values  /  df.to_numpy()                   # (N, d) ndarray
df['col'].values                              # (N,)
list(df['text'])                              # list of strings for tokenizer / encode
```
Series vs scalar: `df['col'].mean()` is a number; `df[['col']].mean()` is a Series. `df['col'][0]` looks up label 0 (may be missing after filtering); `df['col'].iloc[0]` takes the first value.

Explicit row loop: `for v in df['col']:` (one column) or `for i in range(len(df)): df['col'].iloc[i]`. `iterrows` is slow and not needed.

-----

## 6. torch

```python
t = torch.tensor([1.0, 2.0]); torch.zeros(N, d); torch.ones(N); torch.arange(N); torch.randn(N, d)
t.shape, t.dtype, t.device, t.dim()

# dtypes: most errors come from here
x = x.float()          # float32: inputs, images, targets for BCE/L1/MSE
y = y.long()           # int64: targets for CrossEntropyLoss, indices
x.to(device); model.to(device)    # both on the same device, otherwise error

# shapes
x.view(N, -1) / x.reshape(N, -1)  # view needs contiguous memory; reshape always works
x.flatten(1)                      # (N, C, H, W) -> (N, C*H*W)
x.unsqueeze(0)                    # (C, H, W) -> (1, C, H, W): one image becomes a batch of 1
x.squeeze(0)                      # the reverse; x.squeeze() removes all size-1 axes
x.permute(1, 2, 0)                # (C, H, W) -> (H, W, C) for imshow
x.transpose(0, 1)                 # swap two axes
torch.cat([a, b], dim=0)          # glue; torch.stack([a, b]) creates a new axis
x[:, 0], x[mask], x[idx]          # as in numpy
torch.roll(x, shifts=(dy, dx), dims=(2, 3))   # pixel shift (augmentation)
torch.flip(x, dims=[3]); torch.rot90(x, 1, dims=[2, 3])

# operations
x.sum(dim=1), x.mean(dim=0), x.max(dim=1).values, x.argmax(dim=1), x.norm(dim=1, keepdim=True)
torch.sigmoid(out); F.softmax(out, dim=1); torch.clamp(x, 0, 1)
(preds == y).float().mean()       # accuracy; bool -> float before mean

# out of the graph
loss.item()                       # one-element tensor -> python number
x.detach().cpu().numpy()          # tensor -> numpy (detach cuts the gradient, cpu moves off the GPU)
x.tolist()
with torch.no_grad():             # at prediction: no gradient, less memory, faster
model.train() / model.eval()      # eval: turns dropout off and freezes batchnorm; without it predictions differ
```
What `torch.no_grad()` means: no history of operations is kept for `backward()`. Used at validation and prediction. It does not replace `model.eval()`; both are needed.

Dataset / DataLoader: `__getitem__(i)` returns `(x, y)` for one example, `x` of shape `(C, H, W)`; DataLoader stacks them into `(B, C, H, W)` and `(B,)`. `shuffle=True` only for fitting. Without DataLoader: `perm = torch.randperm(N)`, then `X_t[perm[start:start+bs]]`.

Training loop, the skeleton that never changes:
```python
for epoch in range(EPOCHS):
  model.train()
  for xb, yb in fit_loader:
    xb, yb = xb.to(device), yb.to(device)
    loss = loss_fn(model(xb), yb)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
  model.eval()
  with torch.no_grad():
    # preds on val, score, keep best_state
```

-----

## 7. Conversions

| from | to | code |
|---|---|---|
| list | np | `np.array(lst)` |
| list of vectors | np `(N, d)` | `np.array(lst)` (all the same length) |
| np | list | `a.tolist()` |
| np | torch | `torch.from_numpy(a).float()` (`.long()` for labels) |
| torch | np | `t.detach().cpu().numpy()` |
| torch with 1 element | number | `t.item()` |
| DataFrame | np | `df.values` or `df.to_numpy()` |
| np | DataFrame | `pd.DataFrame(a, columns=cols)` |
| Series | np | `s.values`; to list: `list(s)` |
| Series | set (fast lookup) | `set(s)` |
| PIL | np | `np.array(img)` -> uint8 0..255, `(H, W, 3)` or `(H, W)` |
| np uint8 | PIL | `Image.fromarray(a)`; from float 0..1: `Image.fromarray((a * 255).astype(np.uint8))` |
| PIL | torch | `transforms.ToTensor()(img)` -> float 0..1, `(C, H, W)` |
| torch image | imshow | `t.permute(1, 2, 0).cpu().numpy()`; gray: `t[0].cpu().numpy()` with `cmap='gray'` |
| np image `(N, C, H, W)` | imshow | `X[k].transpose(1, 2, 0)` |
| sparse TF-IDF | dense | `V.toarray()` (watch memory); not needed for KMeans/LR |
| sparse TF-IDF | 2D for plotting | `TruncatedSVD(2).fit_transform(V)` |
| string holding a list | list | `json.loads(s)` (e.g. `"[[1, 2, 3, 4]]"`) |
| list | string for csv | `json.dumps(lst)` or `' '.join(str(v) for v in lst)` |
| text labels | numbers | `classes = sorted(y.unique()); class_to_id = {c: i ...}`; back with `classes[pred]` |
| probabilities `(N, K)` | class | `.argmax(1)`; binary `(p[:, 1] > threshold).astype(int)` |
| bool | 0/1 | `.astype(int)` (np, pandas), `.float()` / `.long()` (torch) |

-----

## 8. Traps that cost time before

- `df[col][i] = x` does not save. Build a list and assign the whole column, or `df.loc[i, col] = x`.
- After shuffle / filtering, `[i]` and `.loc[i]` no longer mean "the i-th". `reset_index(drop=True)` or `.iloc`.
- `pd.to_numeric` before `strip('_')` turns `"23_"` into NaN. Clean first, convert after.
- `str(t)` on a missing text gives `'nan'`. `fillna('')` first.
- `CrossEntropyLoss` wants `long` targets of shape `(N,)` and outputs `(N, K)`, no softmax. `BCEWithLogitsLoss` wants `float` targets with the same shape as the output; `out[:, 0]` when the output is `(N, 1)`.
- `(N,)` vs `(N, 1)`: scikit wants `y` of shape `(N,)`. Subtracting `(N,)` from `(N, 1)` broadcasts to `(N, N)` with no error and gives garbage.
- `ToTensor()` scales to 0..1; `np.array(img)` stays 0..255. Never mixed in the same model.
- ImageNet normalization `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` only for pretrained models. From scratch: mean and std of the data, or just 0..1.
- `img.size` is `(W, H)`; `resize((W, H))`; `np.array(img).shape` is `(H, W, C)`. Boxes: `[x1, y1, x2, y2]`, x along the width.
- Slicing in numpy / torch is a "view": changing the slice changes the original. `.copy()` / `.clone()` when the original must stay.
- `model.eval()` + `torch.no_grad()` for every prediction. Without `eval()`, dropout and batchnorm stay in training mode.
- A GPU tensor does not go into numpy / matplotlib: `.cpu()` first.
- `StratifiedKFold.split(X, y)` wants class labels in `y`, not real numbers. For regression `KFold`.
- `scaler.fit` on train only; `transform` on both. Same for the vectorizer and any statistic (means, category lists).
- `KMeans` without scaling on a table: the column with the largest values decides the clusters alone.
- Counting dict: `if k not in d: d[k] = 0` before `d[k] += 1`, or `d[k] = d.get(k, 0) + 1`.
- `sorted(os.listdir(...))`: `listdir` has no guaranteed order; ids come from the filename, not the position.
- `to_csv(..., index=False)` always for the submission.
