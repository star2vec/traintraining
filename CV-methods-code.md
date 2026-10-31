# CV-methods-code.md — code blocks for every approach in CV-methods.md

Each block says what goes in, what comes out (type and shape), then the code. No Pipelines, no Trainer: every step is visible. Section letters follow the order of CV-methods.md.

Imports used everywhere:
```python
import numpy as np, pandas as pd, os, copy
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); np.random.seed(0)
```

-----

## 0. Load and look

In: folder of images + csv with ids and labels. Out: `X` numpy `(N, H, W)` or `(N, H, W, 3)`, `y` `(N,)` ints.
```python
train = pd.read_csv('train.csv')              # columns: id, label (+ maybe group)
classes = sorted(train['label'].unique())
class_to_id = {}
for i in range(len(classes)): class_to_id[classes[i]] = i
y = np.array([class_to_id[c] for c in train['label']])

def load_gray(path, size=28):
  img = Image.open(path).convert('L').resize((size, size))   # PIL resize takes (W, H)
  return np.array(img)                                        # (size, size) uint8 0..255

def load_rgb(path, size=224):
  img = Image.open(path).convert('RGB').resize((size, size))
  return np.array(img)                                        # (size, size, 3) uint8

images = []
for i in range(len(train)):
  images.append(load_gray(os.path.join('train_images', str(train['id'].iloc[i]) + '.png')))
X = np.array(images)                                          # (N, 28, 28)
print(X.shape, X.dtype, X.min(), X.max())

# first look
img = Image.open(path); print(img.size, img.mode)             # (W, H), 'L' / 'RGB' / 'RGBA' / 'P'
print(pd.Series([Image.open(p).size for p in paths[:200]]).value_counts())   # are sizes mixed?
```
Grid of examples per class: see vizualizari.md section 4.

Train vs test shift check (same scale?):
```python
print(X_train.mean(), X_train.std(), X_test.mean(), X_test.std())
plt.hist(X_train.ravel(), bins=50, alpha=0.5, density=True); plt.hist(X_test.ravel(), bins=50, alpha=0.5, density=True); plt.show()
```

-----

## A. Flatten + KNN / LogisticRegression (small synthetic images)

In: `X` `(N, H, W)` uint8. Out: `pred` `(M,)`.
```python
Xf = X.reshape(len(X), -1) / 255.0            # (N, H*W) floats in [0, 1]
Xf_test = X_test.reshape(len(X_test), -1) / 255.0
knn = KNeighborsClassifier(n_neighbors=3).fit(Xf, y)
pred = knn.predict(Xf_test)
lr = LogisticRegression(C=0.1, max_iter=2000).fit(Xf, y)
```
Crop to the ink, pad to square, resize (do this before flattening when objects are not centered):
```python
def crop_center_resize(gray, size=28, ink_threshold=128):
  mask = gray < ink_threshold                 # True on dark pixels; use > for light ink on dark background
  rows = np.where(mask.any(axis=1))[0]        # row indices that contain ink
  cols = np.where(mask.any(axis=0))[0]
  if len(rows) == 0: return np.zeros((size, size), dtype=np.uint8)
  crop = gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
  h, w = crop.shape
  side = max(h, w)
  square = np.full((side, side), 255, dtype=np.uint8)      # white background
  top = (side - h) // 2; left = (side - w) // 2
  square[top:top + h, left:left + w] = crop
  return np.array(Image.fromarray(square).resize((size, size)))
```

-----

## B. Numpy images -> torch tensors, Dataset, DataLoader

In: `X` `(N, H, W)` or `(N, H, W, 3)` uint8. Out: tensors `(N, C, H, W)` float.
```python
# gray
X_t = torch.from_numpy(X).float().unsqueeze(1) / 255.0        # (N, 1, H, W)
# rgb
X_t = torch.from_numpy(X).float().permute(0, 3, 1, 2) / 255.0 # (N, H, W, 3) -> (N, 3, H, W)
y_t = torch.from_numpy(y).long()                              # (N,)

# split
fit_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=0)
```
Batches without a DataLoader (enough when everything fits in memory):
```python
perm = np.random.permutation(len(fit_idx))
for start in range(0, len(perm), 64):
  idx = fit_idx[perm[start:start + 64]]
  xb = X_t[idx].to(DEVICE); yb = y_t[idx].to(DEVICE)
```
Dataset + DataLoader (when images are read from disk or augmented with torchvision):
```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDS(Dataset):
  def __init__(self, paths, labels, tfm):
    self.paths = paths; self.labels = labels; self.tfm = tfm
  def __len__(self): return len(self.paths)
  def __getitem__(self, i):
    img = Image.open(self.paths[i]).convert('RGB')
    x = self.tfm(img)                                   # (3, H, W) float tensor
    if self.labels is None: return x, -1
    return x, self.labels[i]

fit_loader = DataLoader(ImageDS(fit_paths, y_fit, train_tfm), batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(ImageDS(val_paths, y_val, eval_tfm), batch_size=64, shuffle=False)
test_loader = DataLoader(ImageDS(test_paths, None, eval_tfm), batch_size=64, shuffle=False)   # never shuffle test
```

-----

## C. Augmentation

### On tensors in memory (small images)
```python
def shift(x, max_px=2):                                 # x (B, C, H, W); one random shift per batch
  dy = int(np.random.randint(-max_px, max_px + 1)); dx = int(np.random.randint(-max_px, max_px + 1))
  return torch.roll(x, shifts=(dy, dx), dims=(2, 3))     # wraps around; fine for 2 px on padded symbols

def flip_lr(x):                                         # only if the label survives a mirror
  return torch.flip(x, dims=[3])

def add_noise(x, std=0.05):
  return (x + std * torch.randn_like(x)).clamp(0, 1)

def small_rotate(x, max_deg=10):                         # x (B, C, H, W)
  deg = float(np.random.uniform(-max_deg, max_deg))
  theta = torch.tensor([[np.cos(np.deg2rad(deg)), -np.sin(np.deg2rad(deg)), 0],
                        [np.sin(np.deg2rad(deg)),  np.cos(np.deg2rad(deg)), 0]], dtype=torch.float32)
  theta = theta.unsqueeze(0).repeat(x.shape[0], 1, 1).to(x.device)      # (B, 2, 3)
  grid = F.affine_grid(theta, x.shape, align_corners=False)
  return F.grid_sample(x, grid, align_corners=False, padding_mode='border')
```
Use inside the training loop only on the fit batch: `xb = shift(xb)`. Per-sample randomness: loop over the batch and shift each `xb[i:i+1]` separately (slower, more variety).

### With torchvision (photos read from disk)
```python
IMAGENET_MEAN = (0.485, 0.456, 0.406); IMAGENET_STD = (0.229, 0.224, 0.225)
train_tfm = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.RandomHorizontalFlip(),                       # drop for digits / text / oriented objects
  transforms.RandomRotation(10),
  transforms.ColorJitter(brightness=0.2, contrast=0.2),
  transforms.ToTensor(),                                   # PIL -> (3, H, W) in [0, 1]
  transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])      # only for pretrained models
eval_tfm = transforms.Compose([
  transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
```
`RandomResizedCrop(224, scale=(0.7, 1.0))` instead of Resize when objects fill the frame. From scratch models: drop the Normalize line.

### Same geometry on image and mask
```python
def augment_pair(img, mask):                             # both PIL
  if np.random.rand() < 0.5:
    img = img.transpose(Image.FLIP_LEFT_RIGHT); mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
  deg = float(np.random.uniform(-10, 10))
  img = img.rotate(deg, resample=Image.BILINEAR)
  mask = mask.rotate(deg, resample=Image.NEAREST)        # NEAREST keeps the mask 0/1
  return img, mask
```
Resizing a mask: `mask.resize((W, H), Image.NEAREST)`.

### Boxes under a horizontal flip
```python
# box [x1, y1, x2, y2] in pixels, image width W
x1, y1, x2, y2 = box
box_flipped = [W - x2, y1, W - x1, y2]
```

-----

## D. Small CNN from scratch, full training loop

In: `X_t` `(N, C, H, W)` float in [0, 1], `y_t` `(N,)` long. Out: trained model, `pred` `(M,)`.
```python
class SmallCNN(nn.Module):
  def __init__(self, in_ch, n_classes, c1=16, c2=32, c3=64, use_bn=True):
    super().__init__()
    def block(a, b):
      layers = [nn.Conv2d(a, b, 3, padding=1, bias=not use_bn)]
      if use_bn: layers.append(nn.BatchNorm2d(b))
      layers.append(nn.ReLU())
      return nn.Sequential(*layers)
    self.features = nn.Sequential(
      block(in_ch, c1), nn.MaxPool2d(2),      # H -> H/2
      block(c1, c2), nn.MaxPool2d(2),         # H/2 -> H/4
      block(c2, c3),
      nn.AdaptiveAvgPool2d(1))                # (B, c3, 1, 1) whatever H was
    self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(c3, n_classes))
  def forward(self, x):
    return self.head(self.features(x))        # logits (B, n_classes), no softmax here

model = SmallCNN(in_ch=1, n_classes=10).to(DEVICE)
print(sum(p.numel() for p in model.parameters()))          # parameter count
print(model(torch.zeros(2, 1, 28, 28).to(DEVICE)).shape)    # (2, 10) shape check before training
```
Parameter count by hand: a `Conv2d(a, b, 3)` has `a*b*9` weights + `b` biases; BatchNorm adds `2*b`; `Linear(a, b)` has `a*b + b`.

```python
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.3, patience=2)

def predict_logits(model, X_t, batch_size=256):
  model.eval(); pieces = []
  with torch.no_grad():
    for start in range(0, len(X_t), batch_size):
      pieces.append(model(X_t[start:start + batch_size].to(DEVICE)).cpu())
  return torch.cat(pieces)                                  # (N, K)

best_acc = -1; best_state = None; bad_epochs = 0
for epoch in range(30):
  model.train()
  perm = np.random.permutation(len(fit_idx))
  total_loss = 0.0
  for start in range(0, len(perm), 64):
    idx = fit_idx[perm[start:start + 64]]
    xb = shift(X_t[idx]).to(DEVICE); yb = y_t[idx].to(DEVICE)
    loss = loss_fn(model(xb), yb)
    opt.zero_grad(); loss.backward(); opt.step()
    total_loss += loss.item() * len(idx)
  val_pred = predict_logits(model, X_t[val_idx]).argmax(dim=1).numpy()
  acc = accuracy_score(y[val_idx], val_pred)
  sched.step(acc)
  print(epoch, round(total_loss / len(perm), 4), round(acc, 4))
  if acc > best_acc:
    best_acc = acc; best_state = copy.deepcopy(model.state_dict()); bad_epochs = 0
  else:
    bad_epochs += 1
    if bad_epochs >= 5: break
model.load_state_dict(best_state)
pred = predict_logits(model, X_test_t).argmax(dim=1).numpy()
```
Reading the numbers: loss not moving by epoch 3 -> lr 1e-2 or 1e-4, check `yb.dtype == torch.int64`, check `xb` is in [0, 1]. Fit loss near 0 and val flat -> more augmentation / dropout 0.5 / smaller net.

Overfit-one-batch test (debug when nothing learns): take 32 images, run 100 steps on just those; accuracy on them must reach ~100%. If not, the bug is in data, labels or the loss, not the architecture.

Per-group evaluation (scanners) and a score formula:
```python
accs = []
for g in sorted(set(groups_val)):
  sel = np.where(groups_val == g)[0]
  accs.append(accuracy_score(y[val_idx][sel], val_pred[sel])); print(g, accs[-1])
P = sum(p.numel() for p in model.parameters())
print('mean acc', np.mean(accs), 'params', P, 'score', score_formula(np.mean(accs), P))
```

-----

## E. Pretrained resnet: features + LR, and fine tune

### Features + LogisticRegression
In: loader of `(3, 224, 224)` normalized tensors. Out: `(N, 512)` features.
```python
import torchvision
backbone = torchvision.models.resnet18(weights='DEFAULT')
backbone.fc = nn.Identity()                                 # output becomes the 512 features
backbone = backbone.to(DEVICE).eval()

def extract(loader):
  feats = []
  with torch.no_grad():
    for xb, _ in loader:
      feats.append(backbone(xb.to(DEVICE)).cpu().numpy())   # (B, 512)
  return np.concatenate(feats)

F_fit = extract(fit_loader_noaug); F_val = extract(val_loader); F_test = extract(test_loader)
clf = LogisticRegression(C=1, max_iter=2000).fit(F_fit, y_fit)
print(accuracy_score(y_val, clf.predict(F_val)))
```
(use `eval_tfm` for all three loaders here; no augmentation when extracting features.)

### Fine tune
```python
model = torchvision.models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

def predict_loader(model, loader):
  model.eval(); pieces = []
  with torch.no_grad():
    for xb, _ in loader: pieces.append(model(xb.to(DEVICE)).cpu())
  return torch.cat(pieces)

best_acc = -1
for epoch in range(10):
  model.train()
  for xb, yb in fit_loader:
    loss = loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE))
    opt.zero_grad(); loss.backward(); opt.step()
  acc = accuracy_score(y_val, predict_loader(model, val_loader).argmax(1).numpy())
  print(epoch, acc)
  if acc > best_acc: best_acc = acc; best_state = copy.deepcopy(model.state_dict())
model.load_state_dict(best_state)
```
Freeze everything except the head (tiny data): `for p in model.parameters(): p.requires_grad = False` before replacing `fc`, then `opt = AdamW(model.fc.parameters(), lr=1e-3)`.

### TTA with the flip
```python
logits = predict_loader(model, test_loader)
logits_flip = []
model.eval()
with torch.no_grad():
  for xb, _ in test_loader: logits_flip.append(model(torch.flip(xb, dims=[3]).to(DEVICE)).cpu())
logits = (logits + torch.cat(logits_flip)) / 2
pred = logits.argmax(1).numpy()
```

-----

## F. Several outputs in one head (class + box + number)

In: targets `y_cls` `(N,)` long, `y_box` `(N, 4)` floats in [0, 1], `y_num` `(N,)` floats in [0, 1]. Out: one `fc` with `K + 4 + 1` outputs.
```python
K = len(classes)
model.fc = nn.Linear(model.fc.in_features, K + 4 + 1)

def split_outputs(out):                 # out (B, K + 5)
  logits = out[:, :K]
  box = torch.sigmoid(out[:, K:K + 4])  # (B, 4) in [0, 1]
  num = torch.sigmoid(out[:, K + 4])    # (B,)
  return logits, box, num

def multi_loss(out, yc, yb, yn):
  logits, box, num = split_outputs(out)
  l_cls = F.cross_entropy(logits, yc)
  l_box = F.l1_loss(box, yb)
  l_iou = (1 - iou_torch(box, yb)).mean()
  l_num = F.l1_loss(num, yn)
  return l_cls + 5 * l_box + l_iou + 10 * l_num
```
Boxes are normalized by image width/height for the target, and multiplied back at the end: `x1 = box[0] * W`.

-----

## G. IoU and NMS

IoU of two boxes `[x1, y1, x2, y2]`:
```python
def iou(a, b):
  ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
  ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
  iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
  inter = iw * ih
  area_a = (a[2] - a[0]) * (a[3] - a[1]); area_b = (b[2] - b[0]) * (b[3] - b[1])
  union = area_a + area_b - inter
  if union <= 0: return 0.0
  return inter / union
```
Torch version for a batch, `A` and `B` both `(N, 4)` (used in the loss):
```python
def iou_torch(A, B):
  ix1 = torch.max(A[:, 0], B[:, 0]); iy1 = torch.max(A[:, 1], B[:, 1])
  ix2 = torch.min(A[:, 2], B[:, 2]); iy2 = torch.min(A[:, 3], B[:, 3])
  inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
  area_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]); area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
  return inter / (area_a + area_b - inter + 1e-6)            # (N,)
```
NMS. In: list of boxes, list of scores. Out: indices of the boxes to keep.
```python
def nms(boxes, scores, iou_threshold=0.5):
  order = list(np.argsort(-np.array(scores)))      # best first
  keep = []
  while len(order) > 0:
    i = order[0]
    keep.append(i)
    remaining = []
    for j in order[1:]:
      if iou(boxes[i], boxes[j]) < iou_threshold: remaining.append(j)
    order = remaining
  return keep
```
Tuning the threshold on train with the official metric:
```python
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
  total = 0
  for img_id in train_images:
    k = nms(boxes[img_id], scores[img_id], t)
    total += official_metric([boxes[img_id][i] for i in k], gt_boxes[img_id])
  print(t, total / len(train_images))
```
Matching predicted boxes to ground truth (for a precision / recall style metric): for each gt box, the predicted box with the highest IoU; a hit if IoU >= 0.5 and that predicted box is not already used.

Per-box features for a second-stage classifier (box-avalanche pattern):
```python
rows = []
for img_id in images:
  b = boxes[img_id]; s = scores[img_id]
  order = np.argsort(-np.array(s))
  for r in range(len(order)):
    i = order[r]
    w = b[i][2] - b[i][0]; h = b[i][3] - b[i][1]
    n_overlap = 0; max_iou_better = 0.0
    for j in range(len(b)):
      if j == i: continue
      v = iou(b[i], b[j])
      if v > 0.3: n_overlap += 1
      if s[j] > s[i] and v > max_iou_better: max_iou_better = v
    rows.append([img_id, s[i], r, w, h, w * h, w / max(h, 1), n_overlap, max_iou_better])
feat = pd.DataFrame(rows, columns=['img', 'score', 'rank', 'w', 'h', 'area', 'aspect', 'n_overlap', 'max_iou_better'])
# label = 1 if iou(box, some gt box) >= 0.5 else 0; GroupKFold by 'img'; CatBoost; keep proba >= tuned threshold
```

-----

## H. Classical processing

### Threshold and components
In: `gray` `(H, W)` uint8. Out: `labels` `(H, W)` ints (0 background, 1..n blobs), one row per blob.
```python
from skimage import measure, filters
mask = gray < 128                                # ink; or gray > filters.threshold_otsu(gray) for bright objects
labels = measure.label(mask, connectivity=2)     # (H, W); connectivity=1 ignores diagonal touches
props = measure.regionprops(labels)
blobs = []
for r in props:
  if r.area < 20: continue                       # drop specks
  y1, x1, y2, x2 = r.bbox                        # note the order: rows first
  cy, cx = r.centroid
  blobs.append({'area': r.area, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'cx': cx, 'cy': cy})
print(len(blobs))                                # a count
```
Noise cleanup before labeling:
```python
from scipy import ndimage
mask = ndimage.binary_opening(mask, iterations=1)    # removes specks
mask = ndimage.binary_closing(mask, iterations=1)    # fills small holes
mask = ndimage.binary_dilation(mask, iterations=1)   # reconnects broken strokes
```

### Projections (staff lines, column splits)
```python
row_ink = mask.mean(axis=1)                      # (H,) share of dark pixels per row
staff_rows = np.where(row_ink > 0.5)[0]          # long horizontal lines
mask_no_lines = mask.copy()
mask_no_lines[staff_rows, :] = False

col_ink = mask_no_lines.any(axis=0)              # (W,) does this column contain ink
segments = []; inside = False
for x in range(len(col_ink)):
  if col_ink[x] and not inside: start = x; inside = True
  if not col_ink[x] and inside: segments.append((start, x)); inside = False
if inside: segments.append((start, len(col_ink)))
```
Grouping consecutive rows into lines (5 staff lines from many marked rows):
```python
lines = []; current = [staff_rows[0]]
for r in staff_rows[1:]:
  if r - current[-1] <= 1: current.append(r)
  else: lines.append(np.mean(current)); current = [r]
lines.append(np.mean(current))                   # list of line centers
```

### Rotation search
```python
best_angle = 0; best_score = -1
for angle in range(-30, 31):
  rot = ndimage.rotate(gray, angle, reshape=False, order=1, cval=255)
  score = (rot < 128).mean(axis=1).var()         # sharp horizontal lines -> high variance of row ink
  if score > best_score: best_score = score; best_angle = angle
gray = ndimage.rotate(gray, best_angle, reshape=False, order=1, cval=255)
```
Shift search (template vs image): loop `dy, dx` over a range, `np.abs(template - np.roll(img, (dy, dx), axis=(0, 1))).mean()`, keep the minimum.

### Crops -> KNN (template classification)
```python
crops = []
for b in blobs:
  crop = gray[b['y1']:b['y2'], b['x1']:b['x2']]
  crops.append(crop_center_resize(255 - (255 - crop), size=28))   # reuse block A's function
Xc = np.array(crops).reshape(len(crops), -1) / 255.0             # (n_blobs, 784)
knn = KNeighborsClassifier(n_neighbors=3).fit(Xc_labeled, y_labeled)
```
Exact repeated symbol: `res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)`, positions where `res > 0.8`.

### Note position relative to staff lines
```python
spacing = (lines[-1] - lines[0]) / 4
pitch_step = round((lines[-1] - cy) / (spacing / 2))   # half steps above the bottom line
```

### Hand-written convolution and edge filters
```python
def conv2d_manual(img, kernel):                  # img (H, W) float, kernel (F, F); output (H-F+1, W-F+1)
  H, W = img.shape; Fh, Fw = kernel.shape
  out = np.zeros((H - Fh + 1, W - Fw + 1))
  for i in range(H - Fh + 1):
    for j in range(W - Fw + 1):
      total = 0.0
      for a in range(Fh):
        for b in range(Fw):
          total += img[i + a, j + b] * kernel[a, b]
      out[i, j] = total
  return out
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
edges = np.abs(conv2d_manual(img, sobel_x)) + np.abs(conv2d_manual(img, sobel_x.T))
# fast check: F.conv2d(torch.tensor(img)[None, None].float(), torch.tensor(sobel_x)[None, None].float())
```

-----

## I. Anomalies: reconstruction error and template distance

Autoencoder given (or trained on normal images). In: `X_t` `(N, C, H, W)`. Out: `score` `(N,)`.
```python
ae.eval(); scores = []
with torch.no_grad():
  for start in range(0, len(X_t), 256):
    xb = X_t[start:start + 256].to(DEVICE)
    rec = ae(xb)                                         # same shape as xb
    err = ((rec - xb) ** 2).mean(dim=(1, 2, 3))          # (B,) one number per image
    scores.append(err.cpu().numpy())
score = np.concatenate(scores)
plt.hist(score, bins=500); plt.show()
thr = np.percentile(score, 95)                           # or the valley read off the histogram
flag = score > thr
```
Training a tiny autoencoder on normal images (when none is given):
```python
class TinyAE(nn.Module):
  def __init__(self, c=1):
    super().__init__()
    self.enc = nn.Sequential(nn.Conv2d(c, 16, 3, stride=2, padding=1), nn.ReLU(),     # H -> H/2
                             nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU())    # H/2 -> H/4
    self.dec = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
                             nn.ConvTranspose2d(16, c, 4, stride=2, padding=1), nn.Sigmoid())
  def forward(self, x): return self.dec(self.enc(x))
# train with F.mse_loss(ae(xb), xb), Adam 1e-3, 20 epochs, on normal images only
```
Template distance. In: `refs` `(R, H, W)` clean templates, `img` `(H, W)`. Out: one number.
```python
best = 1e9
for r in range(len(refs)):
  d = np.abs(refs[r].astype(float) - img.astype(float)).mean()
  if d < best: best = d
```
Resnet features version: `score = np.linalg.norm(F_test - F_normal.mean(axis=0), axis=1)` or `IsolationForest` (unsupervised.md section 7).

-----

## J. Segmentation: classical, tiny U-Net, Dice

Classical. In: `rgb` `(H, W, 3)`. Out: `mask` `(H, W)` bool.
```python
r = rgb[:, :, 0].astype(float); g = rgb[:, :, 1].astype(float); b = rgb[:, :, 2].astype(float)
mask = (g > r + 20) & (g > b + 20)              # "green things"; adjust per task
mask = ndimage.binary_opening(mask, iterations=2)
labels = measure.label(mask)
for reg in measure.regionprops(labels):
  if reg.area < 100: mask[labels == reg.label] = False
```
Multispectral index: `ndvi = (nir - red) / (nir + red + 1e-6)`, threshold around 0.3 for vegetation.

Tiny U-Net. In: `(B, 3, H, W)` with H, W divisible by 4. Out: logits `(B, 1, H, W)`.
```python
class TinyUNet(nn.Module):
  def __init__(self, in_ch=3, c=32):
    super().__init__()
    def block(a, b): return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU())
    self.e1 = block(in_ch, c); self.e2 = block(c, 2 * c); self.e3 = block(2 * c, 4 * c)
    self.d2 = block(4 * c + 2 * c, 2 * c); self.d1 = block(2 * c + c, c)
    self.out = nn.Conv2d(c, 1, 1)
  def forward(self, x):
    e1 = self.e1(x)                                   # (B, c, H, W)
    e2 = self.e2(F.max_pool2d(e1, 2))                 # (B, 2c, H/2, W/2)
    e3 = self.e3(F.max_pool2d(e2, 2))                 # (B, 4c, H/4, W/4)
    u2 = F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=False)
    d2 = self.d2(torch.cat([u2, e2], dim=1))          # skip connection: cat along channels
    u1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
    d1 = self.d1(torch.cat([u1, e1], dim=1))
    return self.out(d1)                               # (B, 1, H, W) logits
```
Loss and metric:
```python
def dice_loss(logits, target):                        # target (B, 1, H, W) float 0/1
  p = torch.sigmoid(logits)
  inter = (p * target).sum(dim=(1, 2, 3))
  return (1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)).mean()
loss = F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)

def dice_np(pred_mask, true_mask):                    # both (H, W) bool
  inter = (pred_mask & true_mask).sum()
  return 2 * inter / (pred_mask.sum() + true_mask.sum() + 1e-6)
def iou_mask(pred_mask, true_mask):
  return (pred_mask & true_mask).sum() / ((pred_mask | true_mask).sum() + 1e-6)
```
Prediction and threshold scan: `prob = torch.sigmoid(logits)`, then `for t in [0.3, 0.4, 0.5, 0.6]: mean dice of (prob > t)` on validation. Multi-class masks: `Conv2d(c, K, 1)`, `CrossEntropyLoss` with target `(B, H, W)` long, prediction `argmax(dim=1)`.

Loading a mask: `np.array(Image.open(p).convert('L')) > 127` -> bool `(H, W)`; to tensor `torch.from_numpy(m).float().unsqueeze(0)` -> `(1, H, W)`.

Pretrained segmentation head (when allowed): `torchvision.models.segmentation.fcn_resnet50(weights='DEFAULT')`, replace `model.classifier[4] = nn.Conv2d(512, K, 1)`, output `model(x)['out']` is `(B, K, H, W)`.

-----

## K. Restoration

Solve for a fixed linear filter from (corrupted, clean) pairs. In: patches. Out: the filter.
```python
# hypothesis: each clean pixel = weighted sum of a 3x3 corrupted neighborhood
rows = []; targets = []
for (c, k) in pairs[:5]:                           # a few images are enough
  for i in range(1, c.shape[0] - 1, 4):
    for j in range(1, c.shape[1] - 1, 4):
      rows.append(c[i - 1:i + 2, j - 1:j + 2].ravel()); targets.append(k[i, j])
w, _, _, _ = np.linalg.lstsq(np.array(rows, dtype=float), np.array(targets, dtype=float), rcond=None)
kernel = w.reshape(3, 3)                            # apply with ndimage.convolve(corrupted, kernel)
```
Residual CNN (same size in and out, no pooling):
```python
class Denoiser(nn.Module):
  def __init__(self, c=3):
    super().__init__()
    self.net = nn.Sequential(nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
                             nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                             nn.Conv2d(32, c, 3, padding=1))
  def forward(self, x): return (x + self.net(x)).clamp(0, 1)   # predicts the correction
# loss = F.l1_loss(model(corrupted), clean); baselines: F.mse_loss(corrupted, clean) and the mean image
def psnr(a, b): return 10 * np.log10(1.0 / (((a - b) ** 2).mean() + 1e-10))   # a, b in [0, 1]
```

-----

## L. CLIP zero-shot and image features

In: PIL images, list of class words. Out: `scores` `(N, K)`.
```python
from transformers import CLIPModel, CLIPProcessor
clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEVICE).eval()
proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

prompts = ['a photo of a ' + w for w in class_words]
with torch.no_grad():
  t = proc(text=prompts, return_tensors='pt', padding=True).to(DEVICE)
  T = clip.get_text_features(**t)                        # (K, 512)
  T = T / T.norm(dim=1, keepdim=True)
  img_feats = []
  for start in range(0, len(pil_images), 32):
    p = proc(images=pil_images[start:start + 32], return_tensors='pt').to(DEVICE)
    I = clip.get_image_features(**p)                     # (B, 512)
    img_feats.append(I / I.norm(dim=1, keepdim=True))
  I = torch.cat(img_feats)                               # (N, 512)
scores = (I @ T.T).cpu().numpy()                         # (N, K) cosine
pred = scores.argmax(axis=1)
```
Flip TTA: encode the flipped images too and average `I`. CLIP image features also work as `(N, 512)` input for LogisticRegression or for image-to-image matching (`I @ I_other.T`).

-----

## M. Domain shift normalization (per image)

In: `gray` `(H, W)`. Out: same shape, comparable across scanners.
```python
def normalize_image(gray):
  x = gray.astype(float)
  if x.mean() < 127: x = 255 - x                  # make the background bright (dark ink convention)
  x = (x - x.mean()) / (x.std() + 1e-6)           # per image standardization
  return x
# then the same function on train and test; optionally ndimage.gaussian_filter(x, 0.7) to kill scanner texture
```
Check: histogram of train vs test after the function (block 0).

-----

## N. Submission

```python
sub = pd.DataFrame({'id': test_ids, 'label': [classes[p] for p in pred]})
# counts: np.round(pred).astype(int); boxes: clip to [0, W] / [0, H]; masks: (prob > t).astype(int)
print(sub.shape); print(sub.head()); print(sub['label'].value_counts(normalize=True))
sub.to_csv('submission.csv', index=False)
```
Masks as text (run-length or flattened) follow the statement's format exactly; write one, read it back, compare with the mask.
