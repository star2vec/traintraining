# CV-methods.md — image problems: types, approaches, traps

Model settings (CNN layers, resnet fine tune lr/epochs, loss by target type) are in setari-model.md sections 4 and 7. This file is about recognizing the problem type and picking the approach. Code blocks go in CV-methods-code.md.
Sources: the notebooks in this repo, the ONIA CV lot guide, and task lists from IOAI, NEOAI, Polish, Kazakh and Georgian selections.

-----

## 0. First five minutes on any image problem

1. What goes in: photos, small synthetic images (digits, symbols), scans, satellite / multispectral, a pretrained model's embeddings, an image plus a caption?
2. What comes out: one class, several labels, a number (count, fraction), a box, a mask per pixel, "normal or odd", a restored image, a ranking?
3. Metric: accuracy, f1, MAE on a count, IoU of boxes, Dice of masks, a score that penalizes parameter count or inference time? The validation function must compute exactly that.
4. Look at 5 images per class, and 10 from test. Sizes (`img.size`), color mode (`img.mode`: L, RGB, RGBA), value range (0..255 or 0..1), background, whether the object is centered and alone.
5. Does test look like train? Different scanner, background, resolution, inverted colors, noise: fix it in preprocessing before any model (panda-mnist).
6. Are there groups (scanner, patient, scene, video)? Validation must split by group.

-----

## 1. Problem types

### 1a. Classify small synthetic images (digits, symbols, icons, tiles)
Seen: panda-mnist (digits from several scanners, parameter budget), bizantin (Byzantine notation symbols), tastatura (keyboard keys).
- First: flatten pixels to `(N, H*W)` in [0, 1], KNN(3) or LogisticRegression. Two minutes, often 90%+.
- Second: small CNN from scratch, Adam 1e-3, 10..30 epochs, shift augmentation of 2 px (`torch.roll`) and flip only if the symbol stays the same symbol.
- Before either: crop to the object (bounding box of the ink), pad to square, resize to one size. Centering is worth more than any model change on symbol data.
- Parameter budget in the score: `sum(p.numel() for p in model.parameters())`; 8/16/32 channels, no BatchNorm, AdaptiveAvgPool before the last Linear. Compute the score formula on validation, not just accuracy.
- Trap: pretrained resnet is overkill and often worse here; it wants 224 px RGB photos.

### 1b. Classify real photos
Seen: ducks (bird species), Georgia selection (train a classifier from captions only), IOAI antique painting authentication.
- First: pretrained resnet18 as a feature extractor (`fc = Identity`), `(N, 512)` -> LogisticRegression. Fast, strong.
- Second: fine tune resnet18/34, AdamW 1e-4, 5..15 epochs, flip + small rotation + color jitter, best epoch kept.
- Third, when time allows: TTA (image + flip, average), 2 seeds averaged.
- Few labels per class (under 20): features + LR or KNN; fine tuning overfits.
- Labels given only as text (captions, class names): CLIP zero-shot gives the first labels; train a classifier on the confident ones (1k).
- Trap: ImageNet normalization only for pretrained models; forgetting `model.eval()` at prediction.

### 1c. Several labels per image, or label plus number plus box
Seen: ducks (species + box + fill fraction in one network), Polish OAI multi-label.
- One backbone, one `fc` with all outputs stacked: `K` logits for the class, 4 numbers for the box, 1 for a fraction. One loss per piece, weighted sum.
- Multi-label (several tags, each yes/no): `Linear(.., K)` + `BCEWithLogitsLoss`, predict `sigmoid > threshold` per column, threshold per column tuned on validation for f1.
- Box as a target: predict `[x1, y1, x2, y2]` normalized to [0, 1] through sigmoid, `L1Loss` + IoU loss; evaluate with real IoU.
- Trap: the pieces of the loss have different scales; weights of 5..10 on the small ones (ducks: `bce + 5*l1_box + giou + 10*l1_fill`).

### 1d. Count things in an image
Seen: IOAI 2025 Chicken Counting, Polish OAI coin counting (detector from scratch), box-avalanche (how many boxes are real).
- First: classical. Threshold -> `measure.label` -> `regionprops`, count components with area in a plausible range. Touching objects: the total ink area divided by the typical single-object area.
- Second: regression. Resnet features -> Ridge or HGB on the count, or a CNN with one output and `L1Loss`. Round at the end.
- Third: a detector (pretrained Faster R-CNN from torchvision) and count the boxes above a confidence threshold, tuned on train with the official metric.
- Trap: MAE on counts rewards predicting the median when unsure; always check what the dummy "median count" scores.

### 1e. Find and clean boxes (detection post-processing)
Seen: box-avalanche (too many candidate boxes, keep the right ones).
- Hand written NMS: sort by score, keep a box, drop all with IoU above a threshold with it, repeat. Threshold chosen on train with the official metric.
- Second step: a classifier on per-box features (score, rank in the image, size, aspect ratio, overlap with better boxes, number of boxes overlapping it) -> CatBoost / HGB with GroupKFold by image, keep boxes with proba above a tuned threshold.
- Trap: box format. `[x1, y1, x2, y2]` vs `[x, y, w, h]` vs normalized; print one box over its image before anything else.

### 1f. Synthetic images with rules (staff lines, scanned text, grids, game boards)
Seen: music (notes on a staff), tastatura (keys on a keyboard), bizantin (symbols on a line).
- Classical processing beats learning here. Threshold `gray < 128` for ink. Row projections `mask.mean(axis=1)` find long horizontal lines; column projections `mask.any(axis=0)` cut symbols apart. `measure.label` + `regionprops` give area, bbox, centroid per blob.
- Rotation: scan angles in `range(-30, 31)`, rotate with `scipy.ndimage.rotate(img, angle, reshape=False)`, pick the angle whose row projection is sharpest (highest variance).
- Position of a note = centroid row relative to the staff lines; which key = centroid column relative to key boundaries.
- Template matching: crop each symbol, resize to 28x28, KNN against labeled crops; or `cv2.matchTemplate` when the symbol is exactly repeated.
- Observation kept from before: when it is about rotations and translations of the whole image, search over an angle and a shift explicitly; the search is cheap.
- Trap: `measure.label` connects diagonally by default (`connectivity=2`); staff lines merge all notes into one blob, so remove the lines first (row mask) and reconnect broken symbols with a small dilation.

### 1g. Which images are odd (anomaly, out of distribution, real vs generated)
Seen: popoluske (autoencoder given, find corrupted images), NOAI real vs fake, Polish OOD detection, IOAI antique authentication.
- Autoencoder given or trained on normal images: reconstruction error per image -> histogram with 500 bins -> threshold in the valley.
- Distance to reference: `np.abs(refs - img).mean()` to the nearest clean template; or resnet features -> distance to the mean of normal features, or IsolationForest on features.
- With a few labeled odd ones: features -> LogisticRegression; or tune the error threshold for f1 on them.
- Real vs generated: resnet features + LR first; then a CNN on high frequency content (image minus its blurred version), since generators smooth textures.
- Trap: the threshold decides the score; never leave it at a default. Plot the two score distributions if any labels exist.

### 1h. Mask per pixel (segmentation)
Seen at other olympiads: NEOAI cartoon segmentation, Polish multispectral segmentation. In this repo only the classical version (bizantin, music masks).
- First, classical: threshold on color or intensity, cleanup with `binary_opening` / `binary_closing`, keep components above a minimum area. Satellite bands: thresholds on band ratios (`(nir - red) / (nir + red)` for vegetation).
- Second: a tiny U-Net (2 down, 2 up, skip connections) with `BCEWithLogitsLoss` + Dice loss, 128 px crops, flips. Few hours of GPU not needed; 20 epochs on a few hundred images works.
- Third: torchvision's pretrained `fcn_resnet50` / `deeplabv3` with the last layer replaced, fine tuned.
- Metric is IoU or Dice per image: compute it on validation with a threshold scan over `sigmoid > t`.
- Observation kept from before: resize masks with nearest neighbor interpolation (`Image.NEAREST`, `interpolation='nearest'`), never bilinear, or the mask gets values that are not 0/1 and class ids get averaged into nonsense.
- Trap: a mask and its image must get the same geometric augmentation (same flip, same crop); color jitter goes on the image only.

### 1i. Restore or transform an image (denoise, undo a filter, deblur, colorize)
Seen at other olympiads: Kazakhstan image restoration (undo 2x2 channel filters), Polish noise estimation, IOAI 2024 SDXL weight editing.
- First: figure out the corruption with pairs (corrupted, clean) if given: is it a fixed linear filter? Solve for it with least squares on a few pixels (`np.linalg.lstsq`). Is it noise? Estimate its std from flat regions.
- Second: a small CNN that maps corrupted -> clean, `L1Loss` or `MSELoss`, no pooling (same size in and out), residual output (`clean = corrupted + net(corrupted)`).
- Metric usually PSNR or MSE; the "return the input unchanged" baseline must be computed.
- Trap: predicting the mean image scores decently on MSE and is the first thing to beat.

### 1j. Match images to text or to other images (retrieval)
Seen: magia-cuvintelor (CLIP), IOAI 2025 restroom icon matching, polyglot (matching embeddings).
- CLIP zero-shot: image features and text features normalized, `scores = img @ txt.T`, argmax. Prompts "a photo of a {word}" beat bare words. Flip TTA is free.
- Image to image: resnet or CLIP image features, cosine, mutual best match for anchors.
- Fine tune CLIP only with lr 1e-6 and a few epochs; zero-shot is the score to beat.
- Trap: text and image features from different models do not share a space; CLIP's two towers are the pair.

### 1k. Compute under a constraint (parameter budget, pixel budget, inference time, underfitting fix)
Seen: panda-mnist (parameter budget), IOAI 2025 Pixel Efficiency (few pixels), NEOAI underfitting CV.
- Read the score formula and compute it on validation from the start; the best accuracy is not the best score.
- Pixel budget: downsample (`resize` to 8x8 / 16x16) and check how accuracy decays; pick the smallest size that keeps it.
- Underfitting model given: lr too low, too few epochs, missing normalization, a frozen layer, a wrong loss. Check in that order; the fix is usually one line.
- Trap: BatchNorm costs 2 parameters per channel and bias 1 per channel; `bias=False` and no BatchNorm when counting matters.

### 1l. Learn convolution itself (write a filter, apply it, detect edges)
Seen at other olympiads: Polish convolutional filters, HAIO.
- A convolution by hand: two loops over rows and columns, a kernel of 3x3, sum of products. Edge kernels: `[[-1,0,1],[-2,0,2],[-1,0,1]]` (vertical edges), its transpose (horizontal), blur `np.ones((3,3))/9`.
- Output size: `(H - F + 2P) / S + 1`.
- `torch.nn.functional.conv2d(img[None, None], kernel[None, None])` to check the hand version.

-----

## 2. Shared building blocks

### Loading and shapes
`Image.open(path).convert('RGB')` -> `np.array(img)` is `(H, W, 3)` uint8 0..255. Gray: `.convert('L')` -> `(H, W)`. Torch wants `(N, C, H, W)` float: `ToTensor()` does both the reorder and the 0..1 scaling. `img.size` is `(W, H)`.

### Normalization
Pretrained: ImageNet mean/std. From scratch: 0..1 is enough, or subtract the dataset mean. Always the same choice for train, validation and test.

### Augmentation rules
Keep the label true: flip left-right for photos, never for digits or text; rotation up to ~15 degrees for photos, 0 for anything with a fixed orientation; shift by 2 px always safe. Masks and boxes get the same geometric transform; color changes go only on the image. Validation and test get no augmentation.

### Domain shift fix (panda-mnist pattern)
When test comes from another scanner / camera: per image, normalize by its own mean and std; invert if the background is bright; blur or threshold to remove scanner texture. Compare train and test histograms before and after.

### Classical toolbox
Threshold (`gray < 128`, `threshold_otsu`), `measure.label`, `regionprops` (area, bbox, centroid), projections (`mask.any(axis=0)`, `mask.mean(axis=1)`), `ndimage.rotate`, `binary_opening/closing`, `cv2.findContours`, `cv2.matchTemplate`, resize with `Image.NEAREST` for masks.

### Two cheap wins
Resnet features + LogisticRegression (any photo task, 2 minutes). TTA with the horizontal flip (any CNN, free).

-----

## 3. Validation on images

- Stratified split 80/20 or StratifiedKFold(5) when training is fast (features + LR); a single split when training a network.
- GroupKFold when images share a scanner, scene, patient, video or source document; evaluate per group too (panda-mnist: accuracy per scanner, then the mean).
- Same preprocessing function for train, validation and test, called once in one place.
- Near-duplicate images across train and test: predict them directly with the train label.
- Compute the contest score formula on validation, not only accuracy.

-----

## 4. Baselines that take under five minutes

| situation | baseline |
|---|---|
| any classification | most frequent class |
| small synthetic images | flatten + KNN(3) |
| real photos | resnet18 features + LogisticRegression |
| counting | threshold + `measure.label`, or the median count |
| boxes | NMS on the given candidates |
| anomaly | reconstruction or template distance + percentile threshold |
| segmentation | color / intensity threshold + `binary_opening` |
| restoration | return the input; the mean image |
| image to text | CLIP zero-shot with "a photo of a {word}" |

-----

## 5. Traps

- `img.size` is `(W, H)`; `np.array(img).shape` is `(H, W, C)`. Boxes are `[x1, y1, x2, y2]` with x along the width.
- Mixing 0..255 and 0..1 inputs in one model.
- RGBA or palette images (`mode` P / RGBA) slipping into a RGB pipeline: `convert('RGB')` on load, `convert('L')` for gray.
- Masks resized with bilinear interpolation; class ids averaged.
- Augmenting validation or test.
- `model.eval()` + `torch.no_grad()` missing at prediction; BatchNorm and Dropout then behave differently.
- `os.listdir` order is random; ids from filenames, sorted.
- A DataLoader with `shuffle=True` on test: predictions no longer match the ids.
- Threshold left at 0.5 / default for anomalies, masks, multi-label and detection confidence.
- Training on all scanners/groups and validating on random rows: the score is inflated and the test, from an unseen group, is worse.
- Pretrained resnet on 28 px gray digits: resize to 224 and repeat the channel if it must be used, but KNN or a small CNN is the right tool.
- Forgetting to round counts, clip boxes to the image, or cast masks back to 0/1 before writing the submission.
