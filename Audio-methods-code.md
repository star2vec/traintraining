# Audio-methods-code.md — code blocks for every approach in Audio-methods.md

Each block says what goes in, what comes out (type and shape), then the code. Spectrograms are written by hand with numpy so every step is visible; the library versions come after. Blocks A to E need only numpy and scipy.

Imports used everywhere:
```python
import numpy as np, pandas as pd, os, copy
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import dct
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); np.random.seed(0)
```

-----

## 0. Load and look

In: path of a wav. Out: `sr` int, `x` numpy `(samples,)` float32 in [-1, 1].
```python
def load_wav(path):
  sr, x = wavfile.read(path)                   # x is (samples,) or (samples, channels); the dtype says the scale
  if x.dtype == np.uint8: x = (x.astype(np.float32) - 128) / 128      # 8 bit is unsigned, silence is 128
  elif x.dtype == np.int16: x = x.astype(np.float32) / 32768
  elif x.dtype == np.int32: x = x.astype(np.float32) / 2147483648
  else: x = x.astype(np.float32)               # already float
  if x.ndim == 2: x = x.mean(axis=1)           # stereo -> mono
  return sr, x

def fix_length(x, n):                          # pad with zeros or crop, so clips can be stacked
  if len(x) >= n: return x[:n]
  out = np.zeros(n, dtype=np.float32)
  out[:len(x)] = x
  return out

waves = []
for i in range(len(train)):
  sr, x = load_wav(train['path'].iloc[i])
  waves.append(fix_length(x, sr * 1))          # 1 second
X_wave = np.array(waves)                       # (N, samples)
```
Other formats:
```python
d = np.load('train_data.npz'); print(d.files); X_wave = d['audio']       # arrays stored in an npz
import librosa
x, sr = librosa.load(path, sr=16000, mono=True)                          # mp3 or wav, resampled, float32
import soundfile as sf
x, sr = sf.read(path, dtype='float32')                                   # (samples,) or (samples, channels)
from scipy.signal import resample_poly
x16 = resample_poly(x, 16000, sr).astype(np.float32)                     # change the sample rate
```
First look:
```python
rows = []
for path in paths[:300]:
  sr, raw = wavfile.read(path)
  rows.append([sr, len(raw), str(raw.dtype), raw.ndim, raw.min(), raw.max()])
info = pd.DataFrame(rows, columns=['sr', 'samples', 'dtype', 'ndim', 'min', 'max'])
print(info['sr'].value_counts()); print(info['samples'].describe()); print(info['dtype'].value_counts())
# with labels: info['label'] = ...; print(info.groupby('label')[['sr', 'samples', 'max']].mean())   # leak check

from IPython.display import Audio, display
display(Audio(x, rate=sr))                     # listen
plt.plot(x); plt.show()                        # waveform
plt.specgram(x, NFFT=256, Fs=sr); plt.show()   # quick spectrogram: time left to right, frequency bottom to top
```

-----

## A. Spectrum of a whole clip, filters

In: `x` `(samples,)`. Out: `spec` `(samples//2 + 1,)` magnitude per frequency, `freqs` the frequency in Hz of each bin.
```python
spec = np.abs(np.fft.rfft(x))
freqs = np.fft.rfftfreq(len(x), d=1 / sr)      # from 0 to sr/2
print('loudest frequency', freqs[spec.argmax()])

# mean spectrum per class (clips of equal length)
specs = []
for i in range(len(X_wave)): specs.append(np.abs(np.fft.rfft(X_wave[i])))
specs = np.array(specs)                        # (N, bins)
for c in sorted(set(y)):
  plt.plot(freqs, np.log(specs[y == c].mean(axis=0) + 1e-10), label=str(c))
plt.legend(); plt.xlabel('Hz'); plt.show()

power = specs ** 2
print('share of energy above 350 Hz', power[:, freqs > 350].sum() / power.sum())
```
Low pass (remove everything above a cutoff):
```python
def lowpass_fft(x, sr, cutoff):
  X = np.fft.rfft(x)
  freqs = np.fft.rfftfreq(len(x), d=1 / sr)
  X[freqs > cutoff] = 0                        # high pass: X[freqs < cutoff] = 0
  return np.fft.irfft(X, n=len(x)).astype(np.float32)

from scipy.signal import butter, sosfiltfilt   # the standard filter, smoother edges
sos = butter(6, 350, btype='low', fs=sr, output='sos')                 # band: butter(4, [300, 3400], btype='band', ...)
x_clean = sosfiltfilt(sos, x).astype(np.float32)
```

-----

## B. Spectrogram, log-mel, MFCC by hand

In: `x` `(samples,)`. Out: `S` `(n_fft//2 + 1, frames)`; row `k` is the frequency `k * sr / n_fft`, column `t` starts at sample `t * hop`.
```python
def stft_mag(x, n_fft=512, hop=160):
  window = np.hanning(n_fft)                   # fades the edges of each piece
  n_frames = 1 + (len(x) - n_fft) // hop       # needs len(x) >= n_fft
  S = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
  for t in range(n_frames):
    frame = x[t * hop : t * hop + n_fft] * window
    S[:, t] = np.abs(np.fft.rfft(frame))
  return S
```
Sizes: window ~25 ms, hop ~10 ms. 16 kHz: `n_fft=512, hop=160`. 1024 Hz: `n_fft=128, hop=32`.

Mel bands: `fb` `(n_mels, n_fft//2 + 1)`, one triangle per band.
```python
def hz_to_mel(f): return 2595 * np.log10(1 + f / 700)
def mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)

def mel_filterbank(sr, n_fft, n_mels=40):
  hz_points = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2))   # band edges, evenly spaced in mel
  bin_freqs = np.fft.rfftfreq(n_fft, d=1 / sr)
  fb = np.zeros((n_mels, len(bin_freqs)), dtype=np.float32)
  for m in range(n_mels):
    left = hz_points[m]; center = hz_points[m + 1]; right = hz_points[m + 2]
    for k in range(len(bin_freqs)):
      f = bin_freqs[k]
      if left < f <= center: fb[m, k] = (f - left) / (center - left)       # rising side
      elif center < f < right: fb[m, k] = (right - f) / (right - center)   # falling side
  return fb

fb = mel_filterbank(sr, 512, 40)               # build once
S = stft_mag(x, 512, 160)
logmel = np.log(fb @ (S ** 2) + 1e-10)         # (40, frames)
mfcc = dct(logmel, type=2, axis=0, norm='ortho')[:13]     # (13, frames)
plt.imshow(logmel, aspect='auto', origin='lower'); plt.show()
```
Check `(fb.sum(axis=1) == 0).sum()` is 0; empty bands mean too many mel bands for this `n_fft`.

Libraries (same shapes; a few more frames because they pad the ends):
```python
import librosa
logmel = librosa.power_to_db(librosa.feature.melspectrogram(y=x, sr=sr, n_fft=512, hop_length=160, n_mels=40))
mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
import torchaudio
mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=512, hop_length=160, n_mels=40)
logmel = torch.log(mel(torch.from_numpy(x)) + 1e-10)      # works on a batch (N, samples) too
```

-----

## C. Clip features + LogisticRegression

In: `X_wave` `(N, samples)`. Out: `X` `(N, 119)` one row per clip (40 + 40 + 13 + 13 + 13), `X_flat` `(N, n_mels * T)`.
```python
N_FFT = 512; HOP = 160
fb = mel_filterbank(sr, N_FFT, 40)

def clip_logmel(x):
  S = stft_mag(x, N_FFT, HOP)
  return np.log(fb @ (S ** 2) + 1e-10)         # (n_mels, T)

def clip_features(x):
  logmel = clip_logmel(x)
  mfcc = dct(logmel, type=2, axis=0, norm='ortho')[:13]
  delta = mfcc[:, 1:] - mfcc[:, :-1]           # change from one frame to the next
  return np.concatenate([logmel.mean(axis=1), logmel.std(axis=1), mfcc.mean(axis=1), mfcc.std(axis=1), delta.std(axis=1)])

feats = []; flats = []
for i in range(len(X_wave)):
  feats.append(clip_features(X_wave[i]))
  flats.append(clip_logmel(X_wave[i]).reshape(-1))        # whole spectrogram, keeps the order in time
X = np.array(feats); X_flat = np.array(flats)

fit_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=0)
scaler = StandardScaler().fit(X[fit_idx])
clf = LogisticRegression(C=1, max_iter=2000).fit(scaler.transform(X[fit_idx]), y[fit_idx])
print(accuracy_score(y[val_idx], clf.predict(scaler.transform(X[val_idx]))))
```
Run it on `X` and on `X_flat`; average the two sets of probabilities if both are good. Clips sharing a speaker: `GroupKFold` instead of `train_test_split`.

-----

## D. Spectrogram -> small CNN

In: `X_wave` `(N, samples)`. Out: `X_t` `(N, 1, n_mels, T)` float, ready for `SmallCNN(in_ch=1, n_classes=K)` and the training loop of CV-methods-code D.
```python
L = []
for i in range(len(X_wave)): L.append(clip_logmel(X_wave[i]))
L = np.array(L, dtype=np.float32)              # (N, n_mels, T)
mean = L[fit_idx].mean(); std = L[fit_idx].std()          # two numbers, from the fit rows
X_t = torch.from_numpy((L - mean) / std).unsqueeze(1)     # (N, 1, n_mels, T)
y_t = torch.from_numpy(y).long()
# test: same mean and std
```
In the loop replace `shift(X_t[idx])` with `spec_mask(X_t[idx])` from block E. With BatchNorm the validation score jumps around in the first epochs: run 20 or more and keep the best.

-----

## E. Augmentation

```python
def aug_wave(x):                               # on the waveform, before the spectrogram
  x = np.roll(x, np.random.randint(-len(x) // 10, len(x) // 10 + 1))   # shift up to 10% (wraps around)
  x = x * np.random.uniform(0.7, 1.3)                                  # volume
  x = x + 0.005 * np.random.randn(len(x))                              # light noise
  return x.astype(np.float32)

def add_noise_snr(x, noise, snr_db):           # mix a noise clip at a chosen signal to noise ratio
  p_x = np.mean(x ** 2); p_n = np.mean(noise ** 2) + 1e-12
  scale = np.sqrt(p_x / (p_n * 10 ** (snr_db / 10)))
  return (x + scale * noise).astype(np.float32)

def spec_mask(spec, max_f=6, max_t=10):        # spec (B, 1, n_mels, T) normalized; hides one band and one time strip
  spec = spec.clone()
  f = np.random.randint(0, max_f + 1); f0 = np.random.randint(0, spec.shape[2] - f + 1)
  t = np.random.randint(0, max_t + 1); t0 = np.random.randint(0, spec.shape[3] - t + 1)
  spec[:, :, f0:f0 + f, :] = 0                 # 0 is the mean after normalization
  spec[:, :, :, t0:t0 + t] = 0
  return spec
```
Long clips: a random crop of fixed length per epoch for training; at prediction average the logits over several crops.

-----

## F. Pretrained encoder -> one vector per clip

The HuBERT block is in nlp-methods-code O. Loading there uses `torchaudio.load`, which on recent torchaudio needs torchcodec; `load_wav` + `resample_poly` from block 0 does the same job.

A chosen layer, mean and std over time. In: 16 kHz `wave` `(samples,)`. Out: `(2 * d,)`.
```python
inputs = proc(wave, sampling_rate=16000, return_tensors='pt').to(DEVICE)
with torch.no_grad():
  out = enc_model(**inputs, output_hidden_states=True)
h = out.hidden_states[6][0]                    # layer 6 of 12 -> (T, d); hidden_states[-1] is last_hidden_state
vec = torch.cat([h.mean(dim=0), h.std(dim=0)]).cpu().numpy()
```
AST (everyday sounds). In: 16 kHz `wave`. Out: `(768,)`, the vector the classification head reads.
```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification
fe = ASTFeatureExtractor.from_pretrained(CHECKPOINT)       # wave -> (1, 1024, 128) log-mel, padded to ~10 s
ast = ASTForAudioClassification.from_pretrained(CHECKPOINT).to(DEVICE).eval()
inputs = fe(wave, sampling_rate=16000, return_tensors='pt').to(DEVICE)
with torch.no_grad():
  emb = ast.audio_spectrogram_transformer(**inputs).pooler_output      # (1, 768)
  logits = ast.classifier.dense(ast.classifier.layernorm(emb))         # the same as ast(**inputs).logits
```
Save once, reuse: `np.save('emb_train.npy', E)`; `E = np.load('emb_train.npy')`.

-----

## G. Speech to text, edit distance, nearest word

In: 16 kHz waves. Out: list of strings.
```python
from transformers import pipeline
asr = pipeline('automatic-speech-recognition', model='openai/whisper-small', device=0 if DEVICE == 'cuda' else -1)
texts = []
for i in range(len(waves16)):
  out = asr({'raw': waves16[i], 'sampling_rate': 16000})
  texts.append(out['text'].lower().strip())
```
Edit distance by hand. In: two strings. Out: number of insertions, deletions and replacements.
```python
def levenshtein(a, b):
  prev = list(range(len(b) + 1))               # distances from the empty prefix of a
  for i in range(1, len(a) + 1):
    cur = [i] + [0] * len(b)
    for j in range(1, len(b) + 1):
      cost = 0 if a[i - 1] == b[j - 1] else 1
      cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
    prev = cur
  return prev[len(b)]

def snap(word, word_list):                     # nearest valid answer
  best = word_list[0]; best_d = levenshtein(word, best)
  for w in word_list[1:]:
    d = levenshtein(word, w)
    if d < best_d: best = w; best_d = d
  return best
```

-----

## H. Speaker clustering with unseen speakers

In: `X_train` `(N, d)` clip features (block C or F) with `spk_train` `(N,)`, `X_test` `(M, d)`. Out: `labels` `(M,)` in 1..K.
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

def cluster_speakers(X_fit, spk_fit, X_new, k):
  scaler = StandardScaler().fit(np.concatenate([X_fit, X_new]))        # no labels used
  lda = LinearDiscriminantAnalysis().fit(scaler.transform(X_fit), spk_fit)
  Z = lda.transform(scaler.transform(X_new))                           # (M, speakers in fit - 1)
  Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
  return AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average').fit_predict(Z)

# validation: 5 whole speakers held out
speakers = np.unique(spk_train); np.random.shuffle(speakers)
is_val = np.isin(spk_train, speakers[:5])
lab = cluster_speakers(X_train[~is_val], spk_train[~is_val], X_train[is_val], 5)
print('ARI with LDA', adjusted_rand_score(spk_train[is_val], lab))
km = KMeans(n_clusters=5, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(X_train[is_val]))
print('ARI plain KMeans', adjusted_rand_score(spk_train[is_val], km))  # the baseline to beat

labels = cluster_speakers(X_train, spk_train, X_test, 10) + 1
print(np.bincount(labels)[1:])                 # cluster sizes; one giant cluster means the features failed
```
Repeat the validation with 3 different held out sets; 5 speakers is a small sample.

-----

## I. Matching: similarity matrix, one to one assignment, DTW

In: `A` `(n, d)`, `B` `(n, d)` rows normalized to length 1. Out: `match` `(n,)`, the row of B for each row of A.
```python
from scipy.optimize import linear_sum_assignment
sims = A @ B.T                                 # (n, n) cosine
greedy = sims.argmax(axis=1)                   # can pick one B row twice
rows, cols = linear_sum_assignment(-sims)      # every B row used once, largest total similarity
match = np.zeros(len(A), dtype=int)
for i in range(len(rows)): match[rows[i]] = cols[i]

# mutual best pairs: the reliable ones
best_b = sims.argmax(axis=1); best_a = sims.argmax(axis=0)
mutual = []
for i in range(len(A)):
  if best_a[best_b[i]] == i: mutual.append((i, best_b[i]))
```
Triplets from three known groups (A, B, C): `match_ab` and `match_ac` as above, triplet `i` is `(a_files[i], b_files[match_ab[i]], c_files[match_ac[i]])`.

Texts instead of vectors: `sims[i, j] = -levenshtein(texts_a[i], texts_b[j])`, then the same assignment.

DTW: distance between two sequences of different lengths. In: `P` `(Ta, d)`, `Q` `(Tb, d)` frame features (`mfcc.T`). Out: a number, small when the content is the same.
```python
def dtw_distance(P, Q):
  Ta = len(P); Tb = len(Q)
  D = np.full((Ta + 1, Tb + 1), np.inf); D[0, 0] = 0
  for i in range(1, Ta + 1):
    for j in range(1, Tb + 1):
      cost = np.linalg.norm(P[i - 1] - Q[j - 1])
      D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])   # stay, stay, or advance both
  return D[Ta, Tb] / (Ta + Tb)
```
Slow in pure Python: keep every 4th frame, and compare only pairs with similar durations.

-----

## J. Fingerprint of the system that made the file

In: `x`, `sr`. Out: `(68,)` features that depend little on what is said.
```python
def fingerprint(x, sr):
  S = stft_mag(x, 1024, 256)
  avg = np.log(S ** 2 + 1e-10).mean(axis=1)    # long term average spectrum (513,)
  edges = np.linspace(0, len(avg), 65).astype(int)
  bands = np.zeros(64, dtype=np.float32)
  for b in range(64): bands[b] = avg[edges[b]:edges[b + 1]].mean()
  extra = [len(x) / sr, np.abs(x).max(), np.sqrt(np.mean(x ** 2)), np.mean(np.abs(x) < 1e-4)]   # length, peak, loudness, share of silence
  return np.concatenate([bands, extra])
```
Then StandardScaler + LogisticRegression, or HGB without scaling. File facts from block 0 (`sr`, `dtype`, `samples`) per class come first.

-----

## K. New classes on a frozen model's head

In: `E` `(N, d)` embeddings from the frozen encoder, `y` `(N,)` in `0..K_NEW-1`, `old_head` with `.layernorm` and `.dense` for `K_OLD` classes. Out: `head` for `K_NEW` classes, exported arrays.
```python
class Head(nn.Module):
  def __init__(self, d, k):
    super().__init__()
    self.layernorm = nn.LayerNorm(d); self.dense = nn.Linear(d, k)
  def forward(self, e): return self.dense(self.layernorm(e))           # (B, d) -> (B, k)

E_t = torch.from_numpy(E).float(); y_t = torch.from_numpy(y).long()
head = Head(E.shape[1], K_NEW)
with torch.no_grad():
  head.layernorm.weight.copy_(old_head.layernorm.weight); head.layernorm.bias.copy_(old_head.layernorm.bias)
  head.dense.weight[:K_OLD] = old_head.dense.weight; head.dense.bias[:K_OLD] = old_head.dense.bias
  Z = head.layernorm(E_t)
  row_norm = old_head.dense.weight.norm(dim=1).mean()
  for c in range(K_OLD, K_NEW):                # new rows from prototypes
    proto = Z[y_t == c].mean(dim=0)
    head.dense.weight[c] = proto / proto.norm() * row_norm
    head.dense.bias[c] = old_head.dense.bias.mean()

def report(h, tag):                            # two numbers, always
  h.eval()
  with torch.no_grad(): p = h(E_val_t).argmax(1).numpy()
  old = y_val < K_OLD
  print(tag, 'old', accuracy_score(y_val[old], p[old]), 'new', accuracy_score(y_val[~old], p[~old]))
report(head, 'prototypes only')
```
Training with class weights and distillation from the old head:
```python
with torch.no_grad(): teacher = old_head(E_t)                           # (N, K_OLD) what the old model said
counts = np.bincount(y, minlength=K_NEW)
w = torch.tensor(1.0 / counts, dtype=torch.float32); w = w / w.mean()   # rare classes weigh more
opt = torch.optim.Adam(head.dense.parameters(), lr=1e-3)                # layernorm not in the optimizer yet
T = 2.0; LAM = 1.0
for epoch in range(100):
  head.train(); perm = np.random.permutation(len(E))
  for start in range(0, len(perm), 64):
    idx = perm[start:start + 64]
    logits = head(E_t[idx])
    ce = F.cross_entropy(logits, y_t[idx], weight=w)
    kd = F.kl_div(F.log_softmax(logits[:, :K_OLD] / T, dim=1), F.softmax(teacher[idx] / T, dim=1), reduction='batchmean') * T * T
    loss = ce + LAM * kd                        # LAM up -> old classes safer, new classes learn slower
    opt.zero_grad(); loss.backward(); opt.step()
  if epoch % 10 == 0: report(head, epoch)
```
Export with the names the statement asks for, then read back and compare predictions:
```python
np.savez('weights.npz',
  classifier_layernorm_weight=head.layernorm.weight.detach().numpy(), classifier_layernorm_bias=head.layernorm.bias.detach().numpy(),
  classifier_dense_weight=head.dense.weight.detach().numpy(), classifier_dense_bias=head.dense.bias.detach().numpy())
chk = np.load('weights.npz')
for k in chk.files: print(k, chk[k].shape, chk[k].dtype)
```

-----

## L. Events per frame from embeddings

In: `E_fit` `(N, T, d)`, `Y_fit` `(N, T, K)` 0/1, `E_val`, `Y_val`. Out: `P` `(M, T, K)` probabilities.
```python
from sklearn.metrics import average_precision_score

def add_context(E, width=2):                   # each frame + `width` neighbors per side; edges repeat the border frame
  N, T, d = E.shape
  out = np.zeros((N, T, d * (2 * width + 1)), dtype=np.float32)
  for t in range(T):
    col = 0
    for o in range(-width, width + 1):
      s = min(max(t + o, 0), T - 1)
      out[:, t, col:col + d] = E[:, s]
      col += d
  return out

def scores(Y, P, tag):                         # the two usual metrics, on flattened frames
  Yf = Y.reshape(-1, Y.shape[2]); Pf = P.reshape(-1, P.shape[2])
  aps = []
  for k in range(Yf.shape[1]):
    if Yf[:, k].sum() > 0: aps.append(average_precision_score(Yf[:, k], Pf[:, k]))
  print(tag, 'macro AP', np.mean(aps), 'micro F1 at 0.5', f1_score(Yf.ravel(), (Pf.ravel() > 0.5).astype(int)))

K = Y_fit.shape[2]
Xf = add_context(E_fit, 2); Xf = Xf.reshape(-1, Xf.shape[2])           # (N*T, 5d)
Xv = add_context(E_val, 2); Xv = Xv.reshape(-1, Xv.shape[2])
scaler = StandardScaler().fit(Xf)
P = np.zeros((len(Xv), K), dtype=np.float32)
for k in range(K):                             # one model per class
  clf = LogisticRegression(C=0.1, max_iter=500).fit(scaler.transform(Xf), Y_fit.reshape(-1, K)[:, k])
  P[:, k] = clf.predict_proba(scaler.transform(Xv))[:, 1]
P = P.reshape(len(E_val), -1, K)
scores(Y_val, P, 'LR + context')
```
GRU over the frames:
```python
class FrameGRU(nn.Module):
  def __init__(self, d, k, h=64):
    super().__init__()
    self.gru = nn.GRU(d, h, batch_first=True, bidirectional=True)
    self.out = nn.Linear(2 * h, k)
  def forward(self, x): return self.out(self.gru(x)[0])                # (B, T, d) -> (B, T, k) logits

d = E_fit.shape[2]
mean = E_fit.reshape(-1, d).mean(axis=0); std = E_fit.reshape(-1, d).std(axis=0) + 1e-6
Xf_t = torch.from_numpy((E_fit - mean) / std).float(); Yf_t = torch.from_numpy(Y_fit).float()
Xv_t = torch.from_numpy((E_val - mean) / std).float()
pos = Y_fit.reshape(-1, K).sum(axis=0); neg = Y_fit.shape[0] * Y_fit.shape[1] - pos
pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 20), dtype=torch.float32).to(DEVICE)   # rare classes count more

model = FrameGRU(d, K).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
for epoch in range(30):
  model.train(); perm = np.random.permutation(len(Xf_t))
  for start in range(0, len(perm), 32):
    idx = perm[start:start + 32]
    loss = loss_fn(model(Xf_t[idx].to(DEVICE)), Yf_t[idx].to(DEVICE))
    opt.zero_grad(); loss.backward(); opt.step()
model.eval()
with torch.no_grad(): P = torch.sigmoid(model(Xv_t.to(DEVICE))).cpu().numpy()
scores(Y_val, P, 'GRU')
```
Best threshold per class moved onto 0.5, then smoothing:
```python
def best_threshold(y, p):
  best_t = 0.5; best_f = -1
  for t in np.arange(0.05, 0.96, 0.05):
    f = f1_score(y, (p > t).astype(int), zero_division=0)
    if f > best_f: best_f = f; best_t = t
  return best_t

def shift_to_half(p, t):                       # p == t becomes 0.5; the order of the values stays
  logit = np.log(p + 1e-7) - np.log(1 - p + 1e-7)
  logit_t = np.log(t) - np.log(1 - t)
  return 1 / (1 + np.exp(-(logit - logit_t)))

thresholds = []
for k in range(K): thresholds.append(best_threshold(Y_val[:, :, k].ravel(), P[:, :, k].ravel()))
# apply to validation (to see the gain) and with the same thresholds to test
for k in range(K): P[:, :, k] = shift_to_half(P[:, :, k], thresholds[k])

def smooth3(P):                                # median of each frame with its two neighbors
  out = P.copy()
  for t in range(1, P.shape[1] - 1):
    out[:, t] = np.median(P[:, t - 1:t + 2], axis=1)
  return out
```
Thresholds tuned and scored on the same validation rows look better than they are; tune on one half, score on the other.

Standardize each scene by itself (try under domain shift, same function for train and test):
```python
def per_scene_norm(E):
  out = np.zeros_like(E)
  for i in range(len(E)):
    out[i] = (E[i] - E[i].mean(axis=0)) / (E[i].std(axis=0) + 1e-6)
  return out
```
Leave one domain out: `for dom in np.unique(domain_ids): fit = domain_ids != dom; val = domain_ids == dom; ...`.

-----

## M. Unknown events and boundaries

In: `P_frame` `(T, C)` class probabilities of one scene, `p_event` `(T,)` probability that any event is on. Out: class per frame, start and end markers, unknown score.
```python
cls = np.full(len(P_frame), -1)                # -1 = background
for t in range(len(P_frame)):
  if P_frame[t].max() > 0.5: cls[t] = P_frame[t].argmax()

def boundaries_from_classes(cls):
  T = len(cls); start = np.zeros(T, dtype=np.float32); end = np.zeros(T, dtype=np.float32)
  for t in range(T):
    if cls[t] == -1: continue
    if t == 0 or cls[t - 1] != cls[t]: start[t] = 1      # first frame of a run
    if t == T - 1 or cls[t + 1] != cls[t]: end[t] = 1    # last frame of a run
  return start, end

# something is on, and it looks like no known class
unknown = p_event * (1 - P_frame[:, :N_KNOWN].max(axis=1))
```
`p_event` comes from a binary model trained with target `Y.max(axis=2)` (any class on).

-----

## N. Audio embeddings to video embeddings

In: `A` `(N, da)`, `V` `(N, dv)`, row `i` of each from the same clip. Out: for each test group of 20, the video index per audio.
```python
from sklearn.linear_model import Ridge
from scipy.optimize import linear_sum_assignment
sa = StandardScaler().fit(A[fit_idx]); sv = StandardScaler().fit(V[fit_idx])
A_s = sa.transform(A).astype(np.float32); V_s = sv.transform(V).astype(np.float32)

def eval_groups(za, zv, tag):                  # row i of za belongs with row i of zv; consecutive groups of 20
  hit_arg = 0; hit_one = 0; total = 0
  for start in range(0, len(za) - 19, 20):
    sims = za[start:start + 20] @ zv[start:start + 20].T
    hit_arg += int((sims.argmax(axis=1) == np.arange(20)).sum())
    rows, cols = linear_sum_assignment(-sims)
    hit_one += int((cols == rows).sum()); total += 20
  print(tag, 'argmax', hit_arg / total, 'one to one', hit_one / total)

ridge = Ridge(alpha=10).fit(A_s[fit_idx], V_s[fit_idx])
pv = ridge.predict(A_s[val_idx]); pv = pv / np.linalg.norm(pv, axis=1, keepdims=True)
vt = V_s[val_idx] / np.linalg.norm(V_s[val_idx], axis=1, keepdims=True)
eval_groups(pv, vt, 'ridge')
```
Two towers with a contrastive loss:
```python
class TwoTowers(nn.Module):
  def __init__(self, da, dv, dz=256):
    super().__init__()
    self.a = nn.Sequential(nn.Linear(da, 512), nn.ReLU(), nn.Linear(512, dz))
    self.v = nn.Sequential(nn.Linear(dv, 512), nn.ReLU(), nn.Linear(512, dz))
  def forward(self, xa, xv):
    return F.normalize(self.a(xa), dim=1), F.normalize(self.v(xv), dim=1)

model = TwoTowers(A.shape[1], V.shape[1]).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
A_t = torch.from_numpy(A_s); V_t = torch.from_numpy(V_s)
for epoch in range(15):
  model.train(); perm = np.random.permutation(len(fit_idx))
  for start in range(0, len(perm), 256):
    idx = fit_idx[perm[start:start + 256]]
    za, zv = model(A_t[idx].to(DEVICE), V_t[idx].to(DEVICE))
    logits = za @ zv.T / 0.07                  # (B, B); the right video for audio i is column i
    target = torch.arange(len(idx)).to(DEVICE)
    loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
    opt.zero_grad(); loss.backward(); opt.step()
  model.eval()
  with torch.no_grad(): za, zv = model(A_t[val_idx].to(DEVICE), V_t[val_idx].to(DEVICE))
  eval_groups(za.cpu().numpy(), zv.cpu().numpy(), epoch)
```
Prediction for one test group: `sims = za_group @ zv_group.T`, `rows, cols = linear_sum_assignment(-sims)`, answer for audio `rows[i]` is `cols[i]`.

-----

## O. Looking for a hidden message

In: path of a wav. Out: prints to read.
```python
sr, raw = wavfile.read(path)                   # keep the integers, no float conversion
print(raw.dtype, raw.min(), raw.max(), len(np.unique(raw)))

bits = raw & 1                                 # lowest bit of every sample
chars = []
for i in range(0, len(bits) - 7, 8):           # 8 bits -> one character
  v = 0
  for j in range(8): v = v * 2 + int(bits[i + j])
  chars.append(chr(v))
print(''.join(chars)[:80])                     # readable text here = found it

x = (raw.astype(np.float32) - 128) / 128       # uint8 case
S = stft_mag(x, 512, 128)
freqs = np.fft.rfftfreq(512, d=1 / sr)
print(freqs[S.argmax(axis=0)][:40])            # loudest frequency in each frame: steps or repeats = a tone code
print(freqs[S[len(S) // 2:].argmax(axis=0) + len(S) // 2][:40])        # the same, upper half of the band only
```
With labels, where does the label show up:
```python
specs = []
for i in range(len(X_wave)): specs.append(np.log(stft_mag(X_wave[i], 512, 128) + 1e-6))
specs = np.array(specs)                        # (N, bins, T)
mean_all = specs.mean(axis=0)
for letter in sorted(set(letters))[:5]:        # letters: numpy array (N,) with the label of each clip
  diff = specs[letters == letter].mean(axis=0) - mean_all
  plt.imshow(diff, aspect='auto', origin='lower'); plt.title(letter); plt.colorbar(); plt.show()
```
Then a classifier on the rows and columns that light up (`specs[:, r0:r1, c0:c1].reshape(N, -1)` + LR), and `snap` from block G for words.

-----

## P. Submission

```python
sub = pd.DataFrame({'id': test_ids, 'answer': pred})
print(sub.shape); print(sub.head()); print(sub['answer'].value_counts(normalize=True))
sub.to_csv('submission.csv', index=False)

# arrays: exact dtype, shape and range
P = np.ascontiguousarray(np.clip(P, 0, 1).astype(np.float32))
assert P.shape == (360, 24, 29) and np.isfinite(P).all()
np.save('predictions.npy', P, allow_pickle=False)
```
Ids are the paths or file names exactly as in the test file, extension included.

-----

## R. Where to find this code during the contest

Blocks A to E use numpy and scipy only: `np.fft.rfft?`, `np.fft.rfftfreq?`, `scipy.signal.butter?`, `scipy.fft.dct?` print docstrings with runnable examples. Whether librosa, torchaudio and soundfile are on the package list needs checking in the appendix; the hand versions in block B replace them.

| block | what to search | package / page | how useful |
|---|---|---|---|
| 0 load | `wavfile.read`, `resample_poly` | scipy.io, scipy.signal | the dtype table for wav files is in the `wavfile.read` docstring |
| A spectrum, filters | `rfft`, `rfftfreq`, `irfft`, `butter`, `sosfiltfilt` | numpy.fft, scipy.signal | `butter` page has a full low pass example |
| B spectrogram | `hanning`, `scipy.signal.stft`, `dct` | numpy, scipy | `scipy.signal.stft` is the ready made version of `stft_mag` |
| B libraries | `melspectrogram`, `mfcc`, `power_to_db` | librosa feature pages; torchaudio "transforms" | parameter names only |
| H clustering | `LinearDiscriminantAnalysis`, `AgglomerativeClustering`, `adjusted_rand_score` | sklearn API + User Guide "Clustering" | good; metric and linkage options listed |
| I matching | `linear_sum_assignment` | scipy.optimize | the docstring example is the whole usage |
| K, L, N networks | `LayerNorm`, `GRU`, `BCEWithLogitsLoss`, `kl_div`, `normalize` | torch.nn, torch.nn.functional | `GRU` page gives output shapes; `BCEWithLogitsLoss` explains `pos_weight` |
| F, G pretrained | model card of the approved checkpoint | transformers docs | check the expected sample rate and the output field names there |
