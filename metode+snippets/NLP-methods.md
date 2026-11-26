# NLP-methods.md — text problems: types, approaches, traps

Model settings (TF-IDF parameters, C, Word2Vec, fine tune lr/epochs) are in setari-model.md section 3. This file is about recognizing the problem type and picking the approach.
Sources: the notebooks in this repo, plus task lists from IOAI, NEOAI, Polish, Hungarian, Georgian and Kazakh selections.

-----

## 0. First five minutes on any text problem

1. What goes in: short texts, long documents, pairs of texts, a query and a set of documents, tokens, embeddings already computed, a model already given?
2. What comes out: one label, several labels, a number, a ranking, a label per word, generated text, a cluster id?
3. Metric: accuracy, f1 macro / weighted / binary, top-k, cosine, edit distance? The validation function must compute exactly that.
4. Look at 10 texts per class. Word lengths with `describe()`. Duplicates. Weird tokens (`$NE$`, scrambled words, codes instead of words, html, urls). Share of test words seen in train.
5. Is there unlabeled text? Is the test text available? Both can be used for vocabulary, Word2Vec, TF-IDF fitting, clustering.

-----

## 1. Problem types

### 1a. Classify by content (topic, category, sentiment, intent)
Seen: mickey_donald subtask 2 (news category), algorel (problem category), NOAI news classification, NEOAI intent detection.
- First: TF-IDF words (1,2) + LogisticRegression.
- Second: sentence embeddings + LogisticRegression; average the two sets of probabilities.
- Few labels (under ~500): embeddings + LR beats TF-IDF. Many labels and GPU: fine tune a small transformer, 2..3 epochs.
- Trap: `stop_words` only for English; for Romanian or mixed text leave them in.

### 1b. Classify by style (author, dialect, human vs AI, native vs non-native, age)
Seen: mickey_donald subtask 1 (dialect), om_ai (human vs AI), ONIA 2025 national.
- First: TF-IDF `char_wb` (2,5) + LogisticRegression. Letter chunks catch spelling, suffixes, punctuation habits that words miss.
- Second: words + chars stacked (`scipy.sparse.hstack`) or probabilities averaged.
- Extra features that often matter for style: text length, average word length, punctuation counts, share of capitals, number of special tokens. Add them as columns next to TF-IDF or feed them to HGB.
- Observation kept from before: when style matters more than meaning, char_wb beats word n-grams; do not reach for larger word n-grams first.
- Trap: `lowercase=False` can help here (capitals are style); it hurts for topic.

### 1c. Words are not real words (scrambled, ciphered, codes, unknown language)
Seen: fox_onia (unlabeled text + test words unseen in train), IOAI 2024 (ciphered language), Polish OAI (cipher breaking).
- Pretrained models are useless: the tokenizer has never seen these tokens.
- First: Word2Vec trained on train + test + unlabeled, mean vector per text, LogisticRegression. `sg=1`, `window=3`, many epochs.
- Second: TF-IDF on the fake words (still works if the same codes repeat across classes), or char n-grams if the cipher keeps letter structure.
- Check first: share of test words that appear in train and in unlabeled. If most test words only appear in unlabeled, Word2Vec on everything is the whole point of the problem.
- Counting subtasks (vocabulary size, most frequent token) are usually there: a dict loop over all texts.

### 1d. Group texts without labels (clustering, topic discovery, find the odd ones)
Seen: om_ai subtask 2, algorel, false-friends (find the group that is not news), Kazakhstan player clustering.
- First: TF-IDF -> TruncatedSVD(100) -> Normalizer -> KMeans. Read 4 texts per cluster, name clusters, map to categories by hand.
- Second: sentence embeddings -> KMeans (no SVD, already dense and normalized).
- Zero-shot when categories are named: one description sentence per category, encode, cosine argmax. Combine with keyword counts for the unsure ones.
- Observations kept from before: TruncatedSVD is the PCA for sparse TF-IDF; Normalizer goes after TruncatedSVD; Normalizer is for KMeans / KNN, never before LogisticRegression or LinearSVC (they do their own scaling).
- K: silhouette over 2..12, then look. One or two extra clusters above the category count catch the mixed ones.
- "Which cluster is noise / not news": the one whose texts read differently, usually also the one with the lowest silhouette or the smallest / largest size. Verify by reading.

### 1e. Match, search, deduplicate (pairs of texts)
Seen: polyglot (match embeddings across two spaces), Georgia selection (duplicate detection with small BERT), Polish OAI (query to document alignment), corectare's metric.
- First: sentence embeddings normalized, `sims = Q @ D.T`, `argmax` per row for top-1, `argsort(-sims)[:, :k]` for top-k.
- Pair classification (duplicate or not): features `[cosine, |a - b|, a * b]` fed to LogisticRegression, or TF-IDF cosine + edit distance as two columns for HGB.
- Mutual best match (a's best is b and b's best is a) is much more reliable than one-sided argmax; use it to pick anchors, then extend (polyglot bootstrap).
- Two embedding spaces that do not share coordinates: relative representation = similarities to a shared set of anchor points, then compare those.
- Trap: TF-IDF cosine needs the same vectorizer fit on both sides; fit on documents, transform queries.

### 1f. Labels per token (named entities, part of speech, slot filling, find the masked word position)
Seen at other olympiads: NEOAI slot filling, Kazakhstan masked word position. Not yet in this repo.
- Simple baseline: one row per token with features (the token, previous token, next token, is capitalized, is digit, suffix 2..3 letters, position in sentence) -> LogisticRegression or HGB on one-hot/count encoded features. Works surprisingly well.
- Dictionary baseline: most frequent label per token in train; unknown tokens get the most frequent label overall.
- Transformer version: `AutoModelForTokenClassification`; the trap is subword alignment (a word becomes several tokens, label only the first, `-100` for the rest).
- Masked position / fill in the blank: a masked language model scores each candidate position; pick the position where the predicted word matches best. Or a simpler check: average probability of the actual next word at each gap.

### 1g. Regression on text (score, readability, reading time, age)
Seen: eye-tracking (predict reading time per word), Kazakhstan code difficulty.
- Word level targets: hand features (length, frequency in the corpus, position in sentence, page) + a target mean per word computed on the fit folds only -> CatBoost / HGB with GroupKFold by document.
- Text level targets: TF-IDF -> `Ridge(alpha=1..10)`, or embeddings -> Ridge. Ridge is the LogisticRegression of regression on text.
- Trap: a target mean per word computed on all of train leaks; compute it inside each fold, and for a word's own row exclude its own document (eye-tracking `addmean`).

### 1h. Generate or correct text (grammar correction, translation, summarization, fill a template)
Seen: corectare (flan-t5 fine tune). Polish OAI translation contest.
- First: the pretrained seq2seq model as is, with a task prefix. Measure the metric before training.
- Second: fine tune 3..4 epochs, keep the best epoch, then a safety rule: if the model changed too much of the text, return the original (threshold on edit distance chosen on val).
- Baseline that often scores high on correction tasks: return the input unchanged. Always compute it; everything must beat it.
- Trap: `max_new_tokens` too small cuts answers; too large wastes time. Set it from the length distribution of the targets.

### 1i. Given a model, do something with it (not train it)
Seen: IOAI 2025 Concepts (hint selection for an LLM guesser), NEOAI Broken BERT (repair embeddings), Polish OAI hallucination detection, Philippines "stripped transformer" (read answers from hidden states with a probe).
- Use the model as a feature extractor: hidden states of a layer -> `(N, d)` -> LogisticRegression ("probe"). Try the middle and the last layers; the middle is often better for meaning.
- Score candidates with the model: for a causal model, the sum of log probabilities of a candidate continuation; pick the highest.
- Embedding arithmetic: class means, differences of means (gender-bias shift), nearest centroid, projecting out a direction.
- Trap: forgetting `model.eval()` and `torch.no_grad()`; forgetting the attention mask when averaging over tokens (padding pulls the mean).

### 1j. Audio (in the syllabus, not excluded with time series)
Not yet in this repo.
- Given a pretrained audio encoder (Whisper, HuBERT, Wav2Vec2): mean of `last_hidden_state` over time -> `(N, d)` -> LogisticRegression. Same recipe as text embeddings.
- Transcribe with Whisper, then treat the transcript as a text problem.
- Load with `torchaudio.load`, mono by averaging channels, resample to 16 kHz. A spectrogram is an image; a small CNN can classify it.
- Trap: same speaker in train and val inflates scores; group folds by speaker.

-----

## 2. Shared building blocks

### Tokenizer + encoder by hand (when SentenceTransformer is not available)
```python
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModel.from_pretrained(NAME).to(DEVICE).eval()

vectors = []
for start in range(0, len(texts), 64):
  batch = texts[start:start + 64]
  enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)
  with torch.no_grad():
    out = model(**enc).last_hidden_state          # (B, L, d)
  mask = enc['attention_mask'].unsqueeze(-1)       # (B, L, 1): 1 on real tokens
  mean = (out * mask).sum(1) / mask.sum(1)         # mean over real tokens only -> (B, d)
  vectors.append(mean.cpu().numpy())
E = np.concatenate(vectors)
E = E / np.linalg.norm(E, axis=1, keepdims=True)   # length 1 -> @ is cosine
```
[CLS] instead of mean: `out[:, 0]`. BERT style models: either works; mean is safer.

### max_length
Not 512 by default. Tokenize 2000 texts, take the 95th percentile of token counts, round up to 128 / 256 / 512. Shorter = faster = more epochs in the same time.

### Long texts
Chunk into ~300 word pieces with 50 word overlap, encode each, average per document (code in README). For TF-IDF no chunking is needed.

### Cleaning
TF-IDF's default tokenizer already drops punctuation and single characters, so cleaning changes little there. Clean when: tokens are custom (keep `$NE$` whole), html/urls present, or char n-grams are used (then decide consciously whether punctuation stays, since it is a style signal).

### Extra hand features next to any text model
Length in words, length in characters, average word length, number of sentences, count of `!` `?` digits capitals, share of unique words, count of special tokens. Ten columns, computed with a loop, fed to HGB alone or stacked with TF-IDF. Often worth 1..3 points on style tasks and on regression.

### Stacking sparse + dense
```python
from scipy.sparse import hstack, csr_matrix
X_all = hstack([X_tfidf_words, X_tfidf_chars, csr_matrix(hand_features)]).tocsr()
```
Scale the hand features first (`StandardScaler`), TF-IDF columns are already in [0, 1].

-----

## 3. Validation on text

- StratifiedKFold(5) for classification; KFold for regression; GroupKFold when several texts come from the same author / document / speaker.
- The vectorizer is fit inside each fold on the fit rows only (Pipeline does this automatically).
- Exception that is allowed: fitting an unsupervised step (TF-IDF vocabulary, Word2Vec, KMeans) on train + test + unlabeled text. No labels are used, so it is not leakage, and it helps when test words are unseen in train.
- Duplicates between train and test: predict them directly with the train label.
- Check the predicted class distribution on test against train. A big difference usually means a bug, sometimes a real shift.

-----

## 4. Baselines that take under five minutes

| situation | baseline |
|---|---|
| any classification | most frequent class |
| content classification | TF-IDF (1,1) + LogisticRegression default |
| style classification | TF-IDF char_wb (2,5) + LogisticRegression |
| matching | TF-IDF cosine argmax |
| clustering with named categories | keyword counts per category, argmax |
| correction / rewriting | return the input |
| token labels | most frequent label per token in train |
| text regression | mean of the target; then Ridge on TF-IDF |
| given embeddings | nearest class centroid |

-----

## 5. Traps

- `str(t)` on a missing text gives `'nan'`; `fillna('')` first.
- Keep the same fitting rule in validation and in the final model: if TF-IDF is fit on train + test for the final run, fit it the same way inside the folds.
- `f1_score` default is binary with positive label 1; with text labels or more than two classes pass `average='macro'` or `'weighted'` to match the statement.
- LogisticRegression on TF-IDF needs a larger C (10) than on dense features (1). Small C underfits sparse data.
- Word2Vec vectors for texts with no known words come out as zeros; fine for LR, but check how many such texts exist.
- KMeans cluster ids change between runs: fix `random_state` and map ids to categories only after the final fit.
- Sentence embeddings from different models are not comparable; one model for queries and documents.
- `TruncatedSVD` output is not normalized; `Normalizer()` after it before KMeans.
- Transformer tokenizer `truncation=True` without `max_length` cuts at the model limit silently; long documents lose their end.
- Padding without attention mask in a mean pooling: short texts get pulled toward the padding vector.
- Fine tuning on under ~1000 texts with lr 2e-5 for 3 epochs often does not beat TF-IDF + LR; check the baseline first and spend the time on features instead.
- Romanian diacritics: texts may mix "ș" and "ş" (two different characters). Normalize with a replace if the vocabulary looks doubled.
