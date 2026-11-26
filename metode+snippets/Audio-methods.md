# Audio-methods.md — audio problems: types, approaches, traps

Model settings (small CNN, training loop, loss by target type) are in setari-model.md sections 4 and 7; a spectrogram goes through the same CNN as an image. This file is about recognizing the problem type and picking the approach. Code blocks go in Audio-methods-code.md.
Sources: the eight audio problems in the olimpiada-ai.ro archive, written as #id. None is solved in this repo yet, so the "Practice" lines say where each approach gets tested, not where it already worked.

-----

## 0. First five minutes on any audio problem

1. What goes in: raw waveforms (wav / mp3 files, arrays in an npz), or embeddings from a frozen encoder (one per clip, or one per frame)? With embeddings there is no signal processing left: it is a table or a sequence problem (1h, 1i, 1j).
2. What comes out: one class per clip, labels per frame, a group id, a matching, text, the weights of a head?
3. Metric: accuracy, ARI, average precision, F1 at a fixed threshold, edit distance? The validation function must compute exactly that.
4. Per file: sample rate, length, dtype (uint8, int16, float32), channels, min and max. `value_counts()` of sample rate and length over all files and per class. A class that always has its own length or sample rate is a leak worth using.
5. Listen to 3 clips per class. Plot the waveform and the spectrogram of the same clips.
6. Mean spectrum per class and train vs test: where the energy is, where classes differ, whether a band is only noise.
7. Groups: speaker, recording, source video, microphone / domain. Validation must split by group.

A title with "signal" is not always audio: #360 Seismic Signals is a grid reconstruction.

-----

## 1. Problem types

### 1a. Classify short clips from the waveform (spoken word, sound class)
Practice: #354 / #358 Șase sau șapte ("six" vs "seven", 1 s at 1024 Hz).
- First: log-mel spectrogram -> mean and std over time of every band + MFCC mean and std -> StandardScaler -> LogisticRegression.
- Clips of equal length: also flatten the whole log-mel `(n_mels * T)` -> LR or KNN. Mean and std throw away the order in time; two words can differ only in when things happen.
- Second: log-mel as a 1 channel image `(N, 1, n_mels, T)` -> the small CNN from CV-methods-code D, with time and frequency masking.
- Third (16 kHz speech or everyday sounds, pretrained models allowed): frozen encoder features + LR, see 1g and section 2.
- Window sizes follow the sample rate: a window of about 25 ms, a hop of about 10 ms. 16 kHz -> `n_fft=512, hop=160`. 1024 Hz -> `n_fft=128, hop=32`, with 20 mel bands or plain FFT bins.
- Trap: library defaults (`n_fft=2048`) are made for 22 kHz music; a clip of 1024 samples becomes a single frame.
- Trap: at an unusual sample rate pretrained encoders do not help. They learned 16 kHz speech; a 1024 Hz clip has nothing above 512 Hz.

### 1b. The signal is under noise
Practice: #354 (noise above ~350 Hz with over 95% of the energy).
- Look first: mean spectrum over all clips. Noise in a band shows as a wall. The share of energy above a cutoff is one line of code.
- Fix: low pass. Zero the FFT bins above the cutoff and invert, or a Butterworth filter. Then 1a.
- Features per band (log-mel rows) let the model ignore noisy bands on its own. The filter matters for everything computed on the waveform (energy, peak normalization) and for a CNN.
- The cutoff in the statement is approximate: scan it on validation (250, 300, 350, 400).
- Trap: scaling each clip by its peak or energy before filtering. The noise sets the scale, so the useful part ends up with a different size in every clip. Filter first, normalize after.

### 1c. A message is hidden in the audio
Practice: #362 Mesaje Ascunse (a letter or a word hidden in 2 s music clips, 8 bit, 22050 Hz).
- Inspection first, model later. Where to look, in order: the spectrogram of several clips (tones, lines, patterns at the top of the band or in fixed time slots); the raw integers (lowest bit of each sample, histogram of values, values never used); the file itself (length, extra bytes).
- With labels: mean spectrogram per label minus the overall mean shows where the label lives. The fastest way to find an encoding nobody describes.
- Once found: decode by rule if the rule is exact, otherwise a classifier on that region only.
- If a word turns out to be letters one after another: cut the clip into equal time slots, run the letter model on each, then replace the decoded string with the nearest word in the given list (edit distance). Half of that score is exact match, so this last step is worth a lot.
- Trap: a message kept in the low bits is destroyed by converting to float or resampling at load. Read the raw integers (`wavfile.read`), not `librosa.load`.
- Trap: 8 bit wav is unsigned; silence is 128, not 0.

### 1d. Who is speaking (speakers not seen in train)
Practice: #352 Vocile lui Petrică (300 clips into 10 speakers, 20 other labeled speakers for training, ARI).
- The train speakers cannot be memorized. The model has to learn which features separate voices in general.
- First: MFCC mean and std per clip -> StandardScaler -> KMeans(10). A voice is its long term average; what is said averages out over 3 s. The ARI of this on held out train speakers is the number to beat.
- Second: LDA fit on the train speakers, which keeps the directions that separate speakers and drops the ones that vary inside a speaker. Transform the test clips, normalize rows, AgglomerativeClustering with cosine distance (or KMeans).
- Third: a network trained to classify the train speakers; the layer before the output is the embedding, cluster those. Or a pretrained speech encoder mean pooled, with the same LDA on top.
- Validation: hold out whole speakers (5 of 20), cluster them with K=5, ARI. Never a random split of clips.
- Trap: loudness and pitch features change with what is said and with the microphone. MFCC statistics and the mean log spectrum are steadier.

### 1e. What is said, whatever the voice (same sentence across voices)
Practice: #274 Roboții vorbăreți task 1 (300 files = 100 sentences x 3 synthesizers, group into triplets).
- The opposite of 1d: keep the content, drop the voice.
- First: transcribe with Whisper, match on text (equal strings, then the smallest edit distance).
- Without a speech model: frame features (MFCC) compared with DTW, which lines up two sequences spoken at different speeds. Duration is a cheap filter: the same sentence has a similar length.
- The structure is a constraint. Each triplet has exactly one file per synthesizer: tell the synthesizers apart first (1f), then solve two one to one matchings (A to B, A to C) with the Hungarian algorithm instead of free clustering.
- Trap: argmax per row hands the same partner to two files. One to one assignment fixes those for free.

### 1f. Which system produced it (synthesizer, real vs generated, device)
Practice: #274 task 2 (three synthesizers, many voices each, 900 labeled clips).
- Opposite again: drop content and voice, keep the fingerprint of the generator.
- First, file facts per class: sample rate, bit depth, length, peak level, share of exact zeros. When these differ by class they are features, sometimes the whole answer.
- Second: long term average spectrum (mean log spectrum over time, averaged into ~64 bands) plus the file facts -> LR or HGB. Generators differ at the top of the band (where the highs stop) and in the noise floor during silence.
- Third: log-mel CNN on random 2 s crops, or encoder features + LR.
- Validation: group by voice when voices can be told apart (cluster MFCC means), since test may bring voices not in train.
- Trap: a model that learns the voices scores high on a random split and drops on test.

### 1g. Frozen pretrained model, only the head trains; new classes without forgetting old ones
Practice: #302 / #300 NightWatch (AST, 16 -> 29 classes, only LayerNorm + Linear may change).
- Run the encoder once per clip, save the embeddings. Everything after is training a tiny head on a table, seconds per run.
- New head = the old one with extra rows. Old rows start from the old weights. New rows start from class prototypes: the mean embedding of the class (after the LayerNorm), scaled to the norm of the old rows.
- Train with: class weights or balanced batches (new classes are few), old class examples in every batch, and a distillation term that keeps the old logits close to the old model's.
- Two numbers every run: accuracy on old classes and on new classes. A single total hides forgetting.
- New rows with a larger norm or bias than the old ones steal predictions. Compare the average logit of old and new classes on validation.
- LayerNorm stays as it was at first; unfreeze it last, with a small lr.
- Before submitting: the exported weights must reproduce the submitted predictions exactly, with the shapes from the statement.
- Trap: mp3 and wav mixed, clips of different lengths. The model's own feature extractor pads and normalizes; a hand written loader has to do the same.

### 1h. Events per frame from embeddings (several can be on at once)
Practice: #310 Peisaje sonore de noapte (`(scenes, 24 frames, 48)` -> 29 yes/no per frame).
- First: every frame is a row, `(N*T, d)` -> one LogisticRegression per class.
- Second: context. Concatenate each frame with its 2 neighbors on each side, same LR. Events last several frames, so neighbors vote.
- Third: bidirectional GRU over the frames, `BCEWithLogitsLoss` with `pos_weight` per class for the rare ones.
- After: median over 3 frames on the probabilities, which removes single frame blips.
- Metric with a fixed 0.5 threshold: find the best threshold per class on validation, then shift the probabilities so that threshold lands on 0.5. Average precision is unchanged (the order is kept), F1 goes up.
- Trap: a random split of frames puts one scene on both sides. Split by scene, and by domain to imitate the hidden part of the test.

### 1i. Unknown events, start and end, changed microphones
Practice: #316 Ecouri în schimbare (12 known classes + unknown, boundaries, shifted domains).
- Unknown: a direct "unknown" classifier only learns the unknown families present in train. Add a second score: "something is happening" (event vs background, trained on all events) times "it looks like no known class" (1 - highest known probability). Average the two.
- Boundaries: derive them from the frame predictions (first and last frame of each run of one class). A separate boundary model only if this loses on validation. With a tolerance of one frame, smoothing the frame classes first matters more than a precise model.
- Domain shift: validate by leaving one domain out, each in turn. Then try standardizing each scene with its own mean and std instead of train statistics, and fitting the scaler on train + test embeddings (no labels involved).
- Trap: tuning on a split that shares domains with train. The hidden part of the leaderboard is the shifted one.

### 1j. Match audio to another modality through embeddings
Practice: #268 Video Shazam (WavLM audio `(N, 768)`, V-JEPA 2 video `(N, 1024)`, the right video among 20).
- The two spaces share nothing; the map has to be learned from the pairs.
- First: Ridge from audio to video embeddings, cosine against the 20 candidates, argmax.
- Second: two small MLPs into one shared space with a contrastive loss (inside a batch every audio must score highest with its own video), batch 256 or more.
- Inside a group the matching is one to one: Hungarian on the 20x20 similarity matrix instead of argmax.
- Validation by source video when it is known. Clips of one video are near duplicates and the private test has unseen videos, so a random split of rows overstates the score.

### 1k. Speech to text, then a text problem
- Whisper (`small` on GPU, `tiny` or `base` on CPU) -> transcripts -> NLP-methods.md.
- Trap: Whisper invents text on silence and noise. Skip clips whose energy is near zero.

-----

## 2. Shared building blocks

### Loading and shapes
A waveform is `(samples,)` float in [-1, 1]. `wavfile.read` gives integers: int16 / 32768, uint8 -> (x - 128) / 128. Stereo is `(samples, 2)`: average the columns. mp3: `librosa.load(path, sr=16000)`. Clips of different lengths: pad with zeros or crop to one length before stacking. Torch wants `(N, samples)` for encoders and `(N, 1, n_mels, T)` for a CNN.

### Sample rate and frequency
Sample rate `sr` = samples per second. The highest frequency in the file is `sr / 2`. An FFT of `n` samples has `n/2 + 1` bins, and bin `k` is the frequency `k * sr / n`. A longer window separates frequencies better and blurs time; a shorter one does the reverse.

### Spectrogram, log-mel, MFCC
Spectrogram: cut the wave into overlapping windows, FFT each, keep the magnitude: `(n_fft/2 + 1, frames)`. Take the log; raw magnitudes span too many orders of size for any model. Mel: average neighboring bins into ~40 bands that are narrow at low frequencies and wide at high ones. MFCC: a DCT of each log-mel column, first 13 numbers; a compact description of the spectrum's shape. Log-mel for CNNs, MFCC statistics for LR / KMeans.

### Clip features or frame features
One vector per clip (mean and std over time) for "what kind of sound / whose voice". The full `(bands, frames)` matrix when the order in time matters (which word, where an event starts).

### Augmentation rules
Keep the label true. Safe: shift in time, small volume change, added noise, masking a few time frames or frequency bands. Unsafe: reversing time, and pitch or speed change when the label is the speaker or the system. Validation and test get none.

### Pretrained encoders
| model | trained on | input | output |
|---|---|---|---|
| Wav2Vec2 / HuBERT / WavLM | speech | 16 kHz waveform | ~50 frames per second, 768 each; mean over time per clip |
| Whisper | speech -> text | 16 kHz, windows of 30 s | text |
| AST | everyday sounds (AudioSet) | 16 kHz, up to ~10 s | one 768 vector per clip |

Speech models for words, speakers, language. AST for dogs, rain, engines. Earlier layers of speech models often describe the speaker better, later layers the words: try a few on validation.

### Two cheap wins
MFCC mean and std + LogisticRegression (any clip classification, runs on CPU). Embeddings saved to disk once, then every experiment is on a table.

-----

## 3. Validation on audio

- Group by speaker, recording, source video or microphone. Clips cut from one recording are near duplicates.
- Clustering or matching on unseen identities: hold out whole identities and compute the real metric (ARI, matching accuracy) on them.
- Domain shift announced in the statement: leave one domain out.
- The scaler and every statistic (mean spectrogram, class prototypes) come from the fit rows only. Exception: unsupervised steps on train + test without labels.
- Same loading and same feature function for train, validation and test, called in one place.
- Exact duplicates between train and test (same bytes): predict them with the train label.

-----

## 4. Baselines that take under five minutes

| situation | baseline |
|---|---|
| any classification | most frequent class |
| clip classification | MFCC mean and std + LogisticRegression |
| equal length clips | flattened log-mel + KNN(3) |
| speaker clustering | MFCC statistics + KMeans |
| same content across voices | match on duration; Whisper text when allowed |
| which system made it | file facts (sample rate, length, peak) + long term average spectrum + LR |
| frozen model, new classes | prototype rows, no training |
| frame events from embeddings | one LR per class on single frames |
| boundaries | first and last frame of each predicted run |
| audio to video embeddings | Ridge + cosine argmax |
| hidden message | mean spectrogram per label minus the overall mean |

-----

## 5. Traps

- Mixed sample rates in one folder: the same word gives different spectrograms. Resample everything to one rate.
- int16 fed to a model without dividing by 32768; uint8 read as if silence were 0.
- Stereo read as `(samples, 2)` and passed on as if it were two samples wide.
- `n_fft` longer than the clip, or library defaults used at a low sample rate.
- Log of zero: `np.log(S + 1e-10)`.
- Spectrogram axes: rows are frequency, columns are time. `plt.imshow(..., origin='lower')` puts low frequencies at the bottom.
- Mean and std for normalizing spectrograms computed on train + validation together.
- Recent torchaudio needs the extra package torchcodec for `torchaudio.load`; `wavfile.read`, `soundfile.read` and `librosa.load` work without it.
- `librosa.load` resamples to 22050 Hz and converts to float unless `sr=None`.
- Encoder embeddings averaged over padded frames: short clips get pulled toward the padding. Encode clips one at a time or mask the padding.
- Random split when clips share a speaker or a recording: the score is inflated.
- Threshold left at 0.5 when the metric is F1 at 0.5 and the classes are rare.
- A DataLoader with `shuffle=True` on test, or `os.listdir` order taken as the id order.
- Probabilities written as float64 or outside [0, 1] when the statement asks for float32 in [0, 1]: the submission is rejected.
