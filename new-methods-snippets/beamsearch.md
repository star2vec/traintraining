# BEAMSEARCH

## Idee de baza:
beamsearch: pastram top-k answers

## Probleme rezolvate:
nitro judge ai / euroai day 2 / interleaved lines:
https://github.com/star2vec/traintraining/blob/main/probleme/nitro/euroai-day2/interleaved-lines.ipynb

## Cod:
```python
def beammask(v, K=5):
  beams = [0] # prima chestiuta
  for e in v:
    cands = []
    for b in beams:
      cands.append(f(e, b))
    cands.sort(key=lambda x: -x)

    beams = []
    for ec in cands[:k]:
      beams.append(ec)

    best = beams[0]
    return best
```

## Exemplu:
```python
def beammask(c, K=5):
    # each beam: [total score, A, B, mask, predictions after A, predictions after B]
    beams = [[0.0, [c[0]], [], [0], nextlogprob([c[0]]), nextlogprob([])]]
    for tok in c[1:]:
        cands = []
        for score, A, B, mask, lpA, lpB in beams:
            cands.append((score + lpA[tok].item(), 0, A, B, mask, lpA, lpB))
            cands.append((score + lpB[tok].item(), 1, A, B, mask, lpA, lpB))
        cands.sort(key=lambda x: -x[0])
        beams = []
        for s, side, A, B, mask, lpA, lpB in cands[:K]:
            if side == 0:
                A2 = A + [tok]
                beams.append([s, A2, B, mask + [0], nextlogprob(A2), lpB])
            else:
                B2 = B + [tok]
                beams.append([s, A, B2, mask + [1], lpA, nextlogprob(B2)])
    best = beams[0]
    for b in beams:
        if b[0] > best[0]:
            best = b
    return best[3]
```
