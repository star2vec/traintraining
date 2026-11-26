## functia de validation

```python
def evaluate(model, X, y, kf, metric):
  scores = []
  for fit_rows, val_rows in kf.split(X, y):
    X_fit = X.iloc[fit_rows]
    y_fit = y.iloc[fit_rows]
    X_val = X.iloc[val_rows]
    y_val = y.iloc[val_rows]

    model.fit(X_fit, y_fit)
    preds = model.predict(X_val)
    scores.append(metric(preds, y_val))

  print('folds: ', [round(s, 3) for s in scares])
  print('mean: ', round(np.mean(scores), 4), 'spread: ', round(np.std(scores), 4))
  return scores    
```
