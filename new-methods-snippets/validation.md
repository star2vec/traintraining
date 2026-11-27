## functia generala: 

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

## kf
```python
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold

### aici sa pun astea pt fiecare ig?? desi deja apare acolo in documentation so theres not really much need?


   
```

### Choosing the validation split
```python
print(train_data[target].value_counts(normalize=True))  # class balance??                 
print(test_data['Customer_ID'].isin(train_data['Customer_ID']).mean())    # cate grupuri sunt shared intre test si train
```
!! rows in grupuri => GroupKFold / StratifiedGroupKFold
!! clasificare => StratifiedKFold

usually se fac 5 folds (maybe 3 daca training is slow)

IMPORTANT: 
1. A model trained on unbalanced data tends to ignore the rare class too. Two fixes: class_weight='balanced', which makes mistakes on the rare class count more
2. tuning the decision threshold, like the cutoff in the anomaly task."

### Groups 
(nu vor fi revealed de duplicates pt ca row-urile nu sunt fully identice, au doar componente asa, in rest pb sunt very similar, de ex ac CostumerID)
```python
print(train_data['Customer_ID'].value_counts().head())
print(test_data['Customer_ID'].isin(train_data['Customer_ID']).mean()) #share of test with known groups
```
!! daca a doua linie iese:
- close to 0 => test are grupuri noi most likely => folosim GroupKFold
- close to 1 => test are grupuri cunoscute => folosim stratified folds (si incercam sa dam consider la features din each other group's row)

