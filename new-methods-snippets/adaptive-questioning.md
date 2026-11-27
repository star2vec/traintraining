# Decision trees

## Idee de baza:

Fiecare intrebare tb sa dea split la choices in 2
  


## Probleme rezolvate:
ML-compete:
* https://github.com/star2vec/traintraining/blob/main/probleme/ml-compete/wilkins-determinist.ipynb
* https://github.com/star2vec/traintraining/blob/main/probleme/ml-compete/wilkins-llm_nu-a-fost-trimis.ipynb

## Cod:

```python
cnt = 0
tree = {}

def newid():
  global cnt
  i = cnt
  cnt += 1
  return i

def guesschain(nod, grup):
  now = nod
  for k in range(len(grup)):
    if k == len(grup)-1:
      nxt = None
    else:
      nxt = newid()
    tree[now] = {'type': 'guess', 'candidat': cands[grup[k]], 'wrong': nxt}
    now = nxt

def build(grup):
  nod = newid()

  if (len(grup) <= 3): # aici putem sa alegem ce vrem noi, usually 3 e the most efficient choice
    guesschain(nod, grup)
    return nod

  bestq = -1
  bestscore = 0
  for q in range(len(questions)):
    nryes = 0
    for e in grup:
      if table[e][q]=='yes':
        nryes += 1
    nrno = len(grup)-nryes
    sc = min(nryes, nrno)
    if sc > bestscore:
      bestscore = sc
      bestq = q

  if bestscore == 0:
    guesschain(nod, grup)
    return nod

  yes = [a for a in group if table[a][bestq] == 1]
  no = [a for a in group if table[a][bestq] == 0]
  yesnode = build(yes)
  nonode = build(no)
  tree[node] = {'type': 'ask', 'question': bestq, 'yes': yesnode, 'no': nonode}
  return node
```


Acum verificare:
```python
def play(target):
  nod = 0
  steps = 0
  while nod is not None:
    steps += 1
    n = tree[nod]
    if n['type']=='ask':
      if table[target][n['question']]==1:
        nod = n['yes']
      else:
        nod = n['no']
    else:
      if n['candidate']==cands[target]:
        return steps
      nod = n['wrong']
  return -1

total = 0
for t in range(len(cands)):
  total += play(t)
print('average steps: ', total / len(cands))

```

## Exemplu:
```python
cnt = 0
nodes = {}                    

def newid():
    global cnt
    i = cnt
    cnt = cnt + 1
    return i

def guess_chain(node, group):
    now = node
    for k in range(len(group)):
        if k == len(group) - 1:
            nxt = None                          
        else:
            nxt = newid()
        nodes[now] = {'type': 'guess', 'animal': cands[group[k]], 'wrong': nxt}
        now = nxt

def build(group):
    node = newid()

    if len(group) <= 3:
        guess_chain(node, group)
        return node

    bestq = 0
    bestscore = -1
    for q in range(len(questions)):
        nryes = 0
        for a in group:
            if table[a][q] == 1:
                nryes += 1
        nrno = len(group) - nryes
        score = min(nryes, nrno)
        if score > bestscore:
            bestscore = score
            bestq = q

    if bestscore == 0:
        guess_chain(node, group)
        return node

    yes = [a for a in group if table[a][bestq] == 1]
    no = [a for a in group if table[a][bestq] == 0]
    yesnode = build(yes)
    nonode = build(no)
    nodes[node] = {'type': 'ask', 'question': bestq, 'yes': yesnode, 'no': nonode}
    return node

table = np.load('table.npy')
build(list(range(len(cands))))
print(len(nodes))
```
