
```
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

P = PCA(n_components=2).fit_transform(train_image_embeddings)
plt.scatter(P[:, 0], P[:, 1], c=gender_labels, s=2)
plt.show()
```
!! ^^^ pt tf-idf cu multe, aia se afce cu truncatedsvd


# axis = 0 inseamna ca facem media pt fiecare din alea 512
mean0 = X[y == 0].mean(axis=0)    
mean1 = X[y == 1].mean(axis=0)     
mid = (mean0 + mean1) / 2
