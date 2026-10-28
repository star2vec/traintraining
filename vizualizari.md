
```
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

P = PCA(n_components=2).fit_transform(train_image_embeddings)
plt.scatter(P[:, 0], P[:, 1], c=gender_labels, s=2)
plt.show()
```
