import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from random import random
import numpy as np

nbSamples = 1000
X0 = [random() for x in range(nbSamples)]
X1 = [3.1416 * x for x in X0]


X = np.matrix((X0, X1)).transpose()
pca = PCA(n_components=2)
pca.fit(X)
print(pca.components_)
print(pca.singular_values_)


plt.scatter(X0, X1)
plt.xlabel("Largeur")
plt.ylabel("Perimetre")
plt.show()
