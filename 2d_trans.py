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
X_r = pca.fit(X).transform(X)
print(pca.components_)
print(pca.singular_values_)


plt.scatter(X_r[:,0], X_r[:,1])
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.show()
