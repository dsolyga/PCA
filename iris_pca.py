import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=4)
X_r = pca.fit(X).transform(X)

colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.title('PCA of IRIS dataset')
props = ["sepal length", "sepal width", "petal length", "petal width"]
for i in range(4):
    x = pca.components_[0][i]
    y = pca.components_[1][i]
    plt.arrow(0,0, x, y, head_width=0.05, head_length=0.1, fc='k', ec='k')
    plt.text(x, y, props[i])

plt.show()
print(pca.components_)
print(pca.singular_values_)
