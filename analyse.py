import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

df = pd.read_csv('glass.csv')
X_scaled = preprocessing.scale(df.as_matrix()[:,:9])

pca = PCA(n_components=2)
pca.fit(X_scaled)

print(pca.components_)
print(pca.explained_variance_ratio_)  
print(pca.singular_values_)

#projection
for i in range(6):
    print("handling %s"%i)
    if len(df[df["Type"] == i+1]) == 0:
        continue
    X_scaled = preprocessing.scale(df[df["Type"] == i+1].as_matrix()[:,:9])
    dt = X_scaled
    d1 = pca.components_[0]
    d2 = pca.components_[1]
    dd =  np.stack((d1,d2)).transpose()
    pp = dt * np.matrix(dd)
    plt.scatter(np.asarray(pp[:,0]), np.asarray(pp[:,1]))
    
plt.show()
