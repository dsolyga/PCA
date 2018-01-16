import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

df = pd.read_csv('04cars.dat.txt')
cols = ['price','invoice price','dealer cost','engine','cylinders','horsepower', 'weight','wheel','length','width', 'cm per gallons', 'hm per gallons']

X_scaled = preprocessing.scale(df[cols].replace('*', float('nan')).dropna().as_matrix())
pe = df[df['price'] > 1000][df['engine'] < 10][['price', 'engine']]
pe_scaled = preprocessing.scale(pe)
plt.scatter(pe['price'], pe['engine'])
plt.xlabel('price')
plt.ylabel('engine')
plt.show()

plt.scatter(pe_scaled[:,0], pe_scaled[:,1])
plt.xlabel('price (norm)')
plt.ylabel('engine (norm)')
plt.show()


pca = PCA(n_components=12)
pca.fit(X_scaled)

print(pca.components_)
print(pca.explained_variance_ratio_)  
print(pca.singular_values_)

#projection
for typ in ['sport', 'utility', 'wagon', 'minivan']:
    print("handling %s"%typ)
    if len(df[df[typ] == 1]) == 0:
        continue
    X_scaled = preprocessing.scale(df[df[typ] == 1][cols].replace('*', float('nan')).dropna().as_matrix())
    dt = X_scaled
    d1 = pca.components_[0]
    d2 = pca.components_[1]
    dd = np.stack((d1,d2)).transpose()
    pp = dt * np.matrix(dd)
    plt.scatter(np.asarray(pp[:,0]), np.asarray(pp[:,1]), label=typ)
    
plt.show()
