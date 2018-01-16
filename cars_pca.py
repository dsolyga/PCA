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

pca = PCA(n_components=2)

# données brutes
pca.fit(pe)
print(pca.explained_variance_ )

# données normalisée
pca.fit(pe_scaled)
print(pca.explained_variance_ )
