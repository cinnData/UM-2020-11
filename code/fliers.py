## Marketing frequent fliers ##

# Importing packages #
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Importing data from CSV file #
url = 'https://raw.githubusercontent.com/cinnData/UM-2020-11/main/data/fliers.csv'
df = pd.read_csv(url)
df.info()
df.head()

# 4-cluster analysis (1) #
clus = KMeans(n_clusters=4, random_state=0)
X = df.drop(columns='id')
clus.fit(X)
labels = clus.labels_
np.unique(labels, return_counts=True)
centers = clus.cluster_centers_
centers = pd.DataFrame(centers)
centers.columns = df.columns[1:]
centers['size'] = np.unique(labels, return_counts=True)[1]
centers

# Normalization #
scaler = MinMaxScaler()
scaler.fit(X)
Z = scaler.transform(X)

# 4-cluster analysis (2) #
clus.fit(Z)
labels = clus.labels_
np.unique(labels, return_counts=True)
centers = clus.cluster_centers_
centers = pd.DataFrame(centers)
centers.columns = df.columns[1:]
centers['size'] = np.unique(labels, return_counts=True)[1]
centers

# 5-cluster analysis #
clus = KMeans(n_clusters=5, random_state=0)
clus.fit(Z)
labels = clus.labels_
np.unique(labels, return_counts=True)
centers = clus.cluster_centers_
centers = pd.DataFrame(centers)
centers.columns = df.columns[1:]
centers['size'] = np.unique(labels, return_counts=True)[1]
centers

# 3-cluster analysis #
clus = KMeans(n_clusters=3, random_state=0)
clus.fit(Z)
labels = clus.labels_
np.unique(labels, return_counts=True)
centers = clus.cluster_centers_
centers = pd.DataFrame(centers)
centers.columns = df.columns[1:]
centers['size'] = np.unique(labels, return_counts=True)[1]
centers

# Additional customer #
x = np.array([63000, 157, 1, 1, 1, 1200, 15, 764, 1, 1214, 0]).reshape(1, 11)
z = scaler.transform(x)
clus.predict(z)
