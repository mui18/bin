from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df=pd.read_csv('cluster_data.csv')
centroids = np.array([[0.1,0.6],[0.3,0.2]])
kmeans = KMeans(n_clusters=2,init=centroids).fit(df)



print("Labels\n",kmeans.labels_)
print("Updated Centroids\n",kmeans.cluster_centers_)
plt.scatter(df['x'], df['y'], alpha=0.9)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red')
plt.show()


"""
Labels
 [0 0 0 0 1 0 1 1]
Updated Centroids
 [[0.148      0.712     ]
 [0.24666667 0.2       ]]

"""