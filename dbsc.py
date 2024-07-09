import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Given data
X = np.array([[3, 7], [4, 6], [5, 5], [7, 2], [7, 3], [6, 2], [7, 2], [3, 4], [3, 3], [2, 6], [3, 4], [2, 4]])

# Applying DBSCAN clustering
clustering = DBSCAN(eps=1.9, min_samples=4).fit(X)

# Extracting the labels
labels = clustering.labels_

# Printing the labels
print("Cluster Labels:", labels)

///# Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', s=100, alpha=0.8)
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
///
