import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

iris = load_iris(as_frame=True)
X = iris.data
print("Dataset Shape:", X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 2
kmeans = KMeans(n_clusters = k, random_state = 42, n_init = 10)

clusters = kmeans.fit_predict(X_scaled)
X_clustered = X.copy()
X_clustered['Cluster'] = clusters
print(X_clustered.head())

plt.figure(figsize=(8,6))
plt.scatter(
    X_scaled[:,0],
    X_scaled[:,1],
    c= clusters,
    cmap='viridis',
    s=50
)

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    c='red',
    marker='X',
    s=200,
    label="centroids"
)
plt.title("K-Means Clustering (Iris)")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.legend()
plt.show()







