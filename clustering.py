import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from collections import Counter


def optimize_k(x, kmax):
    sse = []
    sil = []
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10).fit(x)
        labels = kmeans.labels_
        sil.append(silhouette_score(X, labels, metric='euclidean'))
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(x)
        curr_sse = 0

        for i in range(len(x)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += distance.euclidean(x.iloc[i], curr_center)

        sse.append(curr_sse)
    return sse, sil


def plot_pca_clusters(X, clusters, model, title, cluster_ids_to_plot=None):
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 7))

    unique_clusters = set(clusters)
    if cluster_ids_to_plot is not None:
        clusters_to_plot = set(cluster_ids_to_plot)
    else:
        clusters_to_plot = unique_clusters

    for cluster_id in clusters_to_plot:
        if cluster_id in unique_clusters:  # Check if cluster exists
            plt.scatter(df_pca[clusters == cluster_id, 0], df_pca[clusters == cluster_id, 1], label=f'Cluster {cluster_id + 1}')

    if model is not None and hasattr(model, 'cluster_centers_'):
        pca_centroids = pca.transform(model.cluster_centers_)
        plt.scatter(pca_centroids[:, 0], pca_centroids[:, 1], s=50, c='black', label='Centroids', marker='X')

    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



def plot_elbow_method(sse, kmax):
    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(2, kmax + 1, 1), sse)
    plt.xticks(np.arange(2, kmax + 1, 1))
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('WSS')
    plt.title('K selection in K-means++ - Elbow Method')
    plt.tight_layout()
    plt.show()


def plot_silhouette_method(sil, kmax):
    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(2, kmax + 1, 1), sil, 'bx-')
    plt.xticks(np.arange(2, kmax + 1, 1))
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('K selection in K-means++ - Silhouette Method')
    plt.tight_layout()
    plt.show()


df = pd.read_csv('output/preprocessed_standard.csv')
X = df.drop(columns=['installCount', 'box_cox_installs', 'installQcut', 'installRange'])

k_max = 10
# sse, sil = optimize_k(X, k)
# results = pd.DataFrame({'k': np.arange(2, k + 1, 1), 'sse': sse, 'sil': sil})
# results.to_csv('output/kmeans_optimization.csv', index=False)
results = pd.read_csv('output/kmeans_optimization.csv')
sse, sil = results['sse'].values, results['sil'].values

plot_elbow_method(sse, k_max)
plot_silhouette_method(sil, k_max)

n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
clusters = kmeans.fit_predict(X)
plot_pca_clusters(X, clusters, kmeans, 'Cluster Visualization with PCA - K-means++')

print("=========================================")
print("DBSCAN")
print("Output: DBSCAN Cluster Visualization by PCA")
print("=========================================")

# For DBSCAN (top 10 clusters)
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X)
cluster_counts = Counter(clusters)
top_clusters = [cluster[0] for cluster in cluster_counts.most_common(10) if cluster[0] != -1]
plot_pca_clusters(X, clusters, None, 'Top 10 Clusters with PCA - DBSCAN', top_clusters)

