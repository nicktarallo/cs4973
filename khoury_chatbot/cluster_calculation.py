import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Sample data
X = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0]
])

# Fit the KMeans model
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Get cluster assignments and centers
cluster_assignments = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Function to calculate similarity score for each cluster center
def calculate_cluster_similarity(cluster_assignments, cluster_centers, X):
    similarity_scores = []
    for cluster_idx in range(cluster_centers.shape[0]):
        # Get points belonging to the current cluster
        cluster_points = X[cluster_assignments == cluster_idx]
        # Calculate distances to the cluster center
        distances = np.linalg.norm(cluster_points - cluster_centers[cluster_idx], axis=1)
        # Compute average distance
        average_distance = np.mean(distances)
        # Calculate similarity score (inverse of average distance)
        similarity_score = 1 / (1 + average_distance)  # Adding 1 to avoid division by zero
        similarity_scores.append(similarity_score)
    return similarity_scores

# Calculate similarity scores for each cluster center
similarity_scores = calculate_cluster_similarity(cluster_assignments, cluster_centers, X)

for idx, score in enumerate(similarity_scores):
    print(f"Cluster {idx} Similarity Score: {score:.4f}")

print("\nCluster Centers:")
print(cluster_centers)
