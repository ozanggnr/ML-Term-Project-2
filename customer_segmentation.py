import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

csv_file_path = 'customer_segments.csv'
df = pd.read_csv(csv_file_path)

print(f"\nData shape: {df.shape}")
print(f"\nMissing values per column:")
print(df.isnull().sum())
print(f"\nData types:")
print(df.dtypes)

feature_columns = [col for col in df.columns if col != 'Archetype']
X = df[feature_columns].copy()

print("\nHandling missing values...")
numerical_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(include=['object']).columns

if len(numerical_cols) > 0:
    for col in numerical_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    numerical_imputer = SimpleImputer(strategy='median')
    X[numerical_cols] = numerical_imputer.fit_transform(X[numerical_cols])

if len(categorical_cols) > 0:
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

X_processed = X.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

print(f"\nProcessed data shape: {X_scaled.shape}")
print("Data preprocessing completed.")

# PART 1: Finding optimal number of clusters

k_range = range(2, 21)
wcss = []
silhouette_scores = []

print("\nCalculating WCSS and Silhouette scores for different numbers of clusters...")

for k in k_range:
    kmeans = KMeans(
        n_clusters=k, 
        random_state=42, 
        n_init=20,
        max_iter=500
    )
    kmeans.fit(X_scaled)
    
    wcss.append(kmeans.inertia_)
    
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette_avg)
    
    print(f"k={k}: WCSS={kmeans.inertia_:.2f}, Silhouette Score={silhouette_avg:.4f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(k_range)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Method for Optimal k', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(k_range)

plt.tight_layout()
plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
print("\nGraphs saved as 'cluster_analysis.png'")
plt.show()

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters based on Silhouette Score: {optimal_k}")
print(f"Maximum Silhouette Score: {max(silhouette_scores):.4f}")
print(f"\nNumber of clusters found in the first deliverable: {optimal_k}")

print("\nCreating 2D visualization using PCA...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

kmeans_vis = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter=500)
labels_vis = kmeans_vis.fit_predict(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_vis, cmap='viridis', alpha=0.6, s=50)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'First Principal Component (explained variance: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
plt.ylabel(f'Second Principal Component (explained variance: {pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
plt.title(f'2D Visualization of Clusters (k={optimal_k})', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('2d_cluster_visualization.png', dpi=300, bbox_inches='tight')
print("2D visualization saved as '2d_cluster_visualization.png'")
plt.show()

# PART 2: Final Clustering Pipeline

final_k = optimal_k

print(f"\nRunning final clustering with k={final_k}...")

best_silhouette = -1
best_labels = None

print("Running multiple initializations to find best clustering...")
for run in range(15):
    final_kmeans = KMeans(
        n_clusters=final_k, 
        random_state=42 + run,
        n_init=20,
        max_iter=500
    )
    labels = final_kmeans.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)
    
    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_labels = labels

final_labels = best_labels

final_silhouette_score = silhouette_score(X_scaled, final_labels)

print(f"Final Silhouette Score: {final_silhouette_score:.4f}")

unique_labels, counts = np.unique(final_labels, return_counts=True)

print(f"\nNumber of clusters you found in the first deliverable {optimal_k}")
print(f"Silhouette score you found in the second deliverable {final_silhouette_score:.4f}")
print(f"All of the clusters obtained in the second deliverable Number of instances")

for label, count in zip(unique_labels, counts):
    print(f"Cluster {label} {count}")
