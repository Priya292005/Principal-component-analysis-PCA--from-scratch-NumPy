Python 3.13.2 (tags/v3.13.2:4f8bb39, Feb  4 2025, 15:23:48) [MSC v.1942 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Implementing and Visualizing PCA from Scratch
... # Author: Priya
... 
... import numpy as np
... import matplotlib.pyplot as plt
... from sklearn.datasets import make_blobs
... from sklearn.decomposition import PCA
... 
... # -------------------------------------------------
... # 1. Generate synthetic dataset (200 samples, 10 dimensions)
... # -------------------------------------------------
... np.random.seed(42)
... 
... X, y = make_blobs(
...     n_samples=200,
...     n_features=10,
...     centers=3,
...     cluster_std=2.5
... )
... 
... # -------------------------------------------------
... # 2. Standardize the data (mean=0, std=1)
... # -------------------------------------------------
... X_mean = np.mean(X, axis=0)
... X_std = np.std(X, axis=0)
... 
... X_standardized = (X - X_mean) / X_std
... 
... # -------------------------------------------------
... # 3. Compute covariance matrix
... # -------------------------------------------------
... cov_matrix = np.cov(X_standardized, rowvar=False)
... 
... # -------------------------------------------------
... # 4. Eigenvalue and Eigenvector decomposition
... # -------------------------------------------------
... eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# -------------------------------------------------
# 5. Sort eigenvalues and eigenvectors (descending)
# -------------------------------------------------
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# -------------------------------------------------
# 6. Explained variance ratio (scratch)
# -------------------------------------------------
explained_variance_ratio_scratch = eigenvalues / np.sum(eigenvalues)

print("Explained Variance Ratio (Scratch PCA):")
print(explained_variance_ratio_scratch[:3])

# -------------------------------------------------
# 7. Select top K principal components
# -------------------------------------------------
K = 2  # change to 3 for 3D visualization
principal_components = eigenvectors[:, :K]

# -------------------------------------------------
# 8. Project data onto principal components
# -------------------------------------------------
X_pca_scratch = np.dot(X_standardized, principal_components)

# -------------------------------------------------
# 9. PCA using scikit-learn (comparison)
# -------------------------------------------------
pca = PCA(n_components=K)
X_pca_sklearn = pca.fit_transform(X_standardized)

print("\nExplained Variance Ratio (Sklearn PCA):")
print(pca.explained_variance_ratio_)

# -------------------------------------------------
# 10. Compare explained variance ratios
# -------------------------------------------------
difference = np.abs(
    explained_variance_ratio_scratch[:K] -
    pca.explained_variance_ratio_
)

print("\nDifference in Explained Variance Ratios:")
print(difference)

# -------------------------------------------------
# 11. 2D Visualization (K = 2)
# -------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca_scratch[:, 0],
    X_pca_scratch[:, 1],
    c=y,
    cmap='viridis'
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA from Scratch (2D Projection)")
plt.colorbar(label="Cluster")
plt.show()

# -------------------------------------------------
# 12. 3D Visualization (Optional, K = 3)
# -------------------------------------------------
if K == 3:
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        X_pca_scratch[:, 0],
        X_pca_scratch[:, 1],
        X_pca_scratch[:, 2],
        c=y,
        cmap='viridis'
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA from Scratch (3D Projection)")
    plt.show()
