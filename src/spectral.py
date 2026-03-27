import numpy as np
from scipy.linalg import eigh


def compute_class_overlap(class_vectors):
    """Compute similarity matrix using 1 - Bray-Curtis distance."""
    num_classes = len(class_vectors)
    similarity_matrix = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            dist = braycurtis(class_vectors[i], class_vectors[j])
            similarity_matrix[i, j] = 1 - dist

    return similarity_matrix


def compute_csg(similarity_matrix):
    """Compute Cumulative Spectral Gradient (simplified version from your code)."""
    # Unnormalized Laplacian
    degree = np.sum(similarity_matrix, axis=1)
    laplacian = np.diag(degree) - similarity_matrix

    # Eigenvalues (sorted ascending)
    eigenvalues = eigh(laplacian, eigvals_only=True)
    eigenvalues = np.sort(eigenvalues)

    if len(eigenvalues) <= 1:
        return 0.0

    # Eigengaps
    eigengaps = np.diff(eigenvalues)

    # Normalization (your original logic)
    n = len(eigenvalues)
    normalized_eigengaps = eigengaps / (n - np.arange(1, n))

    csg = np.cumsum(normalized_eigengaps)[-1]
    return float(csg)