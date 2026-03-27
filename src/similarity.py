import numpy as np
from scipy.spatial.distance import braycurtis


def build_class_mean_vectors(triplets, head_embeddings, relation_embeddings, tail_to_hr):
    """
    Create one mean vector per tail class (using concatenated h+r embeddings).
    """
    class_to_vectors = defaultdict(list)

    for h, r, t in triplets:
        hr_vec = np.concatenate([head_embeddings[h], relation_embeddings[r]])
        class_to_vectors[t].append(hr_vec)   # Use tail string as class key (more readable)

    # Compute mean vector per class
    class_vectors = []
    class_labels = []   # Keep track of tail names if needed
    for tail, vectors in class_to_vectors.items():
        mean_vec = np.mean(vectors, axis=0)
        class_vectors.append(mean_vec)
        class_labels.append(tail)

    return np.array(class_vectors), class_labels