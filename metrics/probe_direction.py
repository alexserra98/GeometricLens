import numpy as np

def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:  # To handle zero division error
        return 0
    return dot_product / (norm_vec1 * norm_vec2)

def compute_cosine_similarities(matrices):
    """Compute cosine similarities between corresponding rows of consecutive matrices."""
    similarities = []
    num_matrices = len(matrices)

    for j in range(num_matrices - 1):
        matrix_1 = matrices[j]
        matrix_2 = matrices[j + 1]
        
        if matrix_1.shape[0] != matrix_2.shape[0]:
            raise ValueError("Matrices must have the same number of rows")

        row_similarities = []
        for i in range(matrix_1.shape[0]):
            sim = cosine_similarity(matrix_1[i], matrix_2[i])
            row_similarities.append(sim)
        similarities.append(row_similarities)
    
    return similarities

# Example usage
if __name__ == "__main__":
    # Creating example matrices
    matrix_list = [
        np.array([[1, 2], [3, 4], [5, 6]]),
        np.array([[2, 3], [4, 5], [6, 7]]),
        np.array([[1, 1], [0, 0], [1, 2]])
    ]

    result = compute_cosine_similarities(matrix_list)
    print(result)
