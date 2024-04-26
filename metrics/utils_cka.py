import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel using PyTorch."""
    x = torch.tensor(x, dtype=torch.float32, device=device)
    return x.matmul(x.t()).cpu().numpy()



def gram_rbf(X, sigma=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel using PyTorch."""
    X = torch.tensor(X, dtype=torch.float32, device=device)
    sq_dists = torch.cdist(X, X) ** 2
    return torch.exp(-sq_dists / (2 * sigma ** 2)).cpu().numpy()

def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix using PyTorch.

    This is equivalent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
      gram: A num_examples x num_examples symmetric matrix.
      unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
                estimate of HSIC. Note that this estimator may be negative.

    Returns:
      A symmetric matrix with centered columns and rows.
    """
    gram = torch.tensor(gram, dtype=torch.float32, device=device)
    
    if not torch.allclose(gram, gram.t(), atol=1e-4):
        raise ValueError('Input must be a symmetric matrix.')
    
    n = gram.shape[0]

    if unbiased:
        torch.fill_diagonal_(gram, 0)
        means = torch.sum(gram, 0) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        torch.fill_diagonal_(gram, 0)
    else:
        means = torch.mean(gram, 0)
        means -= torch.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram.cpu().numpy()



def cka(gram_x, gram_y, debiased=False):
    """Compute CKA using PyTorch.

    Args:
      gram_x: A num_examples x num_examples Gram matrix.
      gram_y: A num_examples x num_examples Gram matrix.
      debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
      The value of CKA between X and Y.
    """
    # Center the Gram matrices
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)
    #import pdb; pdb.set_trace()
    # Convert to torch tensors
    gram_x = torch.tensor(gram_x, dtype=torch.float32, device=device)
    gram_y = torch.tensor(gram_y, dtype=torch.float32, device=device)

    # Compute scaled HSIC (element-wise product and sum of flattened matrices)
    scaled_hsic = (gram_x.view(-1) * gram_y.view(-1)).sum()

    # Compute normalizations
    normalization_x = torch.norm(gram_x)
    normalization_y = torch.norm(gram_y)

    # Compute and return CKA
    cka_value = scaled_hsic / (normalization_x * normalization_y)
    return cka_value.cpu().numpy()


def _debiased_dot_product_similarity_helper(xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n):
    """Helper for computing debiased dot product similarity (i.e., linear HSIC) using PyTorch."""
    return (
        xty - n / (n - 2.) * torch.dot(sum_squared_rows_x, sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2))
    )


def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space, using PyTorch.

    Args:
        features_x: A num_examples x num_features matrix of features.
        features_y: A num_examples x num_features matrix of features.
        debiased: Use unbiased estimator of dot product similarity. CKA may still be biased.

    Returns:
        The value of CKA between X and Y.
    """
    
    features_x = torch.tensor(features_x, dtype=torch.float32, device=device)
    features_x = features_x- torch.mean(features_x, dim=0, keepdim=True)
    features_y = torch.tensor(features_y, dtype=torch.float32, device=device)
    features_y = features_y- torch.mean(features_y, dim=0, keepdim=True)

    dot_product_similarity = torch.norm(features_x.t().matmul(features_y)) ** 2
    normalization_x = torch.norm(features_x.t().matmul(features_x))
    normalization_y = torch.norm(features_y.t().matmul(features_y))

    if debiased:
        n = features_x.shape[0]
        sum_squared_rows_x = torch.sum(features_x ** 2, dim=1)
        sum_squared_rows_y = torch.sum(features_y ** 2, dim=1)
        squared_norm_x = torch.sum(sum_squared_rows_x)
        squared_norm_y = torch.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        normalization_x = torch.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = torch.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))

    return (dot_product_similarity / (normalization_x * normalization_y)).cpu().numpy()

