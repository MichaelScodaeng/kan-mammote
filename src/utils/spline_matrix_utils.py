# kan_mamote/src/utils/spline_matrix_utils.py

import torch

def compute_bspline_matrix(k_order: int) -> torch.Tensor:
    """
    Computes the Psi^k (bspline_matrix) as defined in MatrixKAN (Eq. 10).
    This matrix represents the transformation from the power basis [1, u, ..., u^(k-1)]
    to the B-spline basis functions of order k.

    Args:
        k_order (int): The order of the B-spline (e.g., 1 for degree 0, 2 for degree 1, 3 for degree 2, etc.).
                       This is 'k' in Psi^k.

    Returns:
        torch.Tensor: The (k_order x k_order) Psi^k matrix.

    Reference:
        MatrixKAN: Parallelized Kolmogorov-Arnold Network (Eq. 10)
        https://github.com/OSU-STARLAB/MatrixKAN/blob/main/matrixkan/spline_matrix.py
    """
    if k_order < 1:
        raise ValueError("B-spline order (k_order) must be at least 1.")

    if k_order == 1:
        # Base case: 0-order B-spline (constant 1 within interval)
        return torch.tensor([[1.0]], dtype=torch.float32)

    # Recursive definition as per MatrixKAN's Eq. 10 and source code
    # Psi^k = sum_{i=0}^{k-1} sum_{j=0}^{k-1} C_ij * Psi^(k-1)[i,j]
    # This translates to specific additions/subtractions of the previous matrix.
    
    # Get the matrix for the previous order (k-1)
    psi_k_minus_1 = compute_bspline_matrix(k_order - 1)
    
    # Initialize the current matrix for order k_order
    psi_k = torch.zeros((k_order, k_order), dtype=torch.float32)

    # Fill based on the previous matrix (k-1)x(k-1)
    # The coefficients are 1 / (k-1) and -1 / (k-1) for specific positions.
    for i in range(k_order - 1):
        for j in range(k_order - 1):
            # Coefficient for Psi^(k-1)[i,j] is 1/(k_order-1)
            psi_k[i, j] += psi_k_minus_1[i, j] / (k_order - 1)
            psi_k[i, j + 1] -= psi_k_minus_1[i, j] / (k_order - 1) # This is B_{i,k-1}(x) * (x - t_i)
            psi_k[i + 1, j] -= psi_k_minus_1[i, j] / (k_order - 1) # This is B_{i+1,k-1}(x) * (t_{i+k} - x)
            psi_k[i + 1, j + 1] += psi_k_minus_1[i, j] / (k_order - 1)

    return psi_k

# Example usage (for testing purposes)
# if __name__ == '__main__':
#     # Psi^1 (k_order=1, degree=0)
#     # Expected: [[1.]]
#     print(f"Psi^1:\n{compute_bspline_matrix(1)}")

#     # Psi^2 (k_order=2, degree=1, linear)
#     # Expected: [[1., -1.], [-1., 1.]]
#     print(f"Psi^2:\n{compute_bspline_matrix(2)}")

#     # Psi^3 (k_order=3, degree=2, quadratic)
#     # Expected: [[0.5, -1., 0.5], [-1., 2., -1.], [0.5, -1., 0.5]] (values will be scaled by 1/(k-1) factor)
#     print(f"Psi^3:\n{compute_bspline_matrix(3)}")

#     # Psi^4 (k_order=4, degree=3, cubic)
#     print(f"Psi^4:\n{compute_bspline_matrix(4)}")