import torch
from src.models.kan.MatrixKANLayer import MatrixKANLayer

def test_matrix_kan_layer():
    # Set input and output dimensions
    in_dim = 3
    out_dim = 2
    batch_size = 4

    # Create random input tensor
    x = torch.randn(batch_size, in_dim)

    # Instantiate the layer    python -m tests.test_matrix_kan_layer
    layer = MatrixKANLayer(in_dim=in_dim, out_dim=out_dim)

    # Forward pass
    y, preacts, postacts, postspline = layer(x)

    # Print output shapes
    print("Output y shape:", y.shape)
    print("Preacts shape:", preacts.shape)
    print("Postacts shape:", postacts.shape)
    print("Postspline shape:", postspline.shape)

    # Optionally, add assertions
    assert y.shape == (batch_size, out_dim)
    assert preacts.shape[0] == batch_size
    assert postacts.shape[0] == batch_size
    assert postspline.shape[0] == batch_size

if __name__ == "__main__":
    test_matrix_kan_layer()