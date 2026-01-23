import pytest
import torch

from src.mlops.model import TinyCNN


@pytest.mark.parametrize(
    "batch_size, num_classes, height, width",
    [
        (1, 43, 64, 64),  # Single image
        (32, 43, 64, 64),  # A full batch
    ],
)
def test_model_forward_pass(batch_size, num_classes, height, width):
    """
    Tests the forward pass of the TinyCNN model.
    Checks if the output tensor has the correct shape.
    """
    # 1. Create an instance of the model
    model = TinyCNN(num_classes=num_classes)

    # 2. Create a dummy input tensor with the correct dimensions
    # (batch_size, channels, height, width)
    dummy_input = torch.randn(batch_size, 3, height, width)

    # 3. Perform a forward pass
    output = model(dummy_input)

    # 4. Assert that the output shape is correct
    # It should be (batch_size, num_classes)
    assert output.shape == (batch_size, num_classes)
    assert output.dtype == torch.float32
