from __future__ import annotations

import torch

from detgpt.model import Model


def train(epochs: int = 1, learning_rate: float = 1e-2) -> Model:
    """Run a minimal training scaffold.

    This entrypoint keeps ``train.py`` focused on training and can be replaced
    with project-specific data loading and optimization code as experiments grow.

    Args:
        epochs: Number of synthetic training epochs.
        learning_rate: Optimizer learning rate.

    Returns:
        Trained model instance.
    """
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(epochs):
        inputs = torch.rand(64, 1)
        targets = 2.0 * inputs + 1.0

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

    return model


if __name__ == "__main__":
    trained_model = train()
    print(f"Training completed. Sample weight: {trained_model.layer.weight.item():.4f}")
