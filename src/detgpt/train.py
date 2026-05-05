from __future__ import annotations

import torch

from detgpt.device import DeviceSpec, resolve_torch_device
from detgpt.model import Model


def train(epochs: int = 1, learning_rate: float = 1e-2, device: DeviceSpec = None) -> Model:
    """Run a minimal training scaffold.

    This entrypoint keeps ``train.py`` focused on training and can be replaced
    with project-specific data loading and optimization code as experiments grow.

    Args:
        epochs: Number of synthetic training epochs.
        learning_rate: Optimizer learning rate.
        device: PyTorch device. Use ``None`` or ``"auto"`` for CUDA, MPS, then CPU.

    Returns:
        Trained model instance.
    """
    resolved_device = resolve_torch_device(device)
    model = Model().to(resolved_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(epochs):
        inputs = torch.rand(64, 1, device=resolved_device)
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
