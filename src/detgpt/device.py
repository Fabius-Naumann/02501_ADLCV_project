from __future__ import annotations

import torch

DeviceSpec = str | torch.device | None


def mps_is_available() -> bool:
    """Return whether PyTorch can execute workloads on Apple's Metal backend."""
    return bool(torch.backends.mps.is_available())


def get_default_device() -> torch.device:
    """Resolve the best available PyTorch device for local workloads."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_torch_device(device: DeviceSpec = None) -> torch.device:
    """Resolve a user-provided or automatic PyTorch device.

    Args:
        device: Optional device specifier. Use ``None`` or ``"auto"`` to select
            CUDA, then MPS, then CPU.

    Returns:
        Resolved PyTorch device.
    """
    if device is None or (isinstance(device, str) and device.strip().lower() == "auto"):
        return get_default_device()

    return torch.device(device)
