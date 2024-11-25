from typing import List
import torch
import flwr
from dataclasses import dataclass, field

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Flower: {flwr.__version__} / PyTorch {torch.__version__}")

# Specify the resources each of your clients need
# If set to none, by default, each client will be allocated 2x CPU and 0x GPUs
backend_config = {"client_resources": None}
if device.type == "cuda":
    backend_config = {"client_resources": {"num_gpus": 1}}


@dataclass
class BaseConfig:
    # Device
    device: torch.device = device

    # Using default_factory for mutable default
    backend_config: dict = field(default_factory=lambda: backend_config)

    # Training
    epochs: int = 10
    lr: float = 0.001
    momentum: float = 0.9
    log_interval: int = 100
    batch_size: int = 32

    # Dataset
    dataset: str = "cifar10"
    num_partitions: int = 10
    num_clients: int = 10
    num_rounds: int = 3 # Number of rounds to simulate
    # num_rounds: int = 50 # Number of rounds to simulate
    
    # BFT
    num_malicious: int = 2
    bft_methods: List[str] = field(default_factory=lambda: ["krum", "trimmed_mean", "median"])
    bft_method: str = "krum"
    attack_types: List[str] = field(default_factory=lambda: ["sign_flip", "gaussian_noise", "constant_bias", "zero_update"])

    # BFT Strategy
    threshold: float = 0.5
    