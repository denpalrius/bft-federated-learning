from dataclasses import dataclass
from typing import Callable
import os
import torch
from torch.utils.data import Dataset
from fed.config_base import BaseConfig

@dataclass
class BFTSimulationConfig:
    """Configuration for BFT Federated Learning simulation."""
    name: str
    num_clients: int
    num_malicious: int
    num_rounds: int
    batch_size: int
    bft_method: str
    device: str
    dataset_fn: Callable[[], Dataset]
    model_fn: Callable[[], torch.nn.Module]

    def __init__(self, experiment_name:str, bft_method, base_config: BaseConfig, dataset_fn: Callable[[], Dataset], model_fn: Callable[[], torch.nn.Module]):
        self.name = experiment_name
        self.num_clients = base_config.num_clients
        self.num_malicious = base_config.num_malicious
        self.num_rounds = base_config.num_rounds
        self.batch_size = base_config.batch_size
        self.bft_method = bft_method
        self.device = base_config.device
        self.dataset_fn = dataset_fn
        self.model_fn = model_fn

        if self.num_malicious >= self.num_clients:
            raise ValueError("Number of malicious clients must be less than total clients")

        os.makedirs(f"results/{self.name}", exist_ok=True)
