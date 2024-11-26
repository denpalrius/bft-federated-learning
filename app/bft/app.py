from utils.path import add_base_path

from dataclasses import dataclass
from typing import Callable, List
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
from torchvision import transforms
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from clients.federated_client import FederatedClient
from clients.malicious_client import MaliciousClient
from nets.model import CNNClassifier
from utils.base_config import BaseConfig
from utils.bft_simulation_config import BFTSimulationConfig
from bft.bft_simulator import BFTSimulator

add_base_path(__file__)


def run_experiments(
    experiment_name: str,
    base_config: BaseConfig,
    dataset_fn: Callable[[], Dataset],
    model_fn: Callable[[], torch.nn.Module],
) -> List[dict]:
    """Run experiments with different BFT methods."""
    results = []

    print(f"Running experiments for {experiment_name}")
    
    # TODO: Make number of clients dynamic between 20 and 100
    # Simulate clients dropping out randomly
    
    for bft_method in base_config.bft_methods:
        sim_config = BFTSimulationConfig(
            experiment_name=experiment_name,
            bft_method=bft_method,
            base_config=base_config,
            dataset_fn=dataset_fn,
            model_fn=model_fn,
        )

        simulator = BFTSimulator(base_config, sim_config)
        result = simulator.run()
        # results.append(
        #     {
        #         "method": bft_method,
        #         "accuracy": result["accuracy"],
        #         "loss": result["loss"],
        #     }
        # )

    return results


if __name__ == "__main__":
    def get_cifar10() -> Dataset:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ]
        )
        return torchvision.datasets.CIFAR10(
            root="../data", train=True, download=True, transform=transform
        )

    def get_simple_cnn() -> torch.nn.Module:
        return CNNClassifier()

    results_exp_1 = run_experiments(
        experiment_name="cifar10_bft",
        base_config=BaseConfig(),
        dataset_fn=get_cifar10,
        model_fn=get_simple_cnn,
    )

    for res in results_exp_1:
        print(f"\nMethod: {res['method']}")
        print(f"Final accuracy: {res['accuracy']:.4f}")
        print(f"Final loss: {res['loss']:.4f}")

    # TODO: Add more experiments