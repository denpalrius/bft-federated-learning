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
from client_federated import FederatedClient
from client_malicious import MaliciousClient
from fed.model import CNNClassifier
from config_base import BaseConfig
from config_sim import BFTSimulationConfig
from bft_simulator import BFTSimulator
from train import ModelTrainer


def run_experiments(
    experiment_name: str,
    base_config: BaseConfig,
    dataset_fn: Callable[[], Dataset],
    model_fn: Callable[[], torch.nn.Module],
) -> List[dict]:
    """Run experiments with different BFT methods."""
    results = []

    print(f"Running experiments for {experiment_name}...")
    
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
        results.append(
            {
                "method": bft_method,
                "accuracy": result["accuracy"],
                "loss": result["loss"],
            }
        )

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
            root="./data", train=True, download=True, transform=transform
        )

    def get_simple_cnn() -> torch.nn.Module:
        return CNNClassifier()

    results = run_experiments(
        experiment_name="cifar10_bft_experiment",
        base_config=BaseConfig(),
        dataset_fn=get_cifar10,
        model_fn=get_simple_cnn,
    )

    # for res in results:
    #     print(f"\nMethod: {res['method']}")
    #     print(f"Final accuracy: {res['accuracy']:.4f}")
    #     print(f"Final loss: {res['loss']:.4f}")
