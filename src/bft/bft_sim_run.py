from dataclasses import dataclass
from typing import Callable, List, Optional
import os
import numpy as np
import torch
from fed.config_base import BaseConfig
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.nn import Sequential, Conv2d, Flatten, Linear, ReLU
from torch.utils.data import DataLoader, Dataset, random_split
import flwr as fl

from bft.config_sim import BFTSimulationConfig


# TODO: To clean up

class BFTSimulationRunner:    
    def __init__(self, config: BFTSimulationConfig):
        self.config = config
        
    def prepare_data(self, dataset: Dataset) -> List[tuple]:
        """Split dataset among clients."""
        samples_per_client = len(dataset) // self.config.num_clients
        client_data = []
        
        for i in range(self.config.num_clients):
            # Get client's partition
            start = i * samples_per_client
            end = start + samples_per_client if i < self.config.num_clients - 1 else len(dataset)
            partition = dataset[start:end]
            
            # Split into train/val
            train_size = int(0.8 * len(partition))
            train_data, val_data = random_split(partition, [train_size, len(partition) - train_size])
            
            # Create dataloaders
            trainloader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
            valloader = DataLoader(val_data, batch_size=self.config.batch_size)
            client_data.append((trainloader, valloader))
            
        return client_data

    def create_clients(self, partitions: List[tuple]) -> List[fl.client.Client]:
        """Create mix of benign and malicious clients."""
        clients = []
        malicious_indices = np.random.choice(
            self.config.num_clients,
            size=self.config.num_malicious,
            replace=False
        )
        # TODO: Experiment with different attack strategies
        
        for i, (trainloader, valloader) in enumerate(partitions):
            model = self.config.model_fn().to(self.config.device)
            
            if i in malicious_indices:
                # Malicious client with random attack
                client = self.create_malicious_client(model, trainloader, valloader)
            else:
                # Benign client
                client = self.create_benign_client(model, trainloader, valloader)
                
            clients.append(client)
        return clients

    def run(self) -> dict:
        """Execute federated learning simulation."""
        dataset = self.config.dataset_fn()
        partitions = self.prepare_data(dataset)
        clients = self.create_clients(partitions)
        
        strategy = self.create_strategy()
        
        # Run simulation
        results = fl.simulation.start_simulation(
            clients=clients,
            num_rounds=self.config.num_rounds,
            strategy=strategy
        )
        
        return self.process_results(results)


if __name__ == "__main__":

    base_config = BaseConfig()
    base_config.num_rounds = 3
    
    # Dataset function
    def cifar10_dataset():
        transform = ToTensor()
        dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        return dataset

    # Model function
    def simple_cnn():
        return Sequential(
            Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Flatten(),
            Linear(32 * 32 * 32, 10)  # CIFAR-10 has 10 classes
        )

    # Initialize configuration
    config = BFTSimulationConfig(
        base_config=base_config,
        dataset_fn=cifar10_dataset,
        model_fn=simple_cnn
    )

    # Run simulation
    runner = BFTSimulationRunner(config)
    results = runner.run()
    print("Simulation Results:", results)
