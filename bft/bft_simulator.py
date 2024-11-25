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
from flwr.server.strategy import FedAvg, Strategy
from bft.bft_strategy import BFTFedAvg
from bft.client_federated import FederatedClient
from bft.client_malicious import MaliciousClient
from fed.model import CNNClassifier
from fed.config_base import BaseConfig
from bft.config_sim import BFTSimulationConfig
from fed.train import ModelTrainer


class BFTSimulator:
    """Runs BFT Federated Learning simulations."""

    def __init__(self, base_config: BaseConfig, sim_config: BFTSimulationConfig):
        self.base_config = base_config
        self.sim_config = sim_config

    def prepare_data(self) -> List[tuple]:
        """Split dataset among clients."""
        dataset = self.sim_config.dataset_fn()
        
        samples_per_client = len(dataset) // self.sim_config.num_clients
        client_data = []

        for i in range(self.sim_config.num_clients):
            # Get client's partition
            start = i * samples_per_client
            end = (
                start + samples_per_client
                if i < self.sim_config.num_clients - 1
                else len(dataset)
            )
            # print(f"\nPartition {i+1}: Start: {start}, End: {end-1}")

            partition_indices = list(range(start, end))
            partition = torch.utils.data.Subset(dataset, partition_indices)

            # Split into train/val
            train_size = int(0.8 * len(partition))
            val_size = len(partition) - train_size
            train_data, val_data = random_split(partition, [train_size, val_size])

            trainloader = DataLoader(
                train_data,
                batch_size=self.sim_config.batch_size,
                shuffle=True,
                num_workers=2,
            )
            valloader = DataLoader(val_data, batch_size=self.sim_config.batch_size)
            client_data.append((trainloader, valloader))

        print(f"Total number of clients: {len(client_data)}")

        return client_data

    def create_clients(self, partitions: List[tuple]) -> List[fl.client.Client]:
        """Create mix of genuine and malicious clients."""
        clients = []
        malicious_indices = np.random.choice(
            self.sim_config.num_clients,
            size=self.sim_config.num_malicious,
            replace=False,
        )

        for i, (trainloader, valloader) in enumerate(partitions):
            model = self.sim_config.model_fn().to(self.sim_config.device)

            if i in malicious_indices:
                client = self.create_malicious_client(model, trainloader, valloader)
            else:
                client = self.create_genuine_client(model, trainloader, valloader)

            clients.append(client)
        return clients

    def create_genuine_client(
        self,
        model: CNNClassifier,
        trainloader: DataLoader,
        valloader: DataLoader,
        trainer: ModelTrainer,
    ) -> fl.client.Client:

        fed_client = FederatedClient(config="config")
        client_app = ClientApp(client_fn=fed_client.create_client)
        return client_app

    def create_malicious_client(
        self,
        partition_idx: int,
        model: CNNClassifier,
        trainer: ModelTrainer,
        trainloader: DataLoader,
        valloader: DataLoader,
    ) -> fl.client.Client:
        attack_type = np.random.choice(self.sim_config.attack_types)
        attack_intensity = np.random.choice(self.sim_config.attack_intensities)
        client = MaliciousClient(
            partition_id=partition_idx,
            trainer=trainer,
            trainloader=trainloader,
            valloader=valloader,
            attack_type=attack_type,
            attack_intensity=attack_intensity,
        )
        return client

    def create_strategy(self) -> Strategy:
        """Create BFT aggregation strategy."""
        return BFTFedAvg(
            threshold=self.base_config.threshold,
            bft_method=self.sim_config.bft_method,
        )

    def process_results(self, results):
        """Process and summarize results."""
        return {
            "accuracy": results.get("accuracy", 0.0),
            "loss": results.get("loss", 0.0),
        }

    def run(self) -> dict:
        print(f"Running simulation with BFT method: {self.sim_config.bft_method}")

        partitions = self.prepare_data()
        clients = self.create_clients(partitions)
        strategy = self.create_strategy()
        

        # Run simulation
        # results = fl.simulation.start_simulation(
        #     clients=clients, num_rounds=self.sim_config.num_rounds, strategy=strategy
        # )

        # return self.process_results(results)
