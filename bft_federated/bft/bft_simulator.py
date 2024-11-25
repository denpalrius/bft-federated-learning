from typing import List
import torch
from torch.utils.data import DataLoader, random_split
import flwr as fl
from flwr.server.strategy import Strategy
from bft.bft_strategy import BFTFedAvg
from clients.federated_client import FederatedClient
from utils.base_config import BaseConfig
from utils.bft_simulation_config import BFTSimulationConfig


class BFTSimulator:
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

        return client_data

    def create_clients(self, partitions: List[tuple]) -> List[fl.client.Client]:
        show_summary = True

        federated_client = FederatedClient(self.sim_config)
        clients = federated_client.create_clients(partitions, show_summary)

        return clients

    def create_strategy(self) -> Strategy:
        """Create BFT aggregation strategy."""
        return BFTFedAvg(
            threshold=self.base_config.threshold,
            bft_method=self.sim_config.bft_method,
        )

    def run(self) -> dict:
        print(f"\nRunning simulation with BFT method: {self.sim_config.bft_method}")

        client_datasets = self.prepare_data()
        clients = self.create_clients(client_datasets)
        strategy = self.create_strategy()

        def client_fn() -> List[fl.client.Client]:
            return clients
        
            
        print(f'Simulation config: {self.sim_config}')
        
        results = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=self.sim_config.num_clients,
            strategy=strategy,
        )

        return {
            "accuracy": results.get("accuracy", 0.0),
            "loss": results.get("loss", 0.0),
        }



