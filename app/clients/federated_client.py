from typing import List
import numpy as np
from flwr.client import NumPyClient
from bft_federated.nets.model import CNNClassifier
from utils.bft_simulation_config import BFTSimulationConfig
from nets.train import ModelTrainer
from torch.utils.data import DataLoader
from clients.genuine_client import GenuineClient
from clients.malicious_client import MaliciousClient


class FederatedClient:
    """Handles the creation and management of federated clients."""

    def __init__(
        self,
        sim_config: BFTSimulationConfig,
    ):
        self.sim_config = sim_config
        self.attack_types = sim_config.get_attack_types()
        self.attack_intensities = sim_config.get_attack_intensities()
        self.malicious_indices = np.random.choice(
            self.sim_config.num_clients,
            size=self.sim_config.num_malicious,
            replace=False,
        )

    def create_genuine_client(
        self,
        partition_idx: int,
        trainer: ModelTrainer,
        trainloader: DataLoader,
        valloader: DataLoader,
    ) -> GenuineClient:
        return GenuineClient(
            partition_id=partition_idx,
            trainer=trainer,
            trainloader=trainloader,
            valloader=valloader,
        )

    def create_malicious_client(
        self,
        partition_idx: int,
        trainer: ModelTrainer,
        trainloader: DataLoader,
        valloader: DataLoader,
    ) -> MaliciousClient:
        # Randomly select attack type and intensity
        attack_type = np.random.choice(self.attack_types)
        attack_intensity = np.random.choice(self.attack_intensities)

        return MaliciousClient(
            partition_id=partition_idx,
            trainer=trainer,
            trainloader=trainloader,
            valloader=valloader,
            attack_type=attack_type,
            attack_intensity=attack_intensity,
        )

    def create_client(
        self,
        partition_idx: int,
        trainer: ModelTrainer,
        trainloader: DataLoader,
        valloader: DataLoader,
        attack_type: str = None,
        attack_intensity: float = None,
    ) -> NumPyClient:
        if attack_type is not None:
            return self.create_malicious_client(
                partition_idx,
                trainer,
                trainloader,
                valloader,
                attack_type,
                attack_intensity,
            )
        else:
            return self.create_genuine_client(
                partition_idx, trainer, trainloader, valloader
            )

    def create_clients(
        self, partitions: List[tuple], show_summary: False
    ) -> List[NumPyClient]:
        """
        Create a mix of genuine and malicious clients based on the configuration.
        """
        clients = []

        assert self.sim_config.num_clients == len(
            partitions
        ), "Number of clients must match number of partitions"

        malicious_indices = np.random.choice(
            len(partitions),
            size=self.sim_config.num_malicious,
            replace=False,
        )

        for i, (trainloader, valloader) in enumerate(partitions):
            trainer = ModelTrainer(
                self.sim_config.model_fn(),
                device=self.sim_config.device,
            )

            if i in malicious_indices:
                # print(f"Creating Malicious Client {i}")
                client = self.create_malicious_client(
                    i, trainer, trainloader, valloader
                )
            else:
                # print(f"Creating Genuine Client {i}")
                client = self.create_genuine_client(i, trainer, trainloader, valloader)

            clients.append(client)

        if show_summary:
            self.print_client_summary(clients)

        return clients

    def create_client(
        self,
        model: CNNClassifier,
        partition_id: str,
        trainloader: DataLoader,
        valloader: DataLoader,
        show_summary: bool = False,
    ) -> NumPyClient:
        trainer = ModelTrainer(model, device=self.sim_config.device)

        if partition_id in self.malicious_indices:
            print(f"=========Creating Malicious Client {i}")
            client = self.create_malicious_client(
                partition_id, trainer, trainloader, valloader
            )
        else:
            print(f"==========Creating Genuine Client {partition_id}")
            client = self.create_genuine_client(
                partition_id, trainer, trainloader, valloader
            )

        return client

    # def print_client_summary(self, clients: List[NumPyClient]):
    #     print(f"\n{len(clients)} clients created")
    #     malicious_clients = [
    #         client for client in clients if hasattr(client, "attack_type")
    #     ]
    #     genuine_clients = [
    #         client for client in clients if not hasattr(client, "attack_type")
    #     ]

    #     print(f"- {len(malicious_clients)} malicious clients")
    #     print(f"- {len(genuine_clients)} genuine clients")

    #     if malicious_clients:
    #         print("Malicious clients Details:")
    #         for client in malicious_clients:
    #             print(
    #                 f"  - Client ID: {client.partition_id}, Attack Type: {client.attack_type}, Intensity: {client.attack_intensity}"
    #             )
