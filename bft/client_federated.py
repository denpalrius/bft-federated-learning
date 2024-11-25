from typing import Dict, List, Tuple
import numpy as np
from torch.utils.data import DataLoader
from flwr.client import ClientApp, NumPyClient
from bft_client_genuine import FlowerClient
from dataset_loader import DatasetLoader
from fed.model import CNNClassifier
from fed.config_base import BaseConfig
from bft.client_malicious import MaliciousClient
from fed.train import ModelTrainer


class FederatedClient:
    def __init__(self, config: BaseConfig):
        self.config = config

    def create_client(self, context: Dict) -> FlowerClient:
        """Create and return a FlowerClient instance."""
        # Read the node_config to fetch data partition associated to this node
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        
        print("Number of partitions: ", num_partitions)

        model = CNNClassifier()
        trainer = ModelTrainer(model, self.config)
        
        datasetloader = DatasetLoader(config=self.config)
        trainloader, valloader, _ = datasetloader.load_datasets(partition_id)
        
        # TODO: Check use of validation loader
        flower_client = FlowerClient(partition_id, trainer, trainloader, valloader)
        
        print(f"ClientFactory created flower client for partition: {partition_id}")
        
        return flower_client.to_client()
    
    def create_malicious_client(self, partition_idx: int, model:CNNClassifier, trainer: ModelTrainer, trainloader: DataLoader, valloader: DataLoader,):
        attack_type = np.random.choice(self.config.attack_types)
        attack_intensity = np.random.choice(self.config.attack_intensities)
        client = MaliciousClient(
            partition_id=partition_idx,
            trainer=trainer,
            trainloader=trainloader,
            valloader=valloader,
            attack_type=attack_type,
            attack_intensity=attack_intensity
        )
        return client



if __name__ == "__main__":
    config = BaseConfig()
    fed_client = FederatedClient(config=config)
    client = ClientApp(client_fn=fed_client.create_client)
    print(f"ClientApp created : {client}")
