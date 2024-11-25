from typing import Dict, List, Tuple
import numpy as np
from torch.utils.data import DataLoader
from flwr.client import ClientApp, NumPyClient
from dataset_loader import DatasetLoader
from fed.model import CNNClassifier
from fed.config_base import BaseConfig
from fed.train import ModelTrainer



class GenuineClient(NumPyClient):
    """Flower client for genuine participants in federated learning."""

    def __init__(self, partition_id: int, trainer: ModelTrainer, trainloader: DataLoader, valloader: DataLoader):
        self.partition_id = partition_id
        self.trainer = trainer
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return the model parameters."""
        print(f"[Client {self.partition_id}] get_parameters")
        return self.trainer.get_parameters()

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model and return updated parameters."""
        print(f"[Client {self.partition_id}] fit, config: {config}")
        self.trainer.set_parameters(parameters)
        self.trainer.train(self.trainloader, epochs=1)
        return self.trainer.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the model and return metrics."""
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        self.trainer.set_parameters(parameters)
        loss, accuracy = self.trainer.test(self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


# class FederatedClient:
#     def __init__(self, config: BaseConfig):
#         self.config = config

#     def create_client(self, context: Dict) -> FlowerClient:
#         """Create and return a FlowerClient instance."""
#         # Read the node_config to fetch data partition associated to this node
#         partition_id = context.node_config["partition-id"]
#         num_partitions = context.node_config["num-partitions"]
        
#         print("Number of partitions: ", num_partitions)

#         model = CNNClassifier()
#         trainer = ModelTrainer(model, self.config)
        
#         datasetloader = DatasetLoader(config=self.config)
#         trainloader, valloader, _ = datasetloader.load_datasets(partition_id)
        
#         # TODO: Check use of validation loader
#         flower_client = FlowerClient(partition_id, trainer, trainloader, valloader)
        
#         print(f"ClientFactory created flower client for partition: {partition_id}")
        
#         return flower_client.to_client()



# if __name__ == "__main__":
#     config = BaseConfig()
#     fed_client = FederatedClient(config=config)
#     client = ClientApp(client_fn=fed_client.create_client)
#     print(f"ClientApp created : {client}")
