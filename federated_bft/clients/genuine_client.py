from utils.path import add_base_path
add_base_path(__file__)

from typing import Dict, List, Tuple
import numpy as np
from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from nets.train import ModelTrainer


class GenuineClient(NumPyClient):
    """Flower client for genuine participants in federated learning."""

    def __init__(
        self,
        partition_id: int,
        trainer: ModelTrainer,
        trainloader: DataLoader,
        valloader: DataLoader,
    ):
        self.partition_id = partition_id
        self.trainer = trainer
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return the model parameters."""
        print(f"[Client {self.partition_id}] get_parameters")
        return self.trainer.get_parameters()

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model and return updated parameters."""
        print(f"[Client {self.partition_id}] fit, config: {config}")
        self.trainer.set_parameters(parameters)
        self.trainer.train(self.trainloader, epochs=1)
        return self.trainer.get_parameters(), len(self.trainloader), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate the model and return metrics."""
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        self.trainer.set_parameters(parameters)
        loss, accuracy = self.trainer.test(self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
