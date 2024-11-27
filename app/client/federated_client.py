from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from app.client.byzantine_attack import ByzantineStrategy, ByzantineAttack
from app.nets.train import ModelTrainer
from app.utils.logger import setup_logger


class FederatedClient(NumPyClient):
    """Federated Client that can be normal or malicious, based on provided attack type."""

    def __init__(
        self,
        partition_id: int,
        trainer: ModelTrainer,
        trainloader: DataLoader,
        valloader: DataLoader,
        attack_type: str = None,  # No attack by default
        attack_intensity: float = 1.0
    ):
        self.logger = setup_logger(self.__class__.__name__)

        self.partition_id = partition_id
        self.trainer = trainer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.trainloader = trainloader
        self.valloader = valloader
        self.attack_type = attack_type
        self.attack_intensity = attack_intensity

        # If attack type is provided, initialize PoisoningAttack class
        if self.attack_type:
            self.attack = ByzantineAttack(
                ByzantineStrategy[self.attack_type.upper()], self.attack_intensity
            )
        else:
            self.attack = None

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        self.logger.info(f"[Client {self.partition_id}] get_weights")
        return self.trainer.get_weights()

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model and return (poisoned) parameters."""
        self.logger.info(f"[Client {self.partition_id}] fit, config: {config}")

        self.trainer.update_weights(parameters)
        self.trainer.train(self.trainloader, epochs=2, device=self.device)

        updated_parameters = self.trainer.get_weights()

        if self.attack:
            self.logger.info(f"[Malicious Client {self.partition_id}] applying poison attack")
            updated_parameters = self.attack.poison_parameters(parameters)

        return (
            updated_parameters,
            len(self.trainloader),
            {"is_malicious": self.attack is not None},
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate the model and return metrics."""
        self.logger.info(f"[Client {self.partition_id}] evaluate, config: {config}")

        self.trainer.update_weights(parameters)
        loss, accuracy = self.trainer.test(self.valloader, self.device)

        return (
            float(loss),
            len(self.valloader),
            {"accuracy": float(accuracy), "is_malicious": self.attack is not None},
        )
