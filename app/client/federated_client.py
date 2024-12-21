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
        local_epochs: int,
        attack_type: str = None,
        attack_intensity: float = 1.0,
    ):
        self.logger = setup_logger(self.__class__.__name__)

        self.partition_id = partition_id
        self.trainer = trainer
        self.local_epochs = local_epochs

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
        # self.logger.info(f"[Client {self.partition_id}] get_weights")
        return self.trainer.get_weights()

    def fit(self, parameters, config):
        self.trainer.update_weights(parameters)
        train_loss = self.trainer.train(
            self.trainloader,
            self.local_epochs        )
        return (
            self.trainer.get_weights(),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        self.trainer.update_weights(parameters)
        loss, accuracy = self.trainer.test(self.valloader)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


    def fit2(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model and return (poisoned) parameters."""
        self.logger.info(f"[Client {self.partition_id}] fit, config: {config}")

        self.trainer.update_weights(parameters)
        results = self.trainer.train(self.trainloader, epochs=self.num_epochs)

        updated_parameters = self.trainer.get_weights()

        if self.attack:
            self.logger.info(
                f"[Malicious Client {self.partition_id}] applying poison attack"
            )
            updated_parameters = self.attack.poison_parameters(parameters)

        return (
            updated_parameters,
            len(self.trainloader),
            results,
            {"is_malicious": self.attack is not None},
        )

    # def evaluate(
    #     self, parameters: List[np.ndarray], config: Dict
    # ) -> Tuple[float, int, Dict]:
    #     """Evaluate the model and return metrics."""
    #     self.logger.info(f"[Client {self.partition_id}] evaluate, config: {config}")

    #     # TODO: What if poison attack is applied to evaluation?
    #     self.trainer.update_weights(parameters)
    #     loss, accuracy = self.trainer.test(self.valloader)

    #     return (
    #         float(loss),
    #         len(self.valloader),
    #         {"accuracy": float(accuracy), "is_malicious": self.attack is not None},
    #     )
    #     return loss, len(testloader.dataset), {"accuracy": accuracy}
