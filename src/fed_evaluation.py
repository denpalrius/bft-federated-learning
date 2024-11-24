from typing import Dict, Optional, Tuple
from flwr.common import NDArrays, Scalar
from dataloader import DatasetLoader
from model import CNNClassifier
from config import BaseConfig
from train import ModelTrainer


class ServerEvaluator:
    def __init__(self, base_config: BaseConfig):
        self.base_config = base_config

    def evaluate(
        self,
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate the model after each federated round using test data.
        """

        print(f'Base config: {self.base_config}')
        
        trainer = ModelTrainer(CNNClassifier(), self.base_config)
        trainer.to(self.base_config.device)

        dataset_loader = DatasetLoader(config=self.base_config)
        testloader = dataset_loader.load_test_set()

        trainer.set_parameters(parameters)

        loss, accuracy = trainer.test(testloader)

        print(
            f"Server-side evaluation (Round {server_round}): loss {loss} / accuracy {accuracy}"
        )

        # Return the loss and accuracy for Flower's federated learning process
        return loss, {"accuracy": accuracy}


if __name__ == "__main__":
    config = BaseConfig()
