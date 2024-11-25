import os
from datetime import datetime
from typing import Dict, Optional, Tuple
from flwr.common import NDArrays, Scalar
from dataset_loader import DatasetLoader
from model import CNNClassifier
from base_config import BaseConfig
from train import ModelTrainer

class ServerEvaluator:
    def __init__(self, base_config: BaseConfig, results_dir: str = "results"):
        self.base_config = base_config
        self.results_dir = results_dir        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = os.path.join(self.results_dir, f"evaluation_results_{timestamp}.txt")
        
        os.makedirs(self.results_dir, exist_ok=True)


    def evaluate(
        self,
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate the model after each federated round using test data.
        """

        dataset_loader = DatasetLoader(config=self.base_config)
        testloader = dataset_loader.load_test_set()

        trainer = ModelTrainer(CNNClassifier(), self.base_config)
        trainer.set_parameters(parameters)

        loss, accuracy = trainer.test(testloader)

        print(
            f"Server-side evaluation (Round {server_round}): loss {loss} / accuracy {accuracy}"
        )

        with open(self.results_file, "a") as f:
            f.write(
                f"Round {server_round}: Loss: {loss}, Accuracy: {accuracy}\n"
            )

        return loss, {"accuracy": accuracy}
