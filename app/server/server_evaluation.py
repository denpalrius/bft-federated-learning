import torch
from flwr.common.typing import UserConfig
from app.nets.train import ModelTrainer
from app.utils import dataset_loader

from app.utils.logger import setup_logger


class ServerEvaluation:
    """Handles centralized evaluation for the federated server."""

    def __init__(self, trainer: ModelTrainer, batch_size):
        self.logger = setup_logger(self.__class__.__name__)

        self.trainer = trainer        
        self.global_test_set  = dataset_loader.load_test_data(batch_size)
        
        self.logger.debug(f"Global Test Set: {self.global_test_set}")

    def gen_evaluate_fn(self):
        """Generate the centralized evaluation function."""

        def evaluate(server_round, parameters_ndarrays, config):
            self.trainer.update_weights(parameters_ndarrays)
            loss, accuracy = self.trainer.test(self.global_test_set)

            return loss, {"centralized_accuracy": accuracy}

        return evaluate

    @staticmethod
    def weighted_average(metrics):
        """Aggregate metrics using a weighted average."""
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}
