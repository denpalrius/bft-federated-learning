import torch
from flwr.common.typing import UserConfig
from app.nets.train import ModelTrainer
from app.utils.dataset_loader import CIFAR10DatasetLoader

from app.utils.logger import setup_logger

class ServerEvaluation:
    """Handles centralized evaluation for the federated server."""

    def __init__(self, trainer: ModelTrainer, run_config: UserConfig):
        self.logger = setup_logger(self.__class__.__name__)

        self.trainer = trainer
        self.run_config = run_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.num_partitions = run_config.get("num-partitions")
        self.batch_size = run_config.get("batch-size")
        
        self.dataset_loader = CIFAR10DatasetLoader(num_partitions=self.num_partitions, batch_size=self.batch_size)
        self.dataset_loader.initialize_fds()
        
        self.global_test_set = self.dataset_loader.load_test_data()


    def gen_evaluate_fn(self):
        """Generate the centralized evaluation function."""
        def evaluate(server_round, parameters_ndarrays, config):
            self.trainer.update_weights(parameters_ndarrays)
            loss, accuracy = self.trainer.test(self.global_test_set, self.device)

            return loss, {"centralized_accuracy": accuracy}

        return evaluate

    @staticmethod
    def weighted_average(metrics):
        """Aggregate metrics using a weighted average."""
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}
