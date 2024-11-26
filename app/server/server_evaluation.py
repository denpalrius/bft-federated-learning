import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from flwr.common import Context
from app.nets.train import ModelTrainer
from app.utils.dataset_loader import CIFAR10DatasetLoader


class ServerEvaluation:
    """Handles centralized evaluation for the federated server."""

    def __init__(self, trainer: ModelTrainer, context: Context):
        self.trainer = trainer
        self.context = context
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_loader = CIFAR10DatasetLoader()
        self.global_test_set = self.dataset_loader.load_test_data()

    def gen_evaluate_fn(self):
        """Generate the centralized evaluation function."""

        def evaluate(server_round, parameters_ndarrays, config):
            """Evaluate the global model on the centralized CIFAR-10 test set."""

            # model = self.trainer.get_model()
            # model.to(self.device)

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
