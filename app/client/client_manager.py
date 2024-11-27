import torch
import numpy as np
from typing import Dict, Any
from flwr.client import Client
from flwr.common import Context
from app.client.federated_client import FederatedClient
from app.nets.model import CNNClassifier
from app.nets.train import ModelTrainer
from app.utils.dataset_loader import CIFAR10DatasetLoader
from app.utils.logger import setup_logger
from app.client.byzantine_attack import ByzantineStrategy


class ClientManager:
    """
    Client management for federated learning with
    Byzantine Fault Tolerance (BFT) support.
    """

    def __init__(self, context: Context = None):
        self.logger = setup_logger(self.__class__.__name__)

        self.trainer = ModelTrainer(CNNClassifier())
        self.run_config = context.run_config

        attack_strategy = self.run_config.get("byzantine-attack-strategy", "sign_flip")
        self.attack_strategy = ByzantineStrategy[attack_strategy.upper()]
        self.attack_intensity = float(
            self.run_config.get("byzantine-attack-intensity", 1.0)
        )

        randomize_byzantine_strategy = self.run_config.get(
            "randomize-byzantine-strategy", False
        )
        if randomize_byzantine_strategy:
            attack_strategies = [strategy.name for strategy in ByzantineStrategy]
            self.attack_strategy = np.random.choice(attack_strategies)
            self.attack_intensity = np.random.choice([0.1, 0.5, 1.0, 2.0])
        self.logger.info("Byzantine attack strategy: %s", self.attack_strategy)
        self.logger.info("Byzantine attack intensity: %s", self.attack_intensity)

        # Ensuring total number of clients satisfy the condition 𝑁 > 3𝑓
        byzantine_clients = self.run_config.get("byzantine-clients", 5)
        num_clients = self.run_config.get("num-clients", 16)
        num_clients = max(num_clients, 3 * byzantine_clients + 1)

        self.malicious_indices = np.random.choice(
            num_clients, size=byzantine_clients, replace=False
        )
        self.logger.info(f"Malicious indices: {self.malicious_indices}")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # TODO: Get partition ID from the client count
        self.partition_id = self.run_config.get("partition-id", 0)
        self.batch_size = self.run_config.get("batch-size", 32)
        self.num_partitions = self.run_config.get("num-partitions", 10)

        self.dataset_loader = CIFAR10DatasetLoader(
            num_partitions=self.num_partitions, batch_size=self.batch_size
        )
        self.dataset_loader.initialize_fds()

        self.trainloader, self.valloader = self.dataset_loader.load_data_partition(
            self.partition_id
        )
        
        # Metrics tracking
        self.client_metrics = {
            "total_rounds": 0,
            "train_losses": [],
            "eval_accuracies": [],
            "byzantine_interventions": 0,
        }

    def create_client(self) -> Client:
        # Determine if the client is malicious
        is_malicious = self.partition_id in self.malicious_indices
        attack_type = self.attack_strategy if is_malicious else None

        client = FederatedClient(
            partition_id=self.partition_id,
            trainer=self.trainer,
            trainloader=self.trainloader,
            valloader=self.valloader,
            attack_type=attack_type,
            attack_intensity=self.attack_intensity,
        )

        # Wrap the client with a hook for logging metrics
        return self._wrap_client_with_metrics(client).to_client()


    def _wrap_client_with_metrics(self, client: Client) -> Client:
        original_fit = client.fit
        original_evaluate = client.evaluate

        def fit_wrapper(*args, **kwargs):
            result = original_fit(*args, **kwargs)
            metrics = result[2]
            self.log_client_metrics({"train_loss": metrics.get("train_loss", None)})
            return result

        def evaluate_wrapper(*args, **kwargs):
            result = original_evaluate(*args, **kwargs)
            metrics = result[2]
            self.log_client_metrics({"accuracy": metrics.get("accuracy", None)})
            return result

        client.fit = fit_wrapper
        client.evaluate = evaluate_wrapper

        return client

    def log_client_metrics(self, metrics: Dict[str, Any]):
        self.client_metrics["total_rounds"] += 1
        
        self.logger.info(
            f"Client {self.partition_id} metrics: {metrics}"
        )

        if "train_loss" in metrics:
            self.client_metrics["train_losses"].append(metrics["train_loss"])

        if "accuracy" in metrics:
            self.client_metrics["eval_accuracies"].append(metrics["accuracy"])

    def print_client_summary(self):
        print("\n--- Client Performance Summary ---")
        print(f"Total Training Rounds: {self.client_metrics['total_rounds']}")
        print(
            f"Average Training Loss: {self._safe_mean(self.client_metrics['train_losses']):.4f}"
        )
        print(
            f"Average Evaluation Accuracy: {self._safe_mean(self.client_metrics['eval_accuracies']):.4f}%"
        )
        print(f"Malicious Client: {self.partition_id in self.malicious_indices}")
        print(f"Byzantine Attack Type: {self.attack_strategy or 'None'}")

    def _safe_mean(self, values):
        return sum(values) / len(values) if values else 0
