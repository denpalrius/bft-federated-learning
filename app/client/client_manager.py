import os
import torch
import numpy as np
from typing import Dict, Any
from flwr.client import Client
from flwr.common import Context
from app.client.federated_client import FederatedClient
from app.nets.cnn_classifier import CNNClassifier
from app.nets.train import ModelTrainer
from app.utils import dataset_loader
from app.utils.logger import setup_logger
from app.client.byzantine_attack import ByzantineStrategy


class ClientManager:
    """
    Client management for federated learning with
    Byzantine Fault Tolerance (BFT) support.
    """

    def __init__(self, context: Context):
        self.logger = setup_logger(self.__class__.__name__)
        self._initialize_context(context)
        self._initialize_trainer()
        self._initialize_data_loaders()
        self.initialize_malicious_client()
        self._initialize_metrics()

    def _initialize_context(self, context: Context):
        self.node_id = context.node_id
        self.run_config = context.run_config
        self.node_config = context.node_config
        self.partition_id = self.node_config.get("partition-id")
        self.num_partitions = self.node_config.get("num-partitions")

    def _initialize_trainer(self):
        pretrained_model_path = self.run_config.get("pretrained-model-path")
        pretrained_model_path = os.path.join(os.getcwd(), pretrained_model_path)
        self.trainer = ModelTrainer(CNNClassifier(), pretrained_model_path)
        self.local_epochs = self.run_config.get("local-epochs")

    def initialize_malicious_client(self):
        if self.num_partitions is None:
            raise ValueError(
                "Number of partitions must be set before initializing malicious clients"
            )

        byzantine_clients = self.run_config.get("byzantine-clients", 0)
        if byzantine_clients > 0:
            # Ensure total number of clients satisfy the condition N > 2f for Krum
            if self.num_partitions <= 2 * byzantine_clients:
                raise RuntimeError(
                    "Number of partitions must be greater than 2 times the number of Byzantine clients for Krum"
                )

            self.malicious_indices = np.random.choice(
                self.num_partitions, size=byzantine_clients, replace=False
            )
            self.logger.info(f"Malicious indices: {self.malicious_indices}")

            # Initialize attack strategy
            attack_strategy = self.run_config.get(
                "byzantine-attack-strategy", "sign_flip"
            )
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
        else:
            self.malicious_indices = []

    def _initialize_data_loaders(self):
        self.logger.info("===============================")
        self.logger.info(f"Partition {self.partition_id}/{self.num_partitions} loaded")
        self.logger.info("===============================")

        self.batch_size = self.run_config.get("batch-size")
        self.trainloader, self.valloader = dataset_loader.load_data_partition(
            self.num_partitions, self.partition_id, self.batch_size
        )

    def _initialize_metrics(self):
        self.client_metrics = {
            "total_rounds": 0,
            "train_losses": [],
            "eval_accuracies": [],
            "byzantine_interventions": 0,
        }

    def create_client(self) -> Client:
        is_malicious = self.partition_id in self.malicious_indices
        attack_type = self.attack_strategy if is_malicious else None
        attack_intensity = self.attack_intensity if self.attack_intensity  else 1.0

        client = FederatedClient(
            partition_id=self.partition_id,
            trainer=self.trainer,
            trainloader=self.trainloader,
            valloader=self.valloader,
            local_epochs=self.local_epochs,
            attack_type=attack_type,
            attack_intensity=attack_intensity,
        )

        return client.to_client()
        # return self._wrap_client_with_metrics(client).to_client()

    def _wrap_client_with_metrics(self, client: Client) -> Client:
        original_fit = client.fit
        original_evaluate = client.evaluate

        def fit_wrapper(*args, **kwargs):
            result = original_fit(*args, **kwargs)
            metrics = result[2]

            print("=========Metrics=========")
            # print("result: ", result)
            print("Metrics: ", metrics)

            if isinstance(metrics, dict):
                self.log_client_metrics({"train_loss": metrics.get("train_loss", None)})
            else:
                self.log_client_metrics({"train_loss": metrics})
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

        self.logger.info(f"Client {self.partition_id} metrics: {metrics}")
        if isinstance(metrics, dict):
            if "train_loss" in metrics:
                self.client_metrics["train_losses"].append(metrics["train_loss"])
            if "accuracy" in metrics:
                self.client_metrics["eval_accuracies"].append(metrics["accuracy"])
        elif isinstance(metrics, float):
            self.client_metrics["train_losses"].append(metrics)

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
