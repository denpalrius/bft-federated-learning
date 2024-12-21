import os
from datetime import datetime
import json
import torch
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import time
from pathlib import Path

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from app.nets.train import ModelTrainer
from app.utils.logger import setup_logger


class BFTFedAvg(FedAvg):
    """
    A custom Byzantine Fault Tolerant (BFT) Strategy for Federated Learning

    Key BFT Mechanisms:
    1. Krum Aggregation: Aggregates client updates by selecting the most consistent updates, protecting against Byzantine clients.
    2. Suspicious Update Detection: Identifies clients with anomalous updates based on z-score, NaN, or infinite values.
    3. Performance Logging: Tracks and logs metrics such as suspected Byzantine clients, aggregation times, and anomalies.
    4. Model Checkpointing: Saves model checkpoints when the best accuracy is achieved.
    5. Result Storage: Stores aggregation results and metrics in JSON files for each round.
    """

    def __init__(
        self,
        *args,
        trainer: ModelTrainer,
        byzantine_threshold: float = 0.7,
        max_deviation_threshold: float = 2.0,
        byzantine_clients = 0,
        **kwargs,
    ):
        """
        Initialize BFT Federated Averaging Strategy

        :param byzantine_threshold: Fraction of potential Byzantine clients
        :param max_deviation_threshold: Maximum allowed standard deviation for update detection
        """
        super().__init__(*args, **kwargs)
        self.logger = setup_logger(self.__class__.__name__)

        self.trainer = trainer
        self.byzantine_threshold = byzantine_threshold
        self.max_deviation_threshold = max_deviation_threshold

        # Performance and fault tracking
        self.performance_logs = {
            "rounds": [],
            "total_clients": [],
            "suspected_byzantine_clients": [],
            "aggregation_times": [],
            "detected_anomalies": [],
        }

        self.logger.info("BFTFedAvg strategy initialized")
        self.logger.info(f"Byzantine Threshold: {byzantine_threshold}")
        self.logger.info(f"Max Deviation Threshold: {max_deviation_threshold}")

        # Initialize path to save results and models
        self.save_path = Path("results")
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.best_acc_so_far = 0.0

    def _detect_suspicious_updates(
        self, updates: List[Tuple[List[np.ndarray], int]]
    ) -> List[int]:
        """
        Detect suspicious client updates based on statistical analysis

        :param updates: List of client weight updates
        :return: List of indices of suspicious clients
        """
        suspicious_indices = []

        # Flatten and analyze updates
        flat_updates = [
            np.concatenate([np.ravel(layer) for layer in update[0]])
            for update in updates
        ]

        # Calculate overall statistics
        update_norms = [np.linalg.norm(update) for update in flat_updates]
        mean_norm = np.mean(update_norms)
        std_norm = np.std(update_norms)

        for i, (norm, update) in enumerate(zip(update_norms, flat_updates)):
            # Detect outliers using z-score
            z_score = abs((norm - mean_norm) / std_norm) if std_norm != 0 else 0

            # Check for extreme deviations
            if (
                z_score > self.max_deviation_threshold
                or np.any(np.isnan(update))
                or np.any(np.isinf(update))
            ):

                suspicious_indices.append(i)
                self.performance_logs["detected_anomalies"].append(
                    {
                        "client_index": i,
                        "norm": norm,
                        "z_score": z_score,
                        "reasons": [
                            (
                                "High z-score"
                                if z_score > self.max_deviation_threshold
                                else ""
                            ),
                            "Contains NaN" if np.any(np.isnan(update)) else "",
                            "Contains Inf" if np.any(np.isinf(update)) else "",
                        ],
                    }
                )

                self.logger.warning(f"Suspicious client detected (Index {i})")

        return suspicious_indices

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results using Krum method with fault detection.

        :param server_round: Current server round.
        :param results: Client training results.
        :param failures: Failed client results.
        :return: Aggregated parameters and metrics.
        """
        start_time = time.time()

        if not results:
            self.logger.warning(f"No results received for round {server_round}.")
            return None, {}

        # Convert results to list of weights and sample sizes
        updates = [
            (parameters_to_ndarrays(res.parameters), res.num_examples)
            for _, res in results
        ]

        # Detect and remove suspicious clients
        num_byzantine = int(self.byzantine_threshold * len(updates))
        suspicious_indices = self._detect_suspicious_updates(updates)

        # Remove suspicious updates
        updates = [
            update for i, update in enumerate(updates) if i not in suspicious_indices
        ]

        try:
            # Perform Krum aggregation
            aggregated_weights = self._krum_aggregate(updates, num_byzantine)

            # Convert aggregated weights to Parameters
            aggregated_parameters = ndarrays_to_parameters(aggregated_weights)

            # Record metrics and logs in a single structure
            end_time = time.time()
            aggregation_time = end_time - start_time

            aggregation_summary = {
                "round": server_round,
                "total_clients": len(results),
                "suspicious_clients": len(suspicious_indices),
                "aggregation_time": aggregation_time,
                "failures": len(failures),
            }

            self.performance_logs.append(aggregation_summary)

            # Save results to the filesystem
            self._store_results(server_round, aggregation_summary, prefix="aggregation")

            # Save model checkpoint if best accuracy is achieved
            self._save_model_checkpoint(aggregated_parameters, server_round)

            # Log aggregation summary
            self.logger.info(f"Round {server_round} Aggregation Summary:")
            for key, value in aggregation_summary.items():
                self.logger.info(f"{key.capitalize()}: {value}")

            return aggregated_parameters, aggregation_summary

        except Exception as e:
            self.logger.error(f"Aggregation failed in round {server_round}: {e}")
            return None, {}

    def _krum_aggregate(
        self, updates: List[Tuple[List[np.ndarray], int]], num_byzantine: int
    ) -> List[np.ndarray]:
        """
        Krum aggregation method with robust error handling

        :param updates: List of client weight updates
        :param num_byzantine: Number of expected Byzantine clients
        :return: Aggregated weights
        """
        n = len(updates)
        f = num_byzantine
        m = n - f - 2  # Number of neighbors to consider

        if m < 1:
            raise ValueError("Not enough non-Byzantine clients for Krum aggregation")

        #  TODO: Checl with the Flower implementation

        # Flatten updates for distance calculation
        flattened_updates = [
            np.concatenate([np.ravel(layer) for layer in update[0]])
            for update in updates
        ]

        # Calculate pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distances[i][j] = distances[j][i] = np.linalg.norm(
                    flattened_updates[i] - flattened_updates[j]
                )

        # Compute Krum scores
        scores = []
        for i in range(n):
            sorted_distances = np.sort(distances[i])
            scores.append(np.sum(sorted_distances[:m]))

        # Select update with minimum score (most consistent)
        selected_index = np.argmin(scores)

        # Log details of the selected client
        self.logger.info(f"Selected client index: {selected_index}")
        self.logger.info(f"Krum score: {scores[selected_index]}")

        return updates[selected_index][0]

    def _store_results(self, round: int, metrics: Dict[str, Scalar], prefix=""):
        """Store results in a JSON file"""
        results = {"round": round, "metrics": metrics}

        # Generate a timestamp-based filename to avoid overwriting
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"results_{prefix}_{timestamp}.json"
        file_path = os.path.join(self.save_path, file_name)
        os.makedirs(self.save_path, exist_ok=True)

        # Write the results to the JSON file
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)

        self.logger.info(f"Results stored in {file_path}")

    def _update_best_acc(self, round: int, accuracy: float, parameters: Parameters):
        """Determines if a new best global model has been found and save to disk."""
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            self.logger.info(f"💡 New best global model found: {accuracy:.4f}")

            # Convert FL parameters to PyTorch model weights
            ndarrays = parameters_to_ndarrays(parameters)

            self.trainer.update_weights(ndarrays)
            model = self.trainer.get_model()

            model_path = self._generate_checkpoint_path(
                round, accuracy, prefix="model_state"
            )

            os.makedirs(self.save_path, exist_ok=True)

            # Save the model state_dict
            torch.save(model.state_dict(), model_path)
            self.logger.info(f"Best model saved at {model_path}")

    def _save_model_checkpoint(self, parameters: Parameters, round: int):
        """Save model checkpoint if a new best accuracy is achieved."""
        accuracy = parameters["accuracy"]  # Assuming accuracy is part of the parameters
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy

            model_path = self._generate_checkpoint_path(round, accuracy)
            os.makedirs(self.save_path, exist_ok=True)

            # Save the model parameters as a checkpoint
            torch.save(parameters, model_path)
            self.logger.info(f"Model checkpoint saved at {model_path}")

    def _generate_checkpoint_path(
        self, round: int, accuracy: float, prefix: str = "model_checkpoint"
    ) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{prefix}_round_{round}_accuracy_{accuracy:.4f}_{timestamp}.pth"
        return os.path.join(self.save_path, file_name)

    def evaluate(self, server_round, parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        try:
            loss, metrics = super().evaluate(server_round, parameters)

            centralized_accuracy = metrics.get("centralized_accuracy", 0)
            self._update_best_acc(server_round, centralized_accuracy, parameters)
            self._store_results(
                round=server_round,
                metrics={
                    "centralized_loss": loss,
                    "centralized_accuracy": centralized_accuracy,
                    **metrics,
                },
                prefix="centralized",
            )

            self._print_performance_summary()

            return loss, metrics
        except Exception as e:
            self.logger.error(f"Error in evaluate at round {server_round}: {str(e)}")
            raise

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        try:
            # Perform the federated evaluation aggregation
            loss, metrics = super().aggregate_evaluate(server_round, results, failures)

            federated_accuracy = metrics.get("federated_accuracy", 0)

            self._store_results(
                round=server_round,
                metrics={
                    "federated_loss": loss,
                    "federated_accuracy": federated_accuracy,
                    **metrics,
                },
                prefix="federated",
            )

            self._print_performance_summary()

            return loss, metrics
        except Exception as e:
            self.logger.error(
                f"Error in aggregate_evaluate at round {server_round}: {str(e)}"
            )
            raise

    def _print_performance_summary(self):
        """
        Print a comprehensive summary of the performance logs
        """
        self.logger.info("========== Aggregate results from federated evaluation ==========")
        self.logger.info(f"Total Rounds: {len(self.performance_logs['rounds'])}")
        self.logger.info(
            f"Average Clients per Round: {np.mean(self.performance_logs['total_clients']):.2f}"
        )
        self.logger.info(
            f"Total Suspected Byzantine Clients: {sum(self.performance_logs['suspected_byzantine_clients'])}"
        )
        self.logger.info(
            f"Total Anomalies Detected: {len(self.performance_logs['detected_anomalies'])}"
        )
