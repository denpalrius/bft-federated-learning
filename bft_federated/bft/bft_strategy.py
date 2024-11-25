from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class BFTFedAvg(FedAvg):
    """Byzantine Fault Tolerant Federated Averaging Strategy."""

    def __init__(
        self,
        *args,
        threshold: float = 0.5,  # Threshold for Krum/Trimmed Mean
        bft_method: str = "krum",  # Options: "krum", "trimmed_mean", "median"
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.bft_method = bft_method
        self.round_metrics: List[Dict] = []

    def _krum_aggregate(
        self, updates: List[Tuple[List[np.ndarray], int]], num_Byzantine: int
    ) -> List[np.ndarray]:
        """Krum aggregation method."""
        n = len(updates)
        f = num_Byzantine
        m = n - f - 2  # Number of neighbors to consider

        if m < 1:
            raise ValueError("Not enough clients for Krum aggregation")

        # Calculate pairwise distances between updates
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = sum(
                    np.linalg.norm(u1 - u2)
                    for u1, u2 in zip(updates[i][0], updates[j][0])
                )
                distances[i][j] = distances[j][i] = dist

        # For each update, sum its distances to closest m others
        scores = []
        for i in range(n):
            sorted_distances = np.sort(distances[i])
            scores.append(np.sum(sorted_distances[:m]))

        # Select update with minimum score
        selected_idx = np.argmin(scores)
        return updates[selected_idx][0]
    
        # Pairwise distances
        # distances = np.array([
        #     [sum(np.linalg.norm(a - b) for a, b in zip(updates[i][0], updates[j][0]))
        #      for j in range(n)]
        #     for i in range(n)
        # ])
        
        # # Scores
        # scores = [np.sum(np.sort(row)[:n - f - 2]) for row in distances]
        # return updates[np.argmin(scores)][0]

    def _trimmed_mean_aggregate(
        self, updates: List[Tuple[List[np.ndarray], int]], beta: float
    ) -> List[np.ndarray]:
        """Trimmed mean aggregation method."""
        all_updates = [update[0] for update in updates]
        n_updates = len(all_updates)
        n_trim = int(beta * n_updates)

        aggregated = []
        for param_idx in range(len(all_updates[0])):
            # Stack same parameter from all updates
            stacked = np.stack([update[param_idx] for update in all_updates])
            # Sort values
            sorted_updates = np.sort(stacked, axis=0)
            # Trim beta fraction from both ends
            trimmed = sorted_updates[n_trim:-n_trim] if n_trim > 0 else sorted_updates
            # Compute mean of remaining values
            aggregated.append(np.mean(trimmed, axis=0))

        return aggregated

    def _median_aggregate(
        self, updates: List[Tuple[List[np.ndarray], int]]
    ) -> List[np.ndarray]:
        """Coordinate-wise median aggregation."""
        all_updates = [update[0] for update in updates]
        aggregated = []

        for param_idx in range(len(all_updates[0])):
            stacked = np.stack([update[param_idx] for update in all_updates])
            aggregated.append(np.median(stacked, axis=0))

        return aggregated

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates using BFT methods."""
        if not results:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Estimate number of Byzantine clients
        n_clients = len(weights_results)
        n_byzantine = int(n_clients * (1 - self.threshold))

        # Aggregate based on selected method
        if self.bft_method == "krum":
            aggregated_weights = self._krum_aggregate(weights_results, n_byzantine)
        elif self.bft_method == "trimmed_mean":
            aggregated_weights = self._trimmed_mean_aggregate(
                weights_results, beta=(1 - self.threshold) / 2
            )
        else:  # median
            aggregated_weights = self._median_aggregate(weights_results)

        # Save metrics
        round_metrics = {
            "round": server_round,
            "n_clients": n_clients,
            "n_byzantine": n_byzantine,
            "aggregation_method": self.bft_method,
        }
        self.round_metrics.append(round_metrics)

        return ndarrays_to_parameters(aggregated_weights), {
            "n_clients": n_clients,
            "n_byzantine": n_byzantine,
        }
