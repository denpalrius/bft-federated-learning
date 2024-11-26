import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate_krum, aggregate_median


class BFTBasicFedAvg(FedAvg):
    """Byzantine Fault Tolerant Federated Averaging Strategy."""

    def __init__(
        self,
        *args,
        threshold: float = 0.5,  # Fraction of Byzantine clients
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.threshold = threshold  # Fraction of Byzantine clients
        self.round_metrics: List[Dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results using Krum or Trimmed Mean aggregation."""
        if not results:
            return None, {}

        # Print failure details if any failures occur
        if failures:
            print(f"Failures: {failures}")

        # Convert results to list of weights and sample sizes
        updates = [
            (parameters_to_ndarrays(res.parameters), res.num_examples)
            for _, res in results
        ]

        # Calculate number of Byzantine clients
        num_Byzantine = int(self.threshold * len(updates))

        # Try to perform Krum aggregation, fallback to trimmed mean if necessary
        try:
            aggregated_weights = self._krum_aggregate(updates, num_Byzantine)
        except ValueError:
            print(f"Krum aggregation not possible, using Trimmed Mean instead.")
            aggregated_weights = aggregate_median(updates)

        # Convert aggregated weights to Parameters
        aggregated_parameters = ndarrays_to_parameters(aggregated_weights)
        metrics = {"round": server_round, "num_Byzantine": num_Byzantine}

        return aggregated_parameters, metrics

    def _krum_aggregate(
        self, updates: List[Tuple[List[np.ndarray], int]], num_Byzantine: int
    ) -> List[np.ndarray]:
        """Krum aggregation method."""
        n = len(updates)
        f = num_Byzantine
        m = n - f - 2  # Number of neighbors to consider

        if m < 1:
            raise ValueError("Not enough clients for Krum aggregation")

        return aggregate_krum(updates, f, to_keep=2)

