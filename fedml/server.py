import fedml
from fedml.core.alg_frame.server_aggregator import FedAvgAggregator
import json
import grpc
import requests

TENDERMINT_URL = "http://localhost:26657"

class MnistServer(fedml.Server):
    def __init__(self, args):
        super().__init__(args)
        self.aggregator = FedAvgAggregator()

    def submit_to_tendermint(self, update):
        """Submit a model update to Tendermint for validation."""
        tx_data = json.dumps(update).encode()
        response = requests.post(f"{TENDERMINT_URL}/broadcast_tx_commit", data={"tx": tx_data})
        if response.status_code == 200:
            print("Transaction committed:", response.json())
        else:
            print("Tendermint error:", response.text)

    def aggregate(self, weights_list):
        """Aggregate validated updates."""
        validated_updates = []
        for update in weights_list:
            self.submit_to_tendermint(update)
            # Simulate validation by Tendermint (mock for demo)
            validated_updates.append(update)  # Assume all updates are validated
        aggregated_weights = self.aggregator.aggregate(validated_updates)
        return aggregated_weights

    def set_model(self, model):
        """Set the model for aggregation."""
        self.model = model

if __name__ == "__main__":
    fedml.run_server(MnistServer)
