from flwr.server import ServerApp
from flwr.client import ClientApp
from flwr.simulation import run_simulation
from base_config import BaseConfig


class SimulationRunner:
    """Orchestrates the federated learning simulation."""

    def __init__(
        self,
        client_app : ClientApp,
        server_app: ServerApp,
        config: BaseConfig,
    ):
        self.client_app = client_app
        self.server_app = server_app
        self.config = config

    def run(self):
        print("Running federated learning simulation...")

        run_simulation(
            server_app=self.server_app,
            client_app=self.client_app,
            num_supernodes=self.config.num_partitions,
            backend_config=self.config.backend_config,
        )
