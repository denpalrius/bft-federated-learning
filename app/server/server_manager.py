from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from bft_flower_app.task import Net, get_weights


class ServerManager:
    """Manages the configuration and creation of the S erver."""

    def __init__(self, context: Context):
        self.context = context

    def get_strategy(self):
        fraction_fit = self.context.run_config["fraction-fit"]
        ndarrays = get_weights(Net())
        parameters = ndarrays_to_parameters(ndarrays)

        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
        )
        return strategy

    def get_config(self):
        """Define and return the server configuration."""
        num_rounds = self.context.run_config["num-server-rounds"]
        return ServerConfig(num_rounds=num_rounds)

    def create_components(self):
        """Create and return the server app components."""
        strategy = self.get_strategy()
        config = self.get_config()
        return ServerAppComponents(strategy=strategy, config=config)
