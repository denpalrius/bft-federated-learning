from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from app.task import Net, get_weights


class ServerManager:
    """Manages the configuration and creation of the S erver."""

    def __init__(self, context: Context):
        self.context = context
        
    def create_strategy(self):
        # TODO: Use BFT strategy
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
        num_rounds = self.context.run_config["num-server-rounds"]
        return ServerConfig(num_rounds=num_rounds)

    def create_components(self):
        strategy = self.create_strategy()
        config = self.get_config()
        return ServerAppComponents(strategy=strategy, config=config)
    