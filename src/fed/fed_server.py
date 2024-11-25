from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import ndarrays_to_parameters, Parameters
from flwr.server.strategy import Strategy
from flwr.common import Context

from config_base import BaseConfig
from model import CNNClassifier
from train import ModelTrainer
from fed_strategy import FederatedStrategy, StrategyConfig


class FederatedServer:
    """Handles the configuration and execution of the federated server."""

    def __init__(
        self,
        strategy: Strategy,
        num_rounds: int,
        initial_parameters: Parameters,
    ):
        self.strategy = strategy
        self.num_rounds = num_rounds
        self.initial_parameters = initial_parameters

    def create_server_components(self, context: Context) -> ServerAppComponents:        
        server_config = ServerConfig(num_rounds=self.num_rounds)
        return ServerAppComponents(strategy=self.strategy, config=server_config)

    def create_server_app(self) -> ServerApp:
        server = ServerApp(server_fn=self.create_server_components)
        return server


if __name__ == "__main__":
    config = BaseConfig()
    model = CNNClassifier()
    trainer = ModelTrainer(model, config)

    # Strategy configuration
    params = trainer.get_parameters()
    strategy_config = StrategyConfig(
        initial_parameters=ndarrays_to_parameters(params),
        fraction_fit=0.4,
        fraction_evaluate=0.4,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=10,
    )

    strategy = FederatedStrategy.get_strategy("FedAvg", strategy_config)
    print(f"Created FedAvg strategy: {strategy}")

    # Server application setup
    fed_server = FederatedServer(
        strategy=strategy,
        num_rounds=config.num_rounds,
        initial_parameters=params,
    )
    server_app = fed_server.create_server_app()
    print(f"Server App created: {server_app}")
