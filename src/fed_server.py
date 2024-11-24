from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import ndarrays_to_parameters, Parameters

from config import BaseConfig
from model import CNNClassifier
from train import ModelTrainer
from fed_strategy import FederatedStrategy, StrategyConfig


class FederatedServer:
    """Handles the configuration and execution of the federated server."""

    def __init__(
        self,
        strategy_name: str,
        strategy_config: StrategyConfig,
        num_rounds: int,
        initial_parameters: Parameters,
    ):
        self.strategy_name = strategy_name
        self.strategy_config = strategy_config
        self.num_rounds = num_rounds
        self.initial_parameters = initial_parameters

    def __create_server_components(self) -> ServerAppComponents:
        strategy = FederatedStrategy.get_strategy(
            self.strategy_name, self.strategy_config
        )
        server_config = ServerConfig(num_rounds=self.num_rounds)

        return ServerAppComponents(strategy=strategy, config=server_config)

    def create_server_app(self) -> ServerApp:
        server = ServerApp(server_fn=self.__create_server_components)
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
        strategy_name="FedAvg",
        strategy_config=strategy_config,
        num_rounds=3,
        initial_parameters=params,
    )
    server_app = fed_server.create_server_app()
    print(f"Server App created: {server_app}")
