from typing import Callable, Dict
from flwr.server.strategy import Strategy
from flwr.common import Parameters
from flwr.server.strategy import FedAvg, FedAdagrad, FedTrimmedAvg, FedProx
from flwr.common import ndarrays_to_parameters

from config import BaseConfig
from model import CNNClassifier
from fed_evaluation import ServerEvaluator
from train import ModelTrainer

class StrategyConfig:    
    def __init__(
        self,
        initial_parameters: Parameters,
        fraction_fit: float = 0.3,
        fraction_evaluate: float = 0.3,
        min_fit_clients: int = 3,
        min_evaluate_clients: int = 3,
        min_available_clients: int = 10,
        **kwargs
    ):
        self.initial_parameters = initial_parameters
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.extra_params = kwargs


class FederatedStrategy:
    
    @staticmethod
    def get_strategy(strategy_name: str, config: StrategyConfig) -> Strategy:
        strategies: Dict[str, Callable] = {
            "FedAvg": FedAvg,
            "FedAdagrad": FedAdagrad,
            "FedTrimmedAvg": FedTrimmedAvg,
            "FedProx": FedProx,
        }

        evaluator = ServerEvaluator(BaseConfig())
        
        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Initialize the strategy with the provided configuration
        StrategyClass = strategies[strategy_name]
        strategy: Strategy = StrategyClass(
            fraction_fit=config.fraction_fit,
            fraction_evaluate=config.fraction_evaluate,
            min_fit_clients=config.min_fit_clients,
            min_evaluate_clients=config.min_evaluate_clients,
            min_available_clients=config.min_available_clients,
            initial_parameters=config.initial_parameters,
            evaluate_fn=evaluator.evaluate,
            **config.extra_params
        )
        
        return strategy



if __name__ == "__main__":
    
    config = BaseConfig()
    model = CNNClassifier()
    trainer = ModelTrainer(model, config)

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
    print(f'Created FedAvg strategy: {strategy}')
    