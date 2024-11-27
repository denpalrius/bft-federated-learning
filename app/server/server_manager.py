import os
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import Strategy
import torch
from app.nets.cifar10_cnn import CIFAR10Model
from app.nets.model import CNNClassifier
from app.nets.train import ModelTrainer
from app.server.server_evaluation import ServerEvaluation
from app.server.bft_strategy import BFTFedAvg
from app.utils.logger import setup_logger

class ServerManager:
    """Manages the configuration and creation of the Server."""

    def __init__(self, context: Context):
        self.logger = setup_logger(self.__class__.__name__)

        self.context = context
        
        pretrained_model_path = context.run_config.get("pretrained-model-path", None)
        pretrained_model_path = os.path.join(os.getcwd(), pretrained_model_path)

        self.trainer = ModelTrainer(CIFAR10Model(), pretrained_model_path)
        self.evaluator = ServerEvaluation(self.trainer, context.run_config)

    def create_strategy(self) -> Strategy:
        fraction_fit = self.context.run_config.get("fraction-fit", 0.7)

        byzantine_threshold = float(self.context.run_config.get("byzantine-threshold", 0.7))
        max_deviation_threshold = float(self.context.run_config.get("max-deviation-threshold", 0.7))
        
        # byzantine_clients = self.context.run_config.get("byzantine-clients", 5)
        # min_clients = self.context.run_config.get("min-clients", 16)

        # # Ensuring total number of clients satisfy the condition 𝑁 > 3𝑓
        # min_clients = max(min_clients, 3 * byzantine_clients + 1)
        # TODO: Pass this server config
        
        strategy = BFTFedAvg(
            trainer=self.trainer,
            byzantine_threshold = byzantine_threshold,
            max_deviation_threshold=max_deviation_threshold,
            evaluate_fn=self.evaluator.gen_evaluate_fn(),
            evaluate_metrics_aggregation_fn=self.evaluator.weighted_average,
            fraction_fit=fraction_fit,
        )

        return strategy

    def get_config(self):
        num_rounds = self.context.run_config["num-server-rounds"]
        return ServerConfig(num_rounds=num_rounds)

    def create_components(self):
        strategy = self.create_strategy()
        config = self.get_config()
        return ServerAppComponents(strategy=strategy, config=config)

