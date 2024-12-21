import os
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import Strategy
from app.nets.cnn_classifier import CNNClassifier
from app.nets.train import ModelTrainer
from app.server.server_evaluation import ServerEvaluation
from app.server.bft_strategy import BFTFedAvg
from app.utils.logger import setup_logger

class ServerManager:
    """Manages the configuration and creation of the Server."""

    def __init__(self, context: Context):
        self.logger = setup_logger(self.__class__.__name__)

        self.context = context
        
        pretrained_model_path = context.run_config.get("pretrained-model-path")
        pretrained_model_path = os.path.join(os.getcwd(), pretrained_model_path)
        
        batch_size = context.run_config.get("batch-size")

        self.trainer = ModelTrainer(CNNClassifier(), pretrained_model_path)
        self.evaluator = ServerEvaluation(self.trainer, batch_size)

    def create_strategy(self) -> Strategy:
        fraction_fit = self.context.run_config.get("fraction-fit", 0.7)

        byzantine_threshold = float(self.context.run_config.get("byzantine-threshold", 0.7))
        max_deviation_threshold = float(self.context.run_config.get("max-deviation-threshold", 0.7))
        
        byzantine_clients = self.context.run_config.get("byzantine-clients", 0)
        
        strategy = BFTFedAvg(
            trainer=self.trainer,
            byzantine_threshold = byzantine_threshold,
            max_deviation_threshold=max_deviation_threshold,
            evaluate_fn=self.evaluator.gen_evaluate_fn(),
            evaluate_metrics_aggregation_fn=self.evaluator.weighted_average,
            fraction_fit=fraction_fit,
            byzantine_clients=byzantine_clients
        )

        return strategy

    def get_config(self):
        num_rounds = self.context.run_config["num-server-rounds"]
        return ServerConfig(num_rounds=num_rounds)

    def create_components(self):
        strategy = self.create_strategy()
        config = self.get_config()
        return ServerAppComponents(strategy=strategy, config=config)

