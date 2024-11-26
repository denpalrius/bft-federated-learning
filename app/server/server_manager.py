from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import Strategy
from app.nets.model import CNNClassifier
from app.nets.train import ModelTrainer
from app.server.bft_basic_strategy import BFTBasicFedAvg
from app.server.server_evaluation import ServerEvaluation
from app.server.bft_strategy import BFTFedAvg

class ServerManager:
    """Manages the configuration and creation of the Server."""

    # TODO: Add better server and strategy configs
    # https://github.com/adap/flower/blob/main/examples/advanced-pytorch/pytorch_example/server_app.py
    
    def __init__(self, context: Context):
        self.context = context
        self.trainer = ModelTrainer(CNNClassifier())
        self.evaluator = ServerEvaluation(self.trainer, context)

    def create_strategy(self) -> Strategy:
        byzantine_threshold = float(self.context.run_config.get("byzantine-threshold", 0.7))
        max_deviation_threshold = float(self.context.run_config.get("max-deviation-threshold", 0.7))
        byzantine_clients = self.context.run_config.get("byzantine-clients", 5)
        min_clients = self.context.run_config.get("min-clients", 16)
        
        # Ensuring total number of clients satisfy the condition 𝑁 > 3𝑓
        min_clients = max(min_clients, 3 * byzantine_clients + 1)
        # TODO: Pass this server config
        
        # Add client eval
        # https://flower.ai/docs/framework/explanation-federated-evaluation.html
        
        strategy = BFTFedAvg(
            trainer=self.trainer,
            byzantine_threshold = byzantine_threshold,
            max_deviation_threshold=max_deviation_threshold,
            initial_parameters=self.trainer.get_model_parameters(),
            evaluate_fn=self.evaluator.gen_evaluate_fn(),
            evaluate_metrics_aggregation_fn=self.evaluator.weighted_average,
        )

        return strategy

    def get_config(self):
        num_rounds = self.context.run_config["num-server-rounds"]
        return ServerConfig(num_rounds=num_rounds)

    def create_components(self):
        strategy = self.create_strategy()
        config = self.get_config()
        return ServerAppComponents(strategy=strategy, config=config)
