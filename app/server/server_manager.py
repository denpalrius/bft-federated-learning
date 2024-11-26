from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import Strategy
from app.nets.model import CNNClassifier
from app.nets.train import ModelTrainer
from app.server.bft_basic_strategy import BFTBasicFedAvg
from app.server.server_evaluation import ServerEvaluation

class ServerManager:
    """Manages the configuration and creation of the Server."""

    # TODO: Add better server and strategy configs
    # https://github.com/adap/flower/blob/main/examples/advanced-pytorch/pytorch_example/server_app.py
    
    def __init__(self, context: Context):
        self.context = context
        self.trainer = ModelTrainer(CNNClassifier())
        self.evaluator = ServerEvaluation(self.trainer, context)

    def create_strategy(self) -> Strategy:
        fraction_fit = self.context.run_config.get("fraction-fit", 0.7) 
        threshold = self.context.run_config.get("threshold", 0.7)       
        byzantine_clients = self.context.run_config.get("byzantine-clients", 5)
        min_clients = self.context.run_config.get("min-clients", 20)
        
        # Ensuring total number of clients satisfy the condition 𝑁 > 3𝑓
        min_clients = max(min_clients, 3 * byzantine_clients + 1)
        # TODO: Pass this server config
        
        # https://flower.ai/docs/framework/explanation-federated-evaluation.html
        strategy = BFTBasicFedAvg(
            threshold=threshold,
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
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
