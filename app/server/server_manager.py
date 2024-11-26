from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import Strategy
from app.nets.model import CNNClassifier
from app.nets.train import ModelTrainer
from app.server.federated_strategy_basic import BasicBFTFedAvg


class ServerManager:
    """Manages the configuration and creation of the Server."""

    # TODO: Add better server and strategy configs
    # https://github.com/adap/flower/blob/main/examples/advanced-pytorch/pytorch_example/server_app.py
    
    def __init__(self, context: Context):
        self.context = context
        self.trainer = ModelTrainer(CNNClassifier())

    def create_strategy(self) -> Strategy:
        fraction_fit = self.context.run_config.get("fraction-fit", 0.7) 
        threshold = self.context.run_config.get("threshold", 0.7)       
        byzantine_clients = self.context.run_config.get("byzantine-clients", 5)
        min_clients = self.context.run_config.get("min-clients", 20)
        
        # Ensuring total number of clients satisfy the condition 𝑁 > 3𝑓
        min_clients = max(min_clients, 3 * byzantine_clients + 1)
        # TODO: Pass this server config
        
        # print('byzantine_clients:', byzantine_clients)
        # print('min_clients:', min_clients)

        # ndarrays = get_weights(Net())
        # parameters = ndarrays_to_parameters(ndarrays)

        # TODO: Add evaluate_fn
        # https://flower.ai/docs/framework/explanation-federated-evaluation.html
        # https://flower.ai/docs/framework/explanation-federated-evaluation.html
        strategy = BasicBFTFedAvg(
            threshold=threshold,
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=self.trainer.get_model_parameters(),
        )

        # TODO: Experiment with different bft_method
        # TODO: What is a good Threshld value?

        return strategy

    def get_config(self):
        num_rounds = self.context.run_config["num-server-rounds"]
        return ServerConfig(num_rounds=num_rounds)

    def create_components(self):
        strategy = self.create_strategy()
        config = self.get_config()
        return ServerAppComponents(strategy=strategy, config=config)
