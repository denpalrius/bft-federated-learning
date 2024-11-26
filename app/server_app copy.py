"""bft-federated: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, Strategy
from bft_federated.bft.bft_strategy import BFTFedAvg
from bft_federated.nets.model import CNNClassifier
from bft_federated.task import get_weights
from bft_federated.utils.base_config import BaseConfig

base_config = BaseConfig()

def create_strategy(base_config) -> Strategy:
        """Create BFT aggregation strategy."""
        strategy = BFTFedAvg(
            threshold=base_config.threshold,
            bft_method=base_config.bft_method,
            # TODO: Experiment with different bft_method
        )
        print('====================')
        print(f'BFT Strategy: {strategy}')
        
        return strategy
    
def server_fn(context: Context):
    # Read from config
    
    print('====================')
    print(f'Context: {context}')
    
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(CNNClassifier())
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = create_strategy(base_config)

    # Define strategy
    # strategy = FedAvg(
    #     fraction_fit=fraction_fit,
    #     fraction_evaluate=1.0,
    #     min_available_clients=2,
    #     initial_parameters=parameters,
    # )
    
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
