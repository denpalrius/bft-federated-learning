import argparse
import logging
from base_config import BaseConfig
from flwr.client import ClientApp
from flwr.common import ndarrays_to_parameters
from fed_client import FederatedClient
from model import CNNClassifier
from train import ModelTrainer
from fed_strategy import FederatedStrategy, StrategyConfig
from fed_server import FederatedServer
from dataset_loader import DatasetLoader
from simulator import SimulationRunner


def simulate(epochs: int):
    # Use the passed epochs or the default from the config
    config = BaseConfig()    
    model = CNNClassifier()
    trainer = ModelTrainer(model, config)

    datasetloader = DatasetLoader(config=config)
    trainloader, _, testloader = datasetloader.load_datasets(0)
    
    trainer.train(trainloader, epochs=epochs)
    trainer.test(testloader)

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
    print(f"\nCreated FedAvg strategy: {strategy}")

    # Server application setup
    fed_server = FederatedServer(
        strategy=strategy,
        num_rounds=config.num_rounds,
        initial_parameters=params,
    )
    server_app = fed_server.create_server_app()
    print(f"\nServer App created: {server_app}")

    # Client application setup
    fed_client = FederatedClient(config=config)
    client_app = ClientApp(client_fn=fed_client.create_client)
    print(f"\nClient App created: {client_app}")

    # Simulation configuration
    sim_runner = SimulationRunner(
        client_app=client_app,
        server_app=server_app,
        config=config,
    )

    # Start simulation
    sim_runner.run()


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Federated Learning Simulation")
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs for training", default=None
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Global logging configuration
    logging.basicConfig(
        filename='output.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Set log level for specific third-party packages
    logging.getLogger("flwr").setLevel(logging.INFO)

    # Your code
    logging.debug("This is a debug message.")
    logging.info("This is an info message.")


    # Parse the command-line arguments
    args = parse_args()
    print(f"Arguments: {args}")

    # If epochs is provided via command line, use it; otherwise, fall back to config's default
    epochs = args.epochs if args.epochs is not None else BaseConfig().epochs

    # Run the simulation with the appropriate epochs
    simulate(epochs)
    
    # config = BaseConfig()
    # print('Type of device:', type(config.device))
    

