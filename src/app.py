from config import BaseConfig
from flwr.client import ClientApp
from flwr.common import ndarrays_to_parameters

from config import BaseConfig
from model import CNNClassifier
from train import ModelTrainer
from fed_strategy import FederatedStrategy, StrategyConfig
from fed_client import FederatedClient
from fed_server import FederatedServer
from dataloader import DatasetLoader
from fed_simulator import SimulationRunner


def simulate():
    config = BaseConfig()
    model = CNNClassifier()
    trainer = ModelTrainer(model, config)

    datasetloader = DatasetLoader(config=config, num_partitions=10)
    trainloader, _, testloader = datasetloader.load_datasets(0)
    
    trainer.train(trainloader, epochs=10)
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
    print("\nCreated FedAvg strategy: {strategy}")

    # Server application setup
    fed_server = FederatedServer(
        strategy_name="FedAvg",
        strategy_config=strategy_config,
        num_rounds=3,
        initial_parameters=params,
    )
    server_app = fed_server.create_server_app()
    print(f"\nServer App created: {server_app}")

    # Client application setup
    fed_client = FederatedClient(config=config)
    client_app = ClientApp(client_fn=fed_client.create_client)
    print("\nClient App created: {client_app}")

    # Simulation configuration
    sim_runner = SimulationRunner(
        client_app=client_app,
        server_app=server_app,
        config=config,
    )

    # # Start simulation
    sim_runner.run()




if __name__ == "__main__":
    simulate()