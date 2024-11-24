from dataclasses import dataclass
from typing import List
from enum import Enum
import random
import logging

from models.experiment_config import ExperimentConfig

# Placeholder imports for referenced classes and functions
# from your_project import PBFTConsensus, BFTServerStrategy, FlowerClient, MaliciousClient, Net, load_datasets
# from your_project import ExperimentTracker, ndarrays_to_parameters, ServerConfig, ServerApp, ServerAppComponents, run_simulation, ClientApp



class FederatedLearningExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tracker = ExperimentTracker(config.experiment_name)
        self.pbft_consensus = None

        if config.use_bft:
            max_faulty = (config.num_clients - 1) // 3
            self.pbft_consensus = PBFTConsensus(config.num_clients, max_faulty)

    def create_clients(self) -> List:
        """Create federated learning clients, including malicious ones."""
        clients = []
        malicious_indices = random.sample(range(self.config.num_clients), self.config.num_malicious)

        for i in range(self.config.num_clients):
            net = Net().to("cpu")  # Placeholder for device setup
            trainloader, valloader, _ = load_datasets(i, self.config.num_clients)

            if i in malicious_indices:
                malicious_type = random.choice(self.config.malicious_types)
                client = MaliciousClient(
                    partition_id=i,
                    net=net,
                    trainloader=trainloader,
                    valloader=valloader,
                    malicious_type=malicious_type,
                    attack_intensity=random.uniform(0.5, 2.0)
                )
            else:
                client = FlowerClient(i, net, trainloader, valloader)

            clients.append(client)
        return clients

    def run(self):
        """Run the federated learning experiment."""
        # Create the server strategy
        strategy = BFTServerStrategy(
            pbft_consensus=self.pbft_consensus,
            experiment_tracker=self.tracker,
            fraction_fit=0.3,
            fraction_evaluate=0.3,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=self.config.num_clients,
            initial_parameters=ndarrays_to_parameters(get_parameters(Net()))
        )

        # Create clients
        clients = self.create_clients()

        # Setup the server
        fl_config = ServerConfig(num_rounds=self.config.num_rounds)
        server = ServerApp(lambda x: ServerAppComponents(strategy=strategy, config=fl_config))

        # Run the simulation
        run_simulation(
            server_app=server,
            client_app=ClientApp(lambda x: random.choice(clients).to_client()),
            num_supernodes=self.config.num_clients,
            backend_config={"client_resources": None if "cpu" else {"num_gpus": 1}}
        )

        # Save experiment results
        self.tracker.save_metrics()

