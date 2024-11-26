"""bft-federated: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp
from flwr.common import Context
# from bft_federated.clients.federated_client import FederatedClient
from clients.flower_client import FlowerClient
from nets.model import CNNClassifier
from task import get_weights, load_data, set_weights, test, train
# from bft_federated.utils.bft_simulation_config import BFTSimulationConfig


# Define the client_fn
def client_fn(context: Context):
    # Load model and data
    net = CNNClassifier()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]

    # Example context configuration
    # context = Context(
    #     node_config={"partition-id": 2, "num-partitions": 10},
    #     run_config={"local-epochs": 3},
    #     sim_config=BFTSimulationConfig(
    #         name="example_sim",
    #         num_clients=10,
    #         num_malicious=2,
    #         num_rounds=5,
    #         batch_size=32,
    #         bft_method="krum",
    #         device=torch.device("cpu"),
    #         dataset_fn=get_cifar10,
    #         model_fn=get_simple_cnn,
    #     )
    # )

    # # Create client
    # client = client_fn(context)


    print('=========context===========')
    print(context)
    
    trainloader, valloader = load_data(partition_id, num_partitions)

    #     experiment_name="cifar10_bft",
        # base_config=BaseConfig(),
        # dataset_fn=get_cifar10,
        # model_fn=get_simple_cnn,
        
    # sim_config = BFTSimulationConfig(
    #     experiment_name="cifar10_bft",
    #     bft_method=bft_method,
    #     base_config=base_config,
    #     dataset_fn=dataset_fn,
    #     model_fn=model_fn,
    # )
    # federated_client = FederatedClient(sim_config)
    # client = federated_client.create_client(partition_id, num_partitions, trainloader, valloader, local_epochs,)

    
    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()
    # return client.to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
