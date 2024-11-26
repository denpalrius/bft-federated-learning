"""bft-federated: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp
from flwr.common import Context
from bft_federated.clients.flower_client import FlowerClient
from bft_federated.nets.model import CNNClassifier
from bft_federated.task import get_weights, load_data, set_weights, test, train


# Define the client_fn
def client_fn(context: Context):
    # Load model and data
    net = CNNClassifier()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]

    # print('=========context===========')
    # print(f'Context: {context}')
    
    trainloader, valloader = load_data(partition_id, num_partitions)

    net = set_weights(net, get_weights(CNNClassifier()))
    
    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
