# import numpy as np
# from typing import List, Dict, Any, Tuple, Optional
# from enum import Enum
# import json
# import time
# import random
# from collections import defaultdict
# import logging
# from dataclasses import dataclass
# import torch
# import matplotlib.pyplot as plt
# from datetime import datetime

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class MessageType(Enum):
#     PRE_PREPARE = "pre-prepare"
#     PREPARE = "prepare"
#     COMMIT = "commit"
#     REPLY = "reply"

# @dataclass
# class BFTMessage:
#     msg_type: MessageType
#     view_number: int
#     sequence_number: int
#     client_id: int
#     content: Any
#     timestamp: float
#     signature: str  # In practice, use proper cryptographic signatures

# class PBFTConsensus:
#     def __init__(self, total_nodes: int, max_faulty: int):
#         self.total_nodes = total_nodes
#         self.max_faulty = max_faulty
#         self.sequence_number = 0
#         self.view_number = 0
#         self.prepare_messages = defaultdict(dict)
#         self.commit_messages = defaultdict(dict)
#         self.prepared_messages = set()
#         self.committed_messages = set()
        
#     def create_message(self, msg_type: MessageType, client_id: int, content: Any) -> BFTMessage:
#         self.sequence_number += 1
#         return BFTMessage(
#             msg_type=msg_type,
#             view_number=self.view_number,
#             sequence_number=self.sequence_number,
#             client_id=client_id,
#             content=content,
#             timestamp=time.time(),
#             signature=f"sig_{client_id}_{self.sequence_number}"  # Simplified signature
#         )
    
#     def validate_message(self, message: BFTMessage) -> bool:
#         # In practice, implement proper message validation including signature verification
#         return True
    
#     def process_message(self, message: BFTMessage) -> bool:
#         if not self.validate_message(message):
#             return False
            
#         if message.msg_type == MessageType.PRE_PREPARE:
#             # Primary sends pre-prepare message
#             return True
            
#         elif message.msg_type == MessageType.PREPARE:
#             self.prepare_messages[message.sequence_number][message.client_id] = message
#             if len(self.prepare_messages[message.sequence_number]) >= 2 * self.max_faulty + 1:
#                 self.prepared_messages.add(message.sequence_number)
#                 return True
                
#         elif message.msg_type == MessageType.COMMIT:
#             self.commit_messages[message.sequence_number][message.client_id] = message
#             if len(self.commit_messages[message.sequence_number]) >= 2 * self.max_faulty + 1:
#                 self.committed_messages.add(message.sequence_number)
#                 return True
                
#         return False

# class MaliciousType(Enum):
#     NONE = "none"
#     RANDOM_UPDATES = "random_updates"
#     SCALED_UPDATES = "scaled_updates"
#     CONSTANT_UPDATES = "constant_updates"

# class MaliciousClient(FlowerClient):
#     def __init__(
#         self,
#         partition_id: int,
#         net: torch.nn.Module,
#         trainloader: torch.utils.data.DataLoader,
#         valloader: torch.utils.data.DataLoader,
#         malicious_type: MaliciousType,
#         attack_intensity: float = 1.0
#     ):
#         super().__init__(partition_id, net, trainloader, valloader)
#         self.malicious_type = malicious_type
#         self.attack_intensity = attack_intensity

#     def get_malicious_parameters(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
#         if self.malicious_type == MaliciousType.NONE:
#             return parameters
            
#         elif self.malicious_type == MaliciousType.RANDOM_UPDATES:
#             return [np.random.randn(*param.shape) * self.attack_intensity for param in parameters]
            
#         elif self.malicious_type == MaliciousType.SCALED_UPDATES:
#             return [param * self.attack_intensity for param in parameters]
            
#         elif self.malicious_type == MaliciousType.CONSTANT_UPDATES:
#             return [np.ones_like(param) * self.attack_intensity for param in parameters]
            
#         return parameters

#     def fit(self, parameters, config):
#         # Original training
#         updated_parameters, num_examples, metrics = super().fit(parameters, config)
        
#         # Apply malicious modifications
#         malicious_parameters = self.get_malicious_parameters(updated_parameters)
        
#         return malicious_parameters, num_examples, metrics

# class ExperimentTracker:
#     def __init__(self, experiment_name: str):
#         self.experiment_name = experiment_name
#         self.metrics = {
#             'accuracy': [],
#             'loss': [],
#             'round_time': [],
#             'consensus_time': [],
#             'num_malicious': [],
#             'malicious_types': [],
#             'bft_enabled': []
#         }
#         self.start_time = time.time()

#     def add_round_metrics(
#         self,
#         accuracy: float,
#         loss: float,
#         round_time: float,
#         consensus_time: float,
#         num_malicious: int,
#         malicious_types: List[MaliciousType],
#         bft_enabled: bool
#     ):
#         self.metrics['accuracy'].append(accuracy)
#         self.metrics['loss'].append(loss)
#         self.metrics['round_time'].append(round_time)
#         self.metrics['consensus_time'].append(consensus_time)
#         self.metrics['num_malicious'].append(num_malicious)
#         self.metrics['malicious_types'].append([mt.value for mt in malicious_types])
#         self.metrics['bft_enabled'].append(bft_enabled)

#     def save_metrics(self):
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"metrics_{self.experiment_name}_{timestamp}.json"
        
#         with open(filename, 'w') as f:
#             json.dump(self.metrics, f, indent=4)
        
#         # Create performance visualization
#         self.plot_metrics(filename.replace('.json', '.png'))
        
#         logger.info(f"Metrics saved to {filename}")

#     def plot_metrics(self, filename: str):
#         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
#         # Plot accuracy over rounds
#         rounds = range(1, len(self.metrics['accuracy']) + 1)
#         ax1.plot(rounds, self.metrics['accuracy'], marker='o')
#         ax1.set_title('Model Accuracy over Rounds')
#         ax1.set_xlabel('Round')
#         ax1.set_ylabel('Accuracy')
        
#         # Plot loss over rounds
#         ax2.plot(rounds, self.metrics['loss'], marker='o', color='red')
#         ax2.set_title('Model Loss over Rounds')
#         ax2.set_xlabel('Round')
#         ax2.set_ylabel('Loss')
        
#         # Plot round time and consensus time
#         ax3.plot(rounds, self.metrics['round_time'], marker='s', label='Total Round Time')
#         ax3.plot(rounds, self.metrics['consensus_time'], marker='^', label='Consensus Time')
#         ax3.set_title('Time Performance')
#         ax3.set_xlabel('Round')
#         ax3.set_ylabel('Time (seconds)')
#         ax3.legend()
        
#         # Plot number of malicious clients
#         ax4.bar(rounds, self.metrics['num_malicious'])
#         ax4.set_title('Number of Malicious Clients per Round')
#         ax4.set_xlabel('Round')
#         ax4.set_ylabel('Number of Malicious Clients')
        
#         plt.tight_layout()
#         plt.savefig(filename)
#         plt.close()

# class BFTServerStrategy(FedAvg):
#     def __init__(
#         self,
#         *args,
#         pbft_consensus: Optional[PBFTConsensus] = None,
#         experiment_tracker: Optional[ExperimentTracker] = None,
#         **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         self.pbft_consensus = pbft_consensus
#         self.experiment_tracker = experiment_tracker

#     def aggregate_fit(
#         self,
#         server_round: int,
#         results: List[Tuple[NDArrays, int]],
#         failures: List[BaseException],
#     ) -> Tuple[Optional[NDArrays], Dict[str, Scalar]]:
        
#         if self.pbft_consensus:
#             consensus_start = time.time()
            
#             # Create pre-prepare message
#             pre_prepare = self.pbft_consensus.create_message(
#                 MessageType.PRE_PREPARE,
#                 client_id=0,  # Server ID
#                 content=results
#             )
            
#             # Simulate PBFT consensus
#             prepare_messages = []
#             commit_messages = []
            
#             for client_id in range(self.pbft_consensus.total_nodes):
#                 # Prepare phase
#                 prepare = self.pbft_consensus.create_message(
#                     MessageType.PREPARE,
#                     client_id=client_id,
#                     content=results
#                 )
#                 self.pbft_consensus.process_message(prepare)
#                 prepare_messages.append(prepare)
                
#                 # Commit phase
#                 commit = self.pbft_consensus.create_message(
#                     MessageType.COMMIT,
#                     client_id=client_id,
#                     content=results
#                 )
#                 self.pbft_consensus.process_message(commit)
#                 commit_messages.append(commit)
            
#             consensus_time = time.time() - consensus_start
            
#             # Check if consensus was reached
#             if pre_prepare.sequence_number not in self.pbft_consensus.committed_messages:
#                 return None, {}
        
#         # Aggregate updates using FedAvg
#         aggregated_updates, metrics = super().aggregate_fit(server_round, results, failures)
        
#         if self.experiment_tracker:
#             round_time = time.time() - consensus_start
#             self.experiment_tracker.add_round_metrics(
#                 accuracy=metrics.get('accuracy', 0.0),
#                 loss=metrics.get('loss', 0.0),
#                 round_time=round_time,
#                 consensus_time=consensus_time if self.pbft_consensus else 0.0,
#                 num_malicious=sum(1 for r in results if isinstance(r[0], MaliciousClient)),
#                 malicious_types=[r[0].malicious_type for r in results if isinstance(r[0], MaliciousClient)],
#                 bft_enabled=bool(self.pbft_consensus)
#             )
        
#         return aggregated_updates, metrics

# def run_federated_learning_experiment(
#     num_rounds: int,
#     num_clients: int,
#     num_malicious: int,
#     malicious_types: List[MaliciousType],
#     use_bft: bool,
#     experiment_name: str
# ):
#     # Initialize experiment tracker
#     tracker = ExperimentTracker(experiment_name)
    
#     # Initialize PBFT consensus if enabled
#     pbft_consensus = None
#     if use_bft:
#         max_faulty = (num_clients - 1) // 3  # Maximum number of Byzantine faults tolerable
#         pbft_consensus = PBFTConsensus(num_clients, max_faulty)
    
#     # Create server strategy
#     strategy = BFTServerStrategy(
#         pbft_consensus=pbft_consensus,
#         experiment_tracker=tracker,
#         fraction_fit=0.3,
#         fraction_evaluate=0.3,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=num_clients,
#         initial_parameters=ndarrays_to_parameters(get_parameters(Net()))
#     )
    
#     # Create clients (including malicious ones)
#     clients = []
#     malicious_indices = random.sample(range(num_clients), num_malicious)
    
#     for i in range(num_clients):
#         net = Net().to(DEVICE)
#         trainloader, valloader, _ = load_datasets(i, num_clients)
        
#         if i in malicious_indices:
#             malicious_type = random.choice(malicious_types)
#             client = MaliciousClient(
#                 partition_id=i,
#                 net=net,
#                 trainloader=trainloader,
#                 valloader=valloader,
#                 malicious_type=malicious_type,
#                 attack_intensity=random.uniform(0.5, 2.0)
#             )
#         else:
#             client = FlowerClient(i, net, trainloader, valloader)
        
#         clients.append(client)
    
#     # Run federated learning
#     fl_config = ServerConfig(num_rounds=num_rounds)
#     server = ServerApp(lambda x: ServerAppComponents(strategy=strategy, config=fl_config))
    
#     run_simulation(
#         server_app=server,
#         client_app=ClientApp(lambda x: random.choice(clients).to_client()),
#         num_supernodes=num_clients,
#         backend_config={"client_resources": None if DEVICE.type == "cpu" else {"num_gpus": 1}}
#     )
    
#     # Save experiment results
#     tracker.save_metrics()

# # Example usage
# if __name__ == "__main__":
#     # Run experiments with different configurations
#     experiments = [
#         {
#             "num_rounds": 10,
#             "num_clients": 10,
#             "num_malicious": 0,
#             "malicious_types": [],
#             "use_bft": False,
#             "experiment_name": "baseline"
#         },
#         {
#             "num_rounds": 10,
#             "num_clients": 10,
#             "num_malicious": 3,
#             "malicious_types": [MaliciousType.RANDOM_UPDATES, MaliciousType.SCALED_UPDATES],
#             "use_bft": False,
#             "experiment_name": "with_malicious_no_bft"
#         },
#         {
#             "num_rounds": 10,
#             "num_clients": 10,
#             "num_malicious": 3,
#             "malicious_types": [MaliciousType.RANDOM_UPDATES, MaliciousType.SCALED_UPDATES],
#             "use_bft": True,
#             "experiment_name": "with_malicious_with_bft"
#         }
#     ]
    
#     for exp_config in experiments:
#         logger.info(f"Running experiment: {exp_config['experiment_name']}")
#         run_federated_learning_experiment(**exp_config)
#         logger.info(f"Completed experiment: {exp_config['experiment_name']}")