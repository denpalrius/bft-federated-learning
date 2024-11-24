import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
import json
import time
import random
from collections import defaultdict
import logging
from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt
from datetime import datetime


class BFTServerStrategy(FedAvg):
    def __init__(
        self,
        *args,
        pbft_consensus: Optional[PBFTConsensus] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pbft_consensus = pbft_consensus
        self.experiment_tracker = experiment_tracker

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[NDArrays, int]],
        failures: List[BaseException],
    ) -> Tuple[Optional[NDArrays], Dict[str, Scalar]]:
        
        if self.pbft_consensus:
            consensus_start = time.time()
            
            # Create pre-prepare message
            pre_prepare = self.pbft_consensus.create_message(
                MessageType.PRE_PREPARE,
                client_id=0,  # Server ID
                content=results
            )
            
            # Simulate PBFT consensus
            prepare_messages = []
            commit_messages = []
            
            for client_id in range(self.pbft_consensus.total_nodes):
                # Prepare phase
                prepare = self.pbft_consensus.create_message(
                    MessageType.PREPARE,
                    client_id=client_id,
                    content=results
                )
                self.pbft_consensus.process_message(prepare)
                prepare_messages.append(prepare)
                
                # Commit phase
                commit = self.pbft_consensus.create_message(
                    MessageType.COMMIT,
                    client_id=client_id,
                    content=results
                )
                self.pbft_consensus.process_message(commit)
                commit_messages.append(commit)
            
            consensus_time = time.time() - consensus_start
            
            # Check if consensus was reached
            if pre_prepare.sequence_number not in self.pbft_consensus.committed_messages:
                return None, {}
        
        # Aggregate updates using FedAvg
        aggregated_updates, metrics = super().aggregate_fit(server_round, results, failures)
        
        if self.experiment_tracker:
            round_time = time.time() - consensus_start
            self.experiment_tracker.add_round_metrics(
                accuracy=metrics.get('accuracy', 0.0),
                loss=metrics.get('loss', 0.0),
                round_time=round_time,
                consensus_time=consensus_time if self.pbft_consensus else 0.0,
                num_malicious=sum(1 for r in results if isinstance(r[0], MaliciousClient)),
                malicious_types=[r[0].malicious_type for r in results if isinstance(r[0], MaliciousClient)],
                bft_enabled=bool(self.pbft_consensus)
            )
        
        return aggregated_updates, metrics
