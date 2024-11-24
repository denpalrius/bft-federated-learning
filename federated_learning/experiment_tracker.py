from models.maliciopus_type import MaliciousType
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

from utils.logger import get_logger

logger = get_logger(__name__)

logger.info("Experiment Tracker loaded")

class ExperimentTracker:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.metrics = {
            'accuracy': [],
            'loss': [],
            'round_time': [],
            'consensus_time': [],
            'num_malicious': [],
            'malicious_types': [],
            'bft_enabled': []
        }
        self.start_time = time.time()

    def add_round_metrics(
        self,
        accuracy: float,
        loss: float,
        round_time: float,
        consensus_time: float,
        num_malicious: int,
        malicious_types: List[MaliciousType],
        bft_enabled: bool
    ):
        self.metrics['accuracy'].append(accuracy)
        self.metrics['loss'].append(loss)
        self.metrics['round_time'].append(round_time)
        self.metrics['consensus_time'].append(consensus_time)
        self.metrics['num_malicious'].append(num_malicious)
        self.metrics['malicious_types'].append([mt.value for mt in malicious_types])
        self.metrics['bft_enabled'].append(bft_enabled)

    def save_metrics(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{self.experiment_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Create performance visualization
        self.plot_metrics(filename.replace('.json', '.png'))
        
        logger.info(f"Metrics saved to {filename}")

    def plot_metrics(self, filename: str):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy over rounds
        rounds = range(1, len(self.metrics['accuracy']) + 1)
        ax1.plot(rounds, self.metrics['accuracy'], marker='o')
        ax1.set_title('Model Accuracy over Rounds')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy')
        
        # Plot loss over rounds
        ax2.plot(rounds, self.metrics['loss'], marker='o', color='red')
        ax2.set_title('Model Loss over Rounds')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Loss')
        
        # Plot round time and consensus time
        ax3.plot(rounds, self.metrics['round_time'], marker='s', label='Total Round Time')
        ax3.plot(rounds, self.metrics['consensus_time'], marker='^', label='Consensus Time')
        ax3.set_title('Time Performance')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Time (seconds)')
        ax3.legend()
        
        # Plot number of malicious clients
        ax4.bar(rounds, self.metrics['num_malicious'])
        ax4.set_title('Number of Malicious Clients per Round')
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Number of Malicious Clients')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
