from dataclasses import dataclass
from typing import List
from enum import Enum
import logging
from federated_learning.experiment import run_federated_learning_experiment
from models.experiment_config import ExperimentConfig
from models.maliciopus_type import MaliciousType
from utils.logger import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    # Define the experiment configurations as instances of ExperimentConfig
    experiments = [
        ExperimentConfig(
            num_rounds=10,
            num_clients=10,
            num_malicious=0,
            malicious_types=[],
            use_bft=False,
            experiment_name="baseline"
        ),
        ExperimentConfig(
            num_rounds=10,
            num_clients=10,
            num_malicious=3,
            malicious_types=[MaliciousType.RANDOM_UPDATES, MaliciousType.SCALED_UPDATES],
            use_bft=False,
            experiment_name="with_malicious_no_bft"
        ),
        ExperimentConfig(
            num_rounds=10,
            num_clients=10,
            num_malicious=3,
            malicious_types=[MaliciousType.RANDOM_UPDATES, MaliciousType.SCALED_UPDATES],
            use_bft=True,
            experiment_name="with_malicious_with_bft"
        ),
    ]
    
    # Run each experiments
    for config in experiments:
        logger.info(f"Starting experiment: {config.experiment_name}")
        experiment = FederatedLearningExperiment(config)
        experiment.run()
        logger.info(f"Completed experiment: {config.experiment_name}")
