from utils.path import add_base_path
import json
import os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

add_base_path(__file__)

class PerformanceEvaluator:
    """Tracks and saves performance metrics for FL experiments."""
    
    def __init__(self, output_dir: str = "../results"):
        self.output_dir = output_dir
        self.metrics: Dict[str, List] = {
            "rounds": [],
            "accuracy": [],
            "loss": [],
            "n_clients": [],
            "n_byzantine": [],
            "aggregation_method": [],
        }
        os.makedirs(output_dir, exist_ok=True)
        
    def add_round_metrics(
        self,
        round_num: int,
        accuracy: float,
        loss: float,
        n_clients: int,
        n_byzantine: int,
        aggregation_method: str
    ):
        """Add metrics for a training round."""
        self.metrics["rounds"].append(round_num)
        self.metrics["accuracy"].append(accuracy)
        self.metrics["loss"].append(loss)
        self.metrics["n_clients"].append(n_clients)
        self.metrics["n_byzantine"].append(n_byzantine)
        self.metrics["aggregation_method"].append(aggregation_method)
        
    def plot_metrics(self):
        """Generate plots of training metrics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics["rounds"], self.metrics["accuracy"], marker='o')
        plt.title("Model Accuracy over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"accuracy_{timestamp}.png"))
        plt.close()
        
        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics["rounds"], self.metrics["loss"], marker='o', color='r')
        plt.title("Model Loss over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"loss_{timestamp}.png"))
        plt.close()
        
        # Save metrics to JSON
        metrics_file = os.path.join(self.output_dir, f"metrics_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def generate_report(self) -> Dict:
        """Generate a summary report of the experiment."""
        report = {
            "experiment_summary": {
                "total_rounds": len(self.metrics["rounds"]),
                "final_accuracy": self.metrics["accuracy"][-1],
                "best_accuracy": max(self.metrics["accuracy"]),
                "final_loss": self.metrics["loss"][-1],
                "avg_clients_per_round": np.mean(self.metrics["n_clients"]),
                "avg_byzantine_per_round": np.mean(self.metrics["n_byzantine"]),
                "aggregation_method": self.metrics["aggregation_method"][-1],
            },
            "comparison": {
                "accuracy_improvement": self.metrics["accuracy"][-1] - self.metrics["accuracy"][0],
                "loss_improvement": self.metrics["loss"][0] - self.metrics["loss"][-1],
            }
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"report_{timestamp}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report