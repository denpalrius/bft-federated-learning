# Federated Learning with Byzantine Fault Tolerance (BFT)

> Client/Server simulation using [Flower](https://flower.ai/)

## Features

- Simulation of malicious nodes
- Performance evaluation

## Install dependencies and project

```bash
pip install -e .
```

## Loading initial weights
Run the notebook to build some initial weights

## Run with the Simulation Engine

At the root of the repository code, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.


## BFT Strategy for Robust Simulation

Ensure the BFT method (e.g., "krum") operates correctly by configuring the following parameters for your system.

### Sufficient Number of Clients
- For BFT to function effectively, the total number of participating clients \(N\) must satisfy the inequality:
  \[
  N > 3f
  \]
  where \(f\) is the maximum number of Byzantine clients you want to tolerate.
- ***Example***: If you want to tolerate \(f = 3\) Byzantine clients, you need at least \(N = 10\) clients (since \(3 \times 3 + 1 = 10\)).
- To account for variability and ensure smooth testing, increase the number of clients beyond the minimum required. For example, set `num-clients` to 15 or 20.

### Fraction of Clients per Round
- The `fraction-fit` parameter determines the fraction of clients that will participate in each round. To ensure robustness in a BFT scenario, ensure that the fraction is high enough to include at least \(f + 1\) clients.
- ***Example***: With `fraction-fit = 0.5` and `num-clients = 15`, approximately 7-8 clients will participate in each round. If \(f = 3\), ensure that at least 4 clients participate in every round to maintain the integrity of the aggregation process.

### Aggregation Method
- The `bft-method` configuration (e.g., "krum") should be set to a Byzantine Fault Tolerant method to handle client updates robustly.
- Ensure that your system logs the aggregation details to verify that the chosen method is functioning correctly.

### Configuration Example
Here’s a sample configuration to implement these strategies in your simulation setup:

```toml
[tool.flwr.app.config]
num-server-rounds = 5           # Increased rounds for better model convergence
fraction-fit = 0.7              # Higher fraction to ensure more clients participate in each round
threshold = 0.7                 # Higher threshold for stronger aggregation consensus
bft-method = "krum"             # Robust BFT method for aggregating updates
local-epochs = 2                # Increased epochs for better local model training

[tool.flwr.federations.local-simulation]
options.num-supernodes = 12     # Increased supernodes to enhance robustness in aggregation
options.num-nodes = 21          # Total nodes satisfying N > 3f (with f = 5 Byzantine clients, N = 16+)
options.num-clients = 21        # Sufficient clients for a robust BFT simulation
options.num-epochs = 10         # Maintain sufficient epochs for effective local training
options.batch-size = 32         # Suitable batch size for training
options.lr = 0.01               # Learning rate for local updates
options.seed = 42               # Seed for reproducibility
```



## Resources
- Flower documentation: [flower.ai/docs](https://flower.ai/docs/)
