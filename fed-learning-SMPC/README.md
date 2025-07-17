# fed-learning-SMPC: A Flower / TensorFlow app

## Install dependencies and project

```bash
pip install -r requirements.txt
```

## Run with the Simulation

In the `fed-learning-SMPC` directory, use `flwr run .` to run a local simulation:

```bash
flwr run .
```

## Code Overview

This is essentially the logic of the federated server and client server. It is used to send ecrypted gradients and LoRA adapters from the client server to the federated server where they are aggregated. Aggregated updates are then sent from the federated server to the client to continue training.

