"""fed-ner: A Flower / PyTorch app."""

import csv
import os
import numpy as np
import wandb

from flwr.common import Context, ndarrays_to_parameters, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import EvaluateRes
from fed_ner.task import Net, get_weights


# Define CSV file path
CSV_FILE = "fl_metrics.csv"


def init_csv():
    """Initialize CSV with headers if it doesn't exist."""
    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "accuracy", "f1", "communication_cost_bytes"])


def append_csv(round_num, accuracy, f1_score, communication_cost):
    """Append metrics to the CSV log."""
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([round_num, accuracy, f1_score, communication_cost])


def parameters_to_ndarrays(parameters: Parameters):
    """Convert Flower Parameters to list of numpy ndarrays."""
    ndarrays = []
    for tensor in parameters.tensors:
        if isinstance(tensor, (bytes, bytearray)):
            arr = np.frombuffer(tensor, dtype=np.float32)
        else:
            arr = np.array(tensor)
        ndarrays.append(arr)
    return ndarrays


class CustomFedAvg(FedAvg):
    """Custom strategy that logs accuracy, F1, and communication cost."""

    def __init__(self, initial_parameters, *args, **kwargs):
        super().__init__(initial_parameters=initial_parameters, *args, **kwargs)
        self.current_round = 0
        self._wandb_initalized = False

    def _init_wandb(self):
        if not self._wandb_initalized:
            wandb.init(project="fed-ner", name="flower-federated-ner", reinit=True)
            self._wandb_initalized = True

    def aggregate_fit(self, rnd, results, failures):
        agg_fit_res = super().aggregate_fit(rnd, results, failures)
        if agg_fit_res is not None:
            self.initial_parameters = agg_fit_res[0]  # Update weights after training
        return agg_fit_res

    def aggregate_evaluate(self, rnd, results, failures):
        self._init_wandb()
        agg_res = super().aggregate_evaluate(rnd, results, failures)

        if agg_res is not None and self.initial_parameters is not None:
            loss, metrics = agg_res

            accuracies = [res.metrics.get("accuracy") for _, res in results if res.metrics]
            f1_scores = [res.metrics.get("f1") for _, res in results if res.metrics]

            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

            model_ndarrays = parameters_to_ndarrays(self.initial_parameters)
            model_size_bytes = sum(arr.nbytes for arr in model_ndarrays)
            communication_cost = model_size_bytes * len(results)

            wandb.log({
                "round": self.current_round,
                "accuracy": avg_accuracy,
                "f1": avg_f1,
                "communication_cost_bytes": communication_cost,
            })
            print(f"-----------rnd: {self.current_round}, acc: {avg_accuracy}, f1: {avg_f1}, com-cost: {communication_cost}-----------")
            append_csv(self.current_round, avg_accuracy, avg_f1, communication_cost)

        self.current_round += 1
        return agg_res


def server_fn(context: Context):
    """Define the server function for Flower app."""
    wandb.init(project="fed-ner", name="flower-federated-ner", reinit=True)

    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)

    init_csv()

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)