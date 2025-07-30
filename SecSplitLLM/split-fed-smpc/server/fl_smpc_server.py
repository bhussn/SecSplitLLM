import os
os.environ["CRYPTEN_USE_CPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import GPUtil
import csv
import pickle
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import flwr as fl
from models.split_bert_LoRA import BertSplitConfig, BertModel_Client, BertModel_Server
import time
from logs.metric_logger import MetricsLogger
from smpc.smpc_aggregator import aggregate_encrypted_updates
import crypten
import io
import numpy as np
import logging
import ast

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    filename="flower_debug.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

run = 0
dataname = "sst2"
MAX_MSG_SIZE = 2_000_000_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_dir = os.path.join("..", "results_smpc", "server_metrics")
os.makedirs(log_dir, exist_ok=True)
csv_file = os.path.join(log_dir, f"server_metrics_{run}.csv")
costs_file = os.path.join(log_dir, f"costs_{run}.csv")
metrics_logger = MetricsLogger(role="server", log_file=costs_file)

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Train Loss", "Train Accuracy", "Eval Loss", "Eval Accuracy"])

with open(f"client_data/global_val_data_{dataname}.pkl", "rb") as f:
    val_data = pickle.load(f)
val_loader = DataLoader(val_data, batch_size=16)

round_history = []
train_acc_history = []
eval_acc_history = []


def centralized_evaluate(parameters):
    split_config = BertSplitConfig(split_layer=5)
    client_model = BertModel_Client(split_config).to(device)
    server_model = BertModel_Server(split_config).to(device)
    client_model.eval()
    server_model.eval()

    weights = fl.common.parameters_to_ndarrays(parameters)

    client_state_dict = client_model.state_dict()
    server_state_dict = server_model.state_dict()

    client_keys = list(client_state_dict.keys())
    server_keys = list(server_state_dict.keys())

    num_client_params = len(client_keys)
    client_weights = weights[:num_client_params]
    server_weights = weights[num_client_params:]

    client_dict = {k: torch.tensor(v) for k, v in zip(client_keys, client_weights)}
    client_model.load_state_dict(client_dict, strict=True)

    server_dict_new = {k: torch.tensor(v) for k, v in zip(server_keys, server_weights)}
    server_state_dict.update(server_dict_new)
    server_model.load_state_dict(server_state_dict, strict=True)

    total_loss = 0.0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            pooled_output = client_model(input_ids, attention_mask)
            outputs = server_model(pooled_output)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"[Central Evaluation] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


class LoggingStrategy(fl.server.strategy.FedAvg):
    def __init__(self, metrics_logger):
        super().__init__(
            evaluate_metrics_aggregation_fn=self.aggregate_client_metrics,
            fit_metrics_aggregation_fn=self.aggregate_fit_metrics,
        )
        self.logger = metrics_logger
        self.last_train_loss = 0.0
        self.last_train_acc = 0.0

    def _model_shapes(self):
        split_config = BertSplitConfig(split_layer=5)
        client_model = BertModel_Client(split_config)
        server_model = BertModel_Server(split_config)
        client_shapes = [v.shape for v in client_model.state_dict().values()]
        server_shapes = [v.shape for v in server_model.state_dict().values()]
        return client_shapes + server_shapes

    def aggregate_client_metrics(self, metrics):
        total_examples = sum(num_examples for num_examples, _ in metrics)
        avg_accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples
        avg_loss = sum(num_examples * m["loss"] for num_examples, m in metrics) / total_examples
        print(f"[Client Evaluation Aggregated] Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        return {"accuracy": avg_accuracy, "loss": avg_loss}

    def aggregate_fit_metrics(self, metrics):
        total_examples = sum(num_examples for num_examples, _ in metrics)
        avg_accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples
        avg_loss = sum(num_examples * m["loss"] for num_examples, m in metrics) / total_examples
        self.last_train_loss = avg_loss
        self.last_train_acc = avg_accuracy
        print(f"[Client Training Aggregated] Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        return {"accuracy": avg_accuracy, "loss": avg_loss}

    def aggregate_fit(self, rnd, results, failures):
        start_time = time.time()

        if not results:
            print("[Server] No client results received.")
            return None, {}

        print(f"[Server] Secure aggregation started for round {rnd}...")

        encrypted_shares = []
        for cid, fit_res in results:
            encrypted_ndarray = fl.common.parameters_to_ndarrays(fit_res.parameters)[0]
            if not isinstance(encrypted_ndarray, np.ndarray) or encrypted_ndarray.dtype != np.uint8:
                raise ValueError(f"[Server] Expected encrypted update as flat np.uint8 array from client {cid}, got {type(encrypted_ndarray)}")

            if "layer_lengths" not in fit_res.metrics:
                raise ValueError(f"[Server] Missing 'layer_lengths' info in metrics from client {cid}")

            layer_lengths = [int(x) for x in ast.literal_eval(fit_res.metrics["layer_lengths"])]

            encrypted_layers = []
            offset = 0
            for length in layer_lengths:
                encrypted_layers.append(encrypted_ndarray[offset:offset + length].tobytes())
                offset += length

            if offset != len(encrypted_ndarray):
                raise ValueError(f"[Server] Total length mismatch when splitting encrypted update from client {cid}")

            encrypted_shares.append(encrypted_layers)

        decrypted_global_update = aggregate_encrypted_updates(encrypted_shares)

        split_config = BertSplitConfig(split_layer=5)
        client_model = BertModel_Client(split_config)
        server_model = BertModel_Server(split_config)
        client_shapes = [v.shape for v in client_model.state_dict().values()]
        server_shapes = [v.shape for v in server_model.state_dict().values()]
        state_shapes = client_shapes + server_shapes

        if len(decrypted_global_update) != len(state_shapes):
            raise ValueError(
                f"Shape mismatch: decrypted update length {len(decrypted_global_update)} != total model params {len(state_shapes)}"
            )

        reshaped_update = [
            decrypted_global_update[i].reshape(state_shapes[i]) for i in range(len(state_shapes))
        ]

        parameters = fl.common.ndarrays_to_parameters(reshaped_update)

        eval_loss, eval_acc = centralized_evaluate(parameters)

        round_history.append(rnd)
        train_acc_history.append(self.last_train_acc)
        eval_acc_history.append(eval_acc)

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([rnd, self.last_train_loss, self.last_train_acc, eval_loss, eval_acc])

        flat_metrics_dict = {}
        for i, (_, res) in enumerate(results):
            if hasattr(res, "metrics"):
                for k, v in res.metrics.items():
                    flat_metrics_dict[f"client{i}_{k}"] = v

        latency, memory, comm_kb = self.logger.log_all(start_time, flat_metrics_dict, event=f"Round_{rnd}")
        print(f"[MetricsLogger] Round {rnd}: Latency={latency:.4f}s | Memory={memory:.2f}MB | Comm={comm_kb:.2f}KB")

        return parameters, {"accuracy": eval_acc}

    def evaluate(self, rnd, parameters):
        loss, accuracy = centralized_evaluate(parameters)
        return loss, {"accuracy": accuracy}


def main():
    if not crypten.is_initialized():
        crypten.init()

    strategy = LoggingStrategy(metrics_logger)
    fl.server.start_server(
        server_address="localhost:8083",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
        grpc_max_message_length=MAX_MSG_SIZE,
    )

    plt.figure()
    plt.plot(round_history, train_acc_history, label="Train Accuracy")
    plt.plot(round_history, eval_acc_history, label="Eval Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Training vs Evaluation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"accuracy_plot_{run}.png")
    print(f"Saved accuracy plot to accuracy_plot_{run}.png")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
