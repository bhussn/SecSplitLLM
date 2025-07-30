import sys
import os
import GPUtil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import csv
import pickle
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import flwr as fl
from models.split_bert_LoRA import BertSplitConfig, BertModel_Client, BertModel_Server
import time
from logs.metric_logger import MetricsLogger 

import logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="flower_debug.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

run = 0
dataname = "sst2"
MAX_MSG_SIZE = 2_000_000_000
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging Setup
log_dir = os.path.join("..", "results_smpc", "server_metrics")
os.makedirs(log_dir, exist_ok=True)
csv_file = os.path.join(log_dir, f"server_metrics_{run}.csv")
costs_file = os.path.join(log_dir, f"costs_{run}.csv")
metrics_logger = MetricsLogger(role="server", log_file=costs_file)

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Train Loss", "Train Accuracy", "Eval Loss", "Eval Accuracy"])

# Load Validation Data 
with open(f"client_data/global_val_data_{dataname}.pkl", "rb") as f:
    val_data = pickle.load(f)
val_loader = DataLoader(val_data, batch_size=16)

round_history = []
train_acc_history = []
eval_acc_history = []

# Centralized Evaluation 
def centralized_evaluate(parameters):
    split_config = BertSplitConfig(split_layer=5)
    client_model = BertModel_Client(split_config)
    server_model = BertModel_Server(split_config)

    weights = fl.common.parameters_to_ndarrays(parameters)
    client_items = list(client_model.state_dict().items())
    server_items = list(server_model.state_dict().items())

    client_state = {k: torch.tensor(v) for (k, _), v in zip(client_items, weights[:len(client_items)])}
    server_state = {k: torch.tensor(v) for (k, _), v in zip(server_items, weights[len(client_items):])}

    client_model.load_state_dict(client_state, strict=False)
    server_model.load_state_dict(server_state, strict=False)

    client_model.to(device).eval()
    server_model.to(device).eval()

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

# Logging Strategy 
class LoggingStrategy(fl.server.strategy.FedAvg):
    def __init__(self, metrics_logger):
        super().__init__(
            evaluate_metrics_aggregation_fn=self.aggregate_client_metrics,
            fit_metrics_aggregation_fn=self.aggregate_fit_metrics
        )
        self.logger = metrics_logger
        self.last_train_loss = 0.0
        self.last_train_acc = 0.0

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

        # Extract communication-related data
        flat_metrics_dict = {}
        for i, (_, res) in enumerate(results):
            if hasattr(res, "metrics"):
                for k, v in res.metrics.items():
                    flat_metrics_dict[f"client{i}_{k}"] = v

        aggregated = super().aggregate_fit(rnd, results, failures)

        if aggregated is not None:
            parameters, _ = aggregated
            if parameters is None:
                print("[Server] Warning: Aggregated parameters are None.")
                return aggregated

            eval_loss, eval_acc = centralized_evaluate(parameters)

            round_history.append(rnd)
            train_acc_history.append(self.last_train_acc)
            eval_acc_history.append(eval_acc)

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([rnd, self.last_train_loss, self.last_train_acc, eval_loss, eval_acc])

            # Log costs
            latency, memory, comm_kb = self.logger.log_all(start_time, flat_metrics_dict, event=f"Round_{rnd}")
            print(f"[MetricsLogger] Round {rnd}: Latency={latency:.4f}s | Memory={memory:.2f}MB | Comm={comm_kb:.2f}KB")

        return aggregated

    def evaluate(self, rnd, parameters):
        loss, accuracy = centralized_evaluate(parameters)
        return loss, {"accuracy": accuracy}

# Main 
def main():
    strategy = LoggingStrategy(metrics_logger)
    fl.server.start_server(
        server_address="localhost:8086",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        grpc_max_message_length=MAX_MSG_SIZE
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
    main()
