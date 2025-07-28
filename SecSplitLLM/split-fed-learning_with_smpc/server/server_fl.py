import sys
import os
import time
import GPUtil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import csv
import pickle
import torch
import matplotlib.pyplot as plt
import argparse
import flwr as fl
from torch.utils.data import DataLoader
from models.split_bert_model import BertSplitConfig, BertModel_Client, BertModel_Server
from smpc.smpc_aggregator import secure_aggregation
from logs.metrics_logger import MetricsLogger


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Federated Server with SMPC Aggregation")
parser.add_argument("--smpc_scheme", type=str, default="additive", choices=["additive", "shamir"],
                    help="Secret sharing scheme to use for SMPC aggregation")
parser.add_argument("--num_rounds", type=int, default=3,
                    help="Number of federated learning rounds")
args = parser.parse_args()

# Setup logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="flower_debug.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load global validation data
with open("client_data/global_val_data.pkl", "rb") as f:
    val_data = pickle.load(f)
val_loader = DataLoader(val_data, batch_size=8)

available_gpus = GPUtil.getAvailable(order='memory', limit=1)
device = torch.device(f"cuda:{available_gpus[0]}" if available_gpus else "cpu")

# CSV file for logging
metrics_dir = "metrics"
os.makedirs(metrics_dir, exist_ok=True)
csv_file = os.path.join(metrics_dir, "server_metrics.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Train Loss", "Train Accuracy", "Eval Loss", "Eval Accuracy"])

# History for plotting
round_history = []
train_acc_history = []
eval_acc_history = []

logger = MetricsLogger(role="flower_server", log_file=os.path.join(metrics_dir, "flower_server_metrics.csv"))
log_file=os.path.join(metrics_dir, "client_metrics.csv")
def centralized_evaluate(parameters):
    start_time = time.time()

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

            activations, mask = client_model(input_ids, attention_mask)
            outputs = server_model(activations, mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    logging.debug(f"[Central Evaluation] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    logger.log_all(start_time, {"parameters": parameters}, event="centralized_evaluation")
    return avg_loss, accuracy

class LoggingStrategy(fl.server.strategy.FedAvg):
    def __init__(self, smpc_scheme="additive"):
        super().__init__(
            evaluate_metrics_aggregation_fn=self.aggregate_client_metrics,
            fit_metrics_aggregation_fn=self.aggregate_fit_metrics
        )
        self.best_eval_acc = 0.0
        self.no_improve_rounds = 0
        self.early_stop_patience = 2
        self.last_train_loss = 0.0
        self.last_train_acc = 0.0
        self.initial_parameters = None  # Will be set after first round
        self.smpc_scheme = smpc_scheme

    def aggregate_client_metrics(self, metrics):
        total_examples = sum(num_examples for num_examples, _ in metrics)
        avg_accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples
        avg_loss = sum(num_examples * m["loss"] for num_examples, m in metrics) / total_examples
        logging.debug(f"[Client Evaluation Aggregated] Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        return {"accuracy": avg_accuracy, "loss": avg_loss}

    def aggregate_fit_metrics(self, metrics):
        total_examples = sum(num_examples for num_examples, _ in metrics)
        avg_accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples
        avg_loss = sum(num_examples * m["loss"] for num_examples, m in metrics) / total_examples
        self.last_train_loss = avg_loss
        self.last_train_acc = avg_accuracy
        logging.debug(f"[Client Training Aggregated] Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        secure_aggregation()  # SMPC aggregation step
        return {"accuracy": avg_accuracy, "loss": avg_loss}

    def aggregate_fit(self, rnd, results, failures):
        start_time = time.time()
        logging.debug(f"[Server] Aggregating fit results for round {rnd}")
        
        # Convert client updates to NumPy arrays
        client_updates = [fl.common.parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        if not client_updates:
            logging.error("[Server] No updates received from clients. Skipping aggregation.")
            return self.initial_parameters, {}

        # Secure aggregation
        aggregated_model = secure_aggregation(client_updates, scheme=args.smpc_scheme, world_size=len(client_updates))
        if aggregated_model is None:
            logging.error("[Server] secure_aggregation returned None. Skipping round.")
            return self.initial_parameters, {}

        # Convert back to Flower Parameters
        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_model)

        logger.log_all(start_time, {"parameters": aggregated_parameters}, event=f"aggregate_fit_round_{rnd}")

        if self.initial_parameters is None:
            self.initial_parameters = aggregated_parameters

        eval_loss, eval_acc = centralized_evaluate(aggregated_parameters)
        round_history.append(rnd)
        train_acc_history.append(self.last_train_acc)
        eval_acc_history.append(eval_acc)

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([rnd, self.last_train_loss, self.last_train_acc, eval_loss, eval_acc])

        # # Early stopping
        # if eval_acc > self.best_eval_acc:
        #     self.best_eval_acc = eval_acc
        #     self.no_improve_rounds = 0
        # else:
        #     self.no_improve_rounds += 1
        #     if self.no_improve_rounds >= self.early_stop_patience:
        #         print(f"[Early Stopping] No improvement for {self.early_stop_patience} rounds. Stopping training.")
        #         raise SystemExit(0)
        return aggregated_parameters, {}

    def evaluate(self, rnd, parameters):
        loss, accuracy = centralized_evaluate(parameters)
        return loss, {"accuracy": accuracy}

def main():
    logging.debug("[Server] Starting Flower server")
    strategy = LoggingStrategy(smpc_scheme=args.smpc_scheme)
    fl.server.start_server(
        server_address="localhost:8082",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        grpc_max_message_length=1024 * 1024 * 1024
    ) 

if __name__ == "__main__":
    main()