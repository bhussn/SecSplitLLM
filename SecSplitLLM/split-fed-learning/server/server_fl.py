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
from models.split_bert_model import BertSplitConfig, BertModel_Client, BertModel_Server

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
csv_file = "server_metrics.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Train Loss", "Train Accuracy", "Eval Loss", "Eval Accuracy"])

# History for plotting
round_history = []
train_acc_history = []
eval_acc_history = []

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

            activations, mask = client_model(input_ids, attention_mask)
            outputs = server_model(activations, mask)
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
    def __init__(self):
        super().__init__(
            evaluate_metrics_aggregation_fn=self.aggregate_client_metrics,
            fit_metrics_aggregation_fn=self.aggregate_fit_metrics
        )
        self.best_eval_acc = 0.0
        self.no_improve_rounds = 0
        self.early_stop_patience = 2
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
        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated is not None:
            parameters, _ = aggregated
            if parameters is None:
                print("[Server] Warning: Aggregated parameters are None.")
                return aggregated
            eval_loss, eval_acc = centralized_evaluate(parameters)

            # Log to CSV
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
        return aggregated

    def evaluate(self, rnd, parameters):
        loss, accuracy = centralized_evaluate(parameters)
        return loss, {"accuracy": accuracy}


def main():
    strategy = LoggingStrategy()
    fl.server.start_server(
        server_address="localhost:8081",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        grpc_max_message_length=1024 * 1024 * 1024
    )

    # Plot accuracy
    plt.figure()
    plt.plot(round_history, train_acc_history, label="Train Accuracy")
    plt.plot(round_history, eval_acc_history, label="Eval Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Training vs Evaluation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_plot.png")
    print("Saved accuracy plot to accuracy_plot.png")

if __name__ == "__main__":
    main()