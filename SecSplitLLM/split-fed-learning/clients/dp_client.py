
import sys
import os
import GPUtil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import grpc
import numpy as np
import pickle
from torch.utils.data import DataLoader
from models.split_bert_model import BertSplitConfig, BertModel_Client
from grpc_generated import split_pb2, split_pb2_grpc
import flwr as fl
import sys
import csv
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from dp_utils.dp_utils import DPAccountant, clip_and_add_noise


print("Using split_pb2 from:", split_pb2.__file__)

import logging

logging.basicConfig(
    level=logging.DEBUG,
    filename="client_debug.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MAX_BATCHES_PER_ROUND = 8  # Limit batches per round
run = 6
MAX_MSG_SIZE = 2_000_000_000
dataname = "mnli"

# Use GPU
# available_gpus = GPUtil.getAvailable(order='memory', limit=1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class SplitLearningClient(fl.client.NumPyClient):
    def __init__(self, cid, local_epochs):
        self.cid = cid
        self.local_epochs = local_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        split_config = BertSplitConfig(split_layer=5)
        self.model = BertModel_Client(split_config).to(self.device)
        self.epoch_metrics = []

        with open(f"client_data/client_{dataname}_{cid}.pkl", "rb") as f:
            full_data = pickle.load(f)
        labels = [example["labels"].item() for example in full_data]
        train_data, val_data = train_test_split(
            full_data, test_size=0.2, stratify=labels, random_state=42
        )
        self.trainloader = DataLoader(train_data, batch_size=MAX_BATCHES_PER_ROUND, shuffle=True)
        self.valloader = DataLoader(val_data, batch_size=MAX_BATCHES_PER_ROUND, shuffle=False)

        self.channel = grpc.insecure_channel("localhost:50057", options=[
            ('grpc.max_send_message_length', MAX_MSG_SIZE), 
            ('grpc.max_receive_message_length', MAX_MSG_SIZE)
        ])
        self.stub = split_pb2_grpc.SplitLearningServiceStub(self.channel)
        self.batch_size = MAX_BATCHES_PER_ROUND
        self.dataset_size = len(train_data)
        self.noise_multiplier = 0.7
        self.dp_accountant = DPAccountant(
            noise_multiplier=self.noise_multiplier,
            sample_rate=self.batch_size / self.dataset_size,
            steps=0
        )
        self.max_norm = 4.2 #changed for mnli

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def send_activations(self, activations, labels, require_gradients=True):
        dp_activations = clip_and_add_noise(activations, clip_threshold=self.max_norm, noise_multiplier=self.noise_multiplier)
        act_np = dp_activations.detach().cpu().numpy()
        if act_np.size == 0:
            raise ValueError(f"[Client {self.cid}] Activations array is empty! shape={act_np.shape}")

        labels_np = labels.detach().cpu().numpy()
        if labels_np.size == 0:
            raise ValueError(f"[Client {self.cid}] Labels array is empty! shape={labels_np.shape}")

        request = split_pb2.ActivationRequest(
            activations=act_np.flatten().tolist(),
            shape=list(act_np.shape),
            labels=labels_np.flatten().tolist()
        )

        metadata = (
            ('client-id', str(self.cid)),
            ('dataset-size', str(self.dataset_size)),
        )

        response = self.stub.SendActivations(request, metadata=metadata)

        if require_gradients:
            grad_np = np.array(response.gradients, dtype=np.float32)
            expected_size = np.prod(response.shape)
            if grad_np.size != expected_size:
                raise ValueError(f"[Client {self.cid}] Gradient size mismatch: received {grad_np.size}, expected {expected_size}")
            grad_np = grad_np.reshape(response.shape)
            activation_grads = torch.tensor(grad_np)
            return activation_grads, response.loss, response.accuracy
        else:
            return [], response.loss, response.accuracy

    def evaluate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.valloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device).float()
                labels = batch["labels"].to(self.device)
                batch_size = input_ids.size(0)

                activations = self.model(input_ids, attention_mask)
                _, loss_value, accuracy_value = self.send_activations(activations, labels, require_gradients=False)

                total_loss += loss_value * batch_size
                total_correct += accuracy_value * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples
        return avg_loss, avg_accuracy
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * len(self.trainloader)),
            num_training_steps=len(self.trainloader) * self.local_epochs
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(self.local_epochs):
            for i, batch in enumerate(self.trainloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device).float()
                labels = batch["labels"].to(self.device)
                batch_size = input_ids.size(0)

                # print(f"[Client {self.cid}] Epoch {epoch} Batch {i} batch_size = {batch_size}")

                if batch_size == 0:
                    print(f"[Client {self.cid}] Skipping empty batch at epoch {epoch} batch {i}")
                    continue

                activations = self.model(input_ids, attention_mask)

                if activations.numel() == 0:
                    print(f"[Client {self.cid}] Skipping batch with empty activations at epoch {epoch} batch {i}")
                    continue

                activation_grads, loss_value, accuracy_value = self.send_activations(activations, labels)
                activations.backward(activation_grads.to(self.device))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss_value * batch_size
                total_correct += accuracy_value * batch_size
                total_samples += batch_size
                self.dp_accountant.step()

# epoch tracking ==
                # epoch_loss += loss_value * batch_size
                # epoch_correct += accuracy_value * batch_size
                # epoch_samples += batch_size

            # Compute averages for the epoch
            # avg_train_loss = epoch_loss / epoch_samples
            # avg_train_acc = epoch_correct / epoch_samples

            # Run validation
            val_loss, val_acc = self.evaluate_epoch()

            # Append all metrics to list
            # self.epoch_metrics.append([
            #     epoch, avg_train_loss, avg_train_acc, val_loss, val_acc
            # ])

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        delta = 1 / self.dataset_size
        epsilon = self.dp_accountant.get_epsilon(delta)

        log_dir = os.path.join("..", "results", "dp_metrics")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"client_{self.cid}_{dataname}_dp_metrics_{run}.csv")
        file_exists = os.path.exists(log_file)

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["epsilon", "delta", "loss", "accuracy", "step"])
            writer.writerow([epsilon, delta, avg_loss, avg_accuracy, self.dp_accountant.steps])

        # # Write all epoch metrics at once
        # metrics_dir = os.path.join("..", "results", "epoch_metrics")
        # os.makedirs(metrics_dir, exist_ok=True)
        # metrics_file = os.path.join(metrics_dir, f"client_{self.cid}_{dataname}_epoch_metrics_{run}.csv")
        # file_exists = os.path.exists(metrics_file)

        # with open(metrics_file, "a", newline="") as f:
        #     writer = csv.writer(f)
        #     if not file_exists:
        #         writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        #     writer.writerows(self.epoch_metrics)
        

        return self.get_parameters(), total_samples, {"loss": avg_loss, "accuracy": avg_accuracy}


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.trainloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device).float()
                labels = batch["labels"].to(self.device)
                batch_size = input_ids.size(0)

                activations = self.model(input_ids, attention_mask)
                _, loss_value, accuracy_value = self.send_activations(activations, labels, require_gradients=False)

                total_loss += loss_value * batch_size
                total_correct += accuracy_value * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        log_dir = os.path.join("..", "results", "client_metrics")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"client_{self.cid}_{dataname}_val_metrics_{run}.csv")
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([avg_loss, avg_accuracy])

        return avg_loss, total_samples, {"loss": avg_loss, "accuracy": avg_accuracy}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cid", type=int, help="Client ID")
    parser.add_argument("--local_epochs", type=int, default=3, help="Number of local training epochs")
    args = parser.parse_args()

    fl.client.start_client(
        server_address="localhost:8087",
        client=SplitLearningClient(args.cid, args.local_epochs).to_client(),
        grpc_max_message_length=MAX_MSG_SIZE
    )

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("cid", type=int, help="Client ID")
#     parser.add_argument("--local_epochs", type=int, default=3, help="Number of local training epochs")
#     args = parser.parse_args()

#     # fl.client.start_client(
#     #     server_address="localhost:8087",
#     #     client=SplitLearningClient(args.cid, args.local_epochs).to_client(),
#     #     grpc_max_message_length=MAX_MSG_SIZE
#     # )

#     try:
#         fl.client.start_client(
#             server_address="localhost:8087",
#             client=SplitLearningClient(args.cid, args.local_epochs).to_client(),
#             grpc_max_message_length=MAX_MSG_SIZE
#         )
#     finally:
#         # Plot and save histogram of all collected norms
#         from dp_utils.dp_utils import GLOBAL_NORMS
#         import matplotlib.pyplot as plt
#         if GLOBAL_NORMS:
#             plt.figure(figsize=(8, 5))
#             plt.hist(GLOBAL_NORMS, bins=60, color="skyblue", edgecolor="black")
#             plt.title("Histogram of L2 Norms Before Clipping")
#             plt.xlabel("L2 Norm")
#             plt.ylabel("Frequency")
#             plt.grid(True)
#             hist_path = f"client_{args.cid}_l2_norm_histogram.png"
#             plt.savefig(hist_path)
#             print(f"[Client {args.cid}] Saved L2 norm histogram to {hist_path}")
#         else:
#             print(f"[Client {args.cid}] No norms were logged, histogram not generated.")

