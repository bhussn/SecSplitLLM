
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
run = 12
MAX_MSG_SIZE = 2_000_000_000
dataname = "sst2"

# Use GPU
# available_gpus = GPUtil.getAvailable(order='memory', limit=1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class SplitLearningClient(fl.client.NumPyClient):
    def __init__(self, cid, local_epochs):
        self.cid = cid
        self.local_epochs = local_epochs
        # self.device = torch.device(f"cuda:{available_gpus[0]}" if available_gpus else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.device)
        split_config = BertSplitConfig(split_layer=5)
        self.model = BertModel_Client(split_config).to(self.device)
        from sklearn.model_selection import train_test_split
        with open(f"client_data/client_{dataname}_{cid}.pkl", "rb") as f:
            full_data = pickle.load(f)
        labels = [example["labels"].item() for example in full_data]
        train_data, val_data = train_test_split(
            full_data, test_size=0.2, stratify=labels, random_state=42
        )
        self.trainloader = DataLoader(train_data, batch_size=MAX_BATCHES_PER_ROUND, shuffle=True)
        self.valloader = DataLoader(val_data, batch_size=MAX_BATCHES_PER_ROUND, shuffle=False)
        self.channel = grpc.insecure_channel("localhost:50058",
                                                options=[
                                                            ('grpc.max_send_message_length', MAX_MSG_SIZE), 
                                                            ('grpc.max_receive_message_length', MAX_MSG_SIZE), 
                                                        ]
                                            )
        self.stub = split_pb2_grpc.SplitLearningServiceStub(self.channel)
        self.batch_size = MAX_BATCHES_PER_ROUND
        self.dataset_size = len(train_data)
        self.noise_multiplier = 0.5
        self.dp_accountant = DPAccountant(
            noise_multiplier=self.noise_multiplier,
            sample_rate=self.batch_size / self.dataset_size,
            steps=0
        )
        self.max_norm = 150.0 # Closer to the actual activation norms

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def send_activations(self, activations, attention_mask, labels, require_gradients=True):
        # Apply DP to activations
        dp_activations = clip_and_add_noise(activations, clip_threshold=self.max_norm, noise_multiplier=self.noise_multiplier)

        act_np = dp_activations.detach().cpu().numpy()
        mask_np = attention_mask.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        # print(f"[Client {self.cid}] Sending activations with shape: {act_np.shape}") # DO_NOT_COMMIT
        request = split_pb2.ActivationRequest(
            activations=act_np.flatten().tolist(),
            shape=list(act_np.shape),
            attention_mask=mask_np.flatten().tolist(),
            mask_shape=list(mask_np.shape),
            labels=labels_np.flatten().tolist()
        )

        metadata = (
            ('client-id', str(self.cid)),
            ('dataset-size', str(self.dataset_size)),
        )

        response = self.stub.SendActivations(request, metadata=metadata)

        if require_gradients:
            # print(f"[Client {self.cid}] Received gradient size: {len(response.gradients)}, expected shape: {np.prod(response.shape)}") # DO_NOT_COMMIT
            # grad_np = np.frombuffer(response.gradients, dtype=np.float32).reshape(list(response.shape))
            # grad_np = np.array(response.gradients, dtype=np.float32).reshape(list(response.shape))

            # Deserialize the gradient bytes into a NumPy array
            grad_np = np.array(response.gradients, dtype=np.float32)

            # Sanity check
            expected_size = np.prod(response.shape)
            if grad_np.size != expected_size:
                raise ValueError(f"[Client {self.cid}] Gradient size mismatch: received {grad_np.size}, expected {expected_size}")

            # Reshape and convert to torch tensor
            grad_np = grad_np.reshape(response.shape)
            activation_grads = torch.tensor(grad_np)

            return activation_grads, response.loss, response.accuracy
            # return torch.tensor(grad_np), response.loss, response.accuracy
        else:
            return [], response.loss, response.accuracy

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5, weight_decay=0.01)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
        # total_steps = min(MAX_BATCHES_PER_ROUND, len(self.trainloader))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * len(self.trainloader)),
            num_training_steps=len(self.trainloader) * self.local_epochs
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(self.local_epochs):
            # print(f"[Client {self.cid}] Epoch {epoch+1}/{self.local_epochs}")
            for i, batch in enumerate(self.trainloader):
                # print(f"[Client {self.cid}] Processing batch {i}")
                # if i >= MAX_BATCHES_PER_ROUND:
                #     print(f"[Client {self.cid}] Reached batch limit, exiting fit loop.")
                #     break
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device).float()
                labels = batch["labels"].to(self.device)
                batch_size = input_ids.size(0)
                activations, mask = self.model(input_ids, attention_mask)
                activation_grads, loss_value, accuracy_value = self.send_activations(activations, mask, labels)
                activations.backward(activation_grads.to(self.device))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss_value * batch_size
                total_correct += accuracy_value * batch_size
                total_samples += batch_size
                self.dp_accountant.step()

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        delta = 1 / self.dataset_size
        epsilon = self.dp_accountant.get_epsilon(delta)
        # print(f"[Client {self.cid}] DP ε = {epsilon:.4f}, δ = {delta:.1e}")
        # Log DP to CSV
        log_dir = os.path.join("..", "results", "dp_metrics")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"client_{self.cid}_{dataname}_dp_metrics_{run}.csv")
        file_exists = os.path.exists(log_file)

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["epsilon", "delta", "loss", "accuracy", "step"])
            writer.writerow([epsilon, delta, avg_loss, avg_accuracy, self.dp_accountant.steps])

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
                activations, mask = self.model(input_ids, attention_mask)
                _, loss_value, accuracy_value = self.send_activations(activations, mask, labels, require_gradients=False)
                total_loss += loss_value * batch_size
                total_correct += accuracy_value * batch_size
                total_samples += batch_size
        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        # Log to file
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
        server_address="localhost:8088",
        client=SplitLearningClient(args.cid, args.local_epochs).to_client(),
        grpc_max_message_length=MAX_MSG_SIZE
    )

