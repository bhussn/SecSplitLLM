import sys
import os
import GPUtil
import logging
import torch
import grpc
import numpy as np
import pickle
import csv
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.utils.data import DataLoader
from models.split_bert_LoRA import BertSplitConfig, BertModel_Client
from grpc_generated import split_pb2, split_pb2_grpc
import flwr as fl
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(
    level=logging.DEBUG,
    filename="client_debug.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MAX_BATCHES_PER_ROUND = 8
run = 0
MAX_MSG_SIZE = 2_000_000_000
dataname = "sst2"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class SplitLearningClient(fl.client.NumPyClient):
    def __init__(self, cid, local_epochs):
        self.cid = cid
        self.local_epochs = local_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        split_config = BertSplitConfig(split_layer=5)
        self.model = BertModel_Client(split_config).to(self.device)

        with open(f"client_data/client_{dataname}_{cid}.pkl", "rb") as f:
            full_data = pickle.load(f)
        labels = [example["labels"].item() for example in full_data]
        train_data, val_data = train_test_split(
            full_data, test_size=0.2, stratify=labels, random_state=42
        )
        self.trainloader = DataLoader(train_data, batch_size=MAX_BATCHES_PER_ROUND, shuffle=True)
        self.valloader = DataLoader(val_data, batch_size=MAX_BATCHES_PER_ROUND, shuffle=False)

        self.channel = grpc.insecure_channel("localhost:50056", options=[
            ('grpc.max_send_message_length', MAX_MSG_SIZE),
            ('grpc.max_receive_message_length', MAX_MSG_SIZE)
        ])
        self.stub = split_pb2_grpc.SplitLearningServiceStub(self.channel)
        self.batch_size = MAX_BATCHES_PER_ROUND
        self.dataset_size = len(train_data)
        self.max_norm = 7.5

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def send_activations(self, activations, labels, require_gradients=True):
        act_np = activations.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        if act_np.size == 0 or labels_np.size == 0:
            raise ValueError(f"[Client {self.cid}] Empty input for activations or labels.")

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
            grad_np = grad_np.reshape(response.shape)
            activation_grads = torch.tensor(grad_np)
            return activation_grads, response.loss, response.accuracy
        else:
            return [], response.loss, response.accuracy

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

                if batch_size == 0:
                    continue

                activations = self.model(input_ids, attention_mask)
                if activations.numel() == 0:
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

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        # Log baseline metrics
        log_dir = os.path.join("..", "results", "client_metrics")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"client_{self.cid}_{dataname}_train_baseline_{run}.csv")
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([avg_loss, avg_accuracy])

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
        log_file = os.path.join(log_dir, f"client_{self.cid}_{dataname}_val_baseline_{run}.csv")
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([avg_loss, avg_accuracy])

        return avg_loss, total_samples, {"loss": avg_loss, "accuracy": avg_accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cid", type=int, help="Client ID")
    parser.add_argument("--local_epochs", type=int, default=3, help="Number of local training epochs")
    args = parser.parse_args()

    fl.client.start_client(
        server_address="localhost:8086",
        client=SplitLearningClient(args.cid, args.local_epochs).to_client(),
        grpc_max_message_length=MAX_MSG_SIZE
    )
