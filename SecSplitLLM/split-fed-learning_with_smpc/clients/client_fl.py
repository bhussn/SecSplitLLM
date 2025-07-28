import GPUtil
import os
import sys 
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import grpc
import numpy as np
import pickle
from torch.utils.data import DataLoader
from models.split_bert_model import BertSplitConfig, BertModel_Client
from grpc_generated import split_pb2, split_pb2_grpc
import flwr as fl
import csv
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from logs.metrics_logger import MetricsLogger
import crypten
import logging
print("Using split_pb2 from:", split_pb2.__file__)

metrics_dir = "metrics"
os.makedirs(metrics_dir, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    filename="client_debug.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MAX_BATCHES_PER_ROUND = 10  # Limit batches per round

crypten.init()
available_gpus = GPUtil.getAvailable(order='memory', limit=1)

class SplitLearningClient(fl.client.NumPyClient):
    def __init__(self, cid, local_epochs):
        self.cid = cid
        self.local_epochs = local_epochs
        self.device = torch.device(f"cuda:{available_gpus[0]}" if available_gpus else "cpu")
        print(f"[Client {self.cid}] Using device: {self.device}")
        split_config = BertSplitConfig(split_layer=5)
        self.model = BertModel_Client(split_config).to(self.device)

        with open(f"client_data/client_{cid}.pkl", "rb") as f:
            full_data = pickle.load(f)
        labels = [example["labels"].item() for example in full_data]
        train_data, val_data = train_test_split(full_data, test_size=0.2, stratify=labels, random_state=42)
        self.trainloader = DataLoader(train_data, batch_size=8, shuffle=True)
        self.valloader = DataLoader(val_data, batch_size=16, shuffle=False)

        self.channel = grpc.insecure_channel("localhost:50052", options=[
            ('grpc.max_send_message_length', 1024 * 1024 * 1024),
            ('grpc.max_receive_message_length', 1024 * 1024 * 1024),
        ])
        self.stub = split_pb2_grpc.SplitLearningServiceStub(self.channel)
        self.logger = MetricsLogger(role=f"client_{self.cid}", log_file=os.path.join(metrics_dir, "client_metrics.csv"))

    def send_activations(self, activations, attention_mask, labels, require_gradients=True):
        logging.debug(f"[Client {self.cid}] Sending activations of shape {activations.shape}")
        # Encrypt activations using CrypTen
        act_np = activations.detach().cpu().numpy()
        mask_np = attention_mask.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy().astype(np.int64)

        encrypted_activations = crypten.cryptensor(act_np, src=0)
        encrypted_mask = crypten.cryptensor(mask_np, src=0)
        encrypted_labels = crypten.cryptensor(labels_np, src=0)

        # Decrypt before sending (simulated secure exchange)
        act_np = encrypted_activations.get_plain_text()
        mask_np = encrypted_mask.get_plain_text()
        labels_np = encrypted_labels.get_plain_text().numpy().astype(np.int64)

        logging.debug('Sending activations to gRPC server')

        request = split_pb2.ActivationRequest(
            activations=act_np.flatten().tolist(),
            shape=list(act_np.shape),
            attention_mask=mask_np.flatten().tolist(),
            mask_shape=list(mask_np.shape),
            labels=labels_np.flatten().tolist()
        )

        start_time = time.time()
        response = self.stub.SendActivations(request)
        self.logger.log_all(start_time, {
            "activations": act_np,
            "mask": mask_np,
            "labels": labels_np
        }, event="SendActivations")
        
        logging.debug('Awaiting gradient response from gRPC server')
        response = self.stub.SendActivations(request)

        if require_gradients:
            grad_np = np.array(response.gradients, dtype=np.float32).reshape(response.shape)
            # Sanity check
            expected_size = np.prod(response.shape)
            if grad_np.size != expected_size:
                raise ValueError(f"[Client {self.cid}] Gradient size mismatch: received {grad_np.size}, expected {expected_size}")
            activation_grads = torch.tensor(grad_np)
            return activation_grads, response.loss, response.accuracy
        else:
            return [], response.loss, response.accuracy

    def fit(self, parameters, config):
        logging.debug(f"[Client {self.cid}] Entering fit()")
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5, weight_decay=0.01)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
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
            logging.debug(f"[Client {self.cid}] Epoch {epoch+1}/{self.local_epochs}")
            for i, batch in enumerate(self.trainloader):
                logging.debug(f"[Client {self.cid}] Processing batch {i}")
                # if i >= MAX_BATCHES_PER_ROUND:
                #     print(f"[Client {self.cid}] Reached batch limit, exiting fit loop.")
                #     break                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device).float()
                labels = batch["labels"].to(self.device)

                activations, mask = self.model(input_ids, attention_mask)
                activation_grads, loss_value, accuracy_value = self.send_activations(activations, mask, labels)

                activations.backward(activation_grads.to(self.device))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                batch_size = input_ids.size(0)
                total_loss += loss_value * batch_size
                total_correct += accuracy_value * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples
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

                activations, mask = self.model(input_ids, attention_mask)
                _, loss_value, accuracy_value = self.send_activations(activations, mask, labels, require_gradients=False)

                batch_size = input_ids.size(0)
                total_loss += loss_value * batch_size
                total_correct += accuracy_value * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        # Log to CSV
        with open(os.path.join(metrics_dir, f"client_{self.cid}_val_metrics.csv"), "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([avg_loss, avg_accuracy])
       
        return avg_loss, total_samples, {"loss": avg_loss, "accuracy": avg_accuracy}

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cid", type=int, help="Client ID")
    parser.add_argument("--local_epochs", type=int, default=3, help="Number of local training epochs")
    args = parser.parse_args()

    fl.client.start_client(
        server_address="localhost:8082",
        client=SplitLearningClient(args.cid, args.local_epochs).to_client(),
        grpc_max_message_length=1024 * 1024 * 1024
    )
