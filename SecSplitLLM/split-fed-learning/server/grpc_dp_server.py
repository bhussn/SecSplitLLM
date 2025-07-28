import signal
import time
import socket
import GPUtil
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from concurrent import futures
import grpc
import numpy as np
import csv

from models.split_bert_model import BertSplitConfig, BertModel_Server
from grpc_generated import split_pb2, split_pb2_grpc
from flwr.common import ndarrays_to_parameters 
from dp_utils.dp_utils import DPAccountant, clip_gradients, add_noise

# Use GPU
available_gpus = GPUtil.getAvailable(order='memory', limit=1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BATCH_SIZE = 8
run = 6
MAX_MSG_SIZE = 2_000_000_000

class SplitLearningService(split_pb2_grpc.SplitLearningServiceServicer):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        split_config = BertSplitConfig(split_layer=5)
        self.model = BertModel_Server(split_config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.dp_accountant = None 
        self.max_norm = 0.106
        self.noise_mult = 0.7

    def SendActivations(self, request, context):
        # Extract metadata
        metadata = dict(context.invocation_metadata())
        client_id = int(metadata.get('client-id', -1))
        dataset_size = int(metadata.get('dataset-size', 0))
        if dataset_size == 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Missing or invalid dataset-size in metadata")
        delta = 1.0 / dataset_size

        # Initialize DP accountant per client/request
        if self.dp_accountant is None or self.dp_accountant.sample_rate != BATCH_SIZE / dataset_size:
            self.dp_accountant = DPAccountant(
                noise_multiplier=self.noise_mult,
                sample_rate=BATCH_SIZE / dataset_size,
                steps=0
            )     

        # Deserialize tensors
        act_shape = tuple(request.shape)
        activations_np = np.array(request.activations, dtype=np.float32).reshape(act_shape)
        activations = torch.tensor(activations_np, dtype=torch.float32, device=self.device, requires_grad=True)

        labels = torch.tensor(list(request.labels), dtype=torch.long).to(self.device)

        # Forward pass: note we only pass activations (pooled output), no attention mask
        outputs = self.model(activations)
        loss = self.criterion(outputs, labels)

        # Compute accuracy
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels).sum().item()
        accuracy = float(correct) / float(labels.size(0))

        # Backward pass
        loss.backward()

        # Apply DP to activation's gradients
        raw_grad = activations.grad.detach()
        clipped_grad = clip_gradients(raw_grad, max_norm=self.max_norm)

        noisy_grad = add_noise(
            clipped_grad,
            noise_multiplier=self.dp_accountant.noise_multiplier,
            max_norm=self.max_norm
        )

        self.dp_accountant.step()
        eps = self.dp_accountant.get_epsilon(delta)
        
        # Log to CSV
        log_dir = os.path.join("..", "results", "dp_metrics")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"dp_log_server_metrics_{run}.csv")
        log_headers = ["step", "client_id", "epsilon", "delta", "loss", "accuracy"]
        log_row = [self.dp_accountant.steps, client_id, eps, delta, loss.item(), accuracy]

        write_headers = not os.path.exists(log_path)
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_headers:
                writer.writerow(log_headers)
            writer.writerow(log_row)

        grad_np = noisy_grad.cpu().numpy().astype(np.float32)

        return split_pb2.GradientResponse(  
            gradients=grad_np.flatten().tolist(),          
            shape=list(grad_np.shape),
            loss=float(loss.item()),
            accuracy=float(accuracy)
        )

def serve_with_retry(max_retries=5, retry_delay=5):
    global grpc_server_instance
    for attempt in range(max_retries):
        try:
            server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=10),
                options=[
                    ('grpc.max_send_message_length', MAX_MSG_SIZE),
                    ('grpc.max_receive_message_length', MAX_MSG_SIZE),
                ]
            )
            grpc_server_instance = server
            split_pb2_grpc.add_SplitLearningServiceServicer_to_server(SplitLearningService(), server)
            server.add_insecure_port('[::]:50057')
            server.start()
            print("[gRPC Server] Started on port 50057")
            server.wait_for_termination()
            break
        except OSError as e:
            print(f"[gRPC Server] Failed to bind port (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(retry_delay)
    else:
        print("[gRPC Server] Failed to start after multiple attempts.")
        sys.exit(1)

def shutdown_handler(signum, frame):
    global grpc_server_instance
    print(f"[gRPC Server] Received shutdown signal ({signum}). Shutting down gracefully...")
    if grpc_server_instance:
        grpc_server_instance.stop(0)
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# if __name__ == "__main__":
#     serve_with_retry()


if __name__ == "__main__":
    try:
        serve_with_retry()
    finally:
        # Save gradient norm histogram
        from dp_utils.dp_utils import GLOBAL_SERVER_GRAD_NORMS
        import matplotlib.pyplot as plt

        if GLOBAL_SERVER_GRAD_NORMS:
            plt.figure(figsize=(8, 5))
            plt.hist(GLOBAL_SERVER_GRAD_NORMS, bins=60, color="salmon", edgecolor="black")
            plt.title("Server Gradient L2 Norms Before Clipping")
            plt.xlabel("L2 Norm")
            plt.ylabel("Frequency")
            plt.grid(True)
            hist_path = f"server_grad_l2_histogram.png"
            plt.savefig(hist_path)
            print(f"[Server] Saved L2 norm histogram to {hist_path}")
        else:
            print("[Server] No gradient norms logged — histogram not generated.")