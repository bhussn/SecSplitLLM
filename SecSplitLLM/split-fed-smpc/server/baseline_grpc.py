import signal
import time
import socket
import sys
import os
import csv
import torch
import torch.nn as nn
from concurrent import futures
import grpc
import numpy as np
import GPUtil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.split_bert_LoRA import BertSplitConfig, BertModel_Server
from server.grpc_generated import split_pb2, split_pb2_grpc

# Use GPU
available_gpus = GPUtil.getAvailable(order='memory', limit=1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
BATCH_SIZE = 8
run = 0
MAX_MSG_SIZE = 2_000_000_000

class SplitLearningService(split_pb2_grpc.SplitLearningServiceServicer):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        split_config = BertSplitConfig(split_layer=5)
        self.model = BertModel_Server(split_config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def SendActivations(self, request, context):
        # Extract metadata
        metadata = dict(context.invocation_metadata())
        client_id = int(metadata.get('client-id', -1))
        dataset_size = int(metadata.get('dataset-size', 0))
        if dataset_size == 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Missing or invalid dataset-size in metadata")

        # Deserialize tensors
        act_shape = tuple(request.shape)
        activations_np = np.array(request.activations, dtype=np.float32).reshape(act_shape)
        activations = torch.tensor(activations_np, dtype=torch.float32, device=self.device, requires_grad=True)

        labels = torch.tensor(list(request.labels), dtype=torch.long).to(self.device)

        # Forward pass
        outputs = self.model(activations)
        loss = self.criterion(outputs, labels)

        # Compute accuracy
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels).sum().item()
        accuracy = float(correct) / float(labels.size(0))

        # Backward pass
        loss.backward()

        # Use raw gradients directly (no DP)
        raw_grad = activations.grad.detach()
        grad_np = raw_grad.cpu().numpy().astype(np.float32)

        # Log to CSV (baseline metrics)
        log_dir = os.path.join("..", "results", "dp_metrics")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"baseline_log_server_metrics_{run}.csv")
        log_headers = ["client_id", "loss", "accuracy"]
        log_row = [client_id, loss.item(), accuracy]

        write_headers = not os.path.exists(log_path)
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_headers:
                writer.writerow(log_headers)
            writer.writerow(log_row)

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
            server.add_insecure_port('[::]:50053')
            server.start()
            print("[gRPC Server] Started on port 50053")
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

if __name__ == "__main__":
    serve_with_retry()