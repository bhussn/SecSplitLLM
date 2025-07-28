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
import crypten  
import logging

from models.split_bert_model import BertSplitConfig, BertModel_Server
from grpc_generated import split_pb2, split_pb2_grpc
from logs.metrics_logger import MetricsLogger 


logging.basicConfig(
    level=logging.DEBUG,
    filename="grpc_debug.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

crypten.init()
available_gpus = GPUtil.getAvailable(order='memory', limit=1)

metrics_dir = "metrics"
os.makedirs(metrics_dir, exist_ok=True)
logger = MetricsLogger(role="server", log_file=os.path.join(metrics_dir, "server_metrics.csv"))
class SplitLearningService(split_pb2_grpc.SplitLearningServiceServicer):
    def __init__(self):
        self.device = torch.device(f"cuda:{available_gpus[0]}" if available_gpus else "cpu")
        split_config = BertSplitConfig(split_layer=5)
        self.model = BertModel_Server(split_config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def SendActivations(self, request, context):
        start_time = time.time()

        # Deserialize tensors
        act_shape = tuple(request.shape)
        mask_shape = tuple(request.mask_shape)

        activations_np = np.array(request.activations, dtype=np.float32).reshape(act_shape)
        attention_mask_np = np.array(request.attention_mask, dtype=np.float32).reshape(mask_shape)
        labels_np = np.array(request.labels, dtype=np.int64)

        # Convert to torch tensors
        activations = torch.tensor(activations_np, dtype=torch.float32, device=self.device, requires_grad=True)
        attention_mask = torch.tensor(attention_mask_np, dtype=torch.float32, device=self.device)
        labels = torch.tensor(labels_np, dtype=torch.long, device=self.device)

        # Encrypt activations using CrypTen
        encrypted_activations = crypten.cryptensor(activations.detach().cpu().numpy(), src=0)
        encrypted_activations.requires_grad = True

        # Simulate secure forward pass (decrypt for local processing)
        decrypted_activations = encrypted_activations.get_plain_text().to(self.device)
        decrypted_activations.requires_grad = True

        logging.debug(f"[gRPC Server] Running forward pass with received activations with shape: {activations.shape}")

        # Forward pass
        outputs = self.model(decrypted_activations, attention_mask)
        loss = self.criterion(outputs, labels)

        # Compute accuracy
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels).sum().item()
        accuracy = correct / labels.size(0)

        logging.debug(f"[gRPC Server] Running backward pass with batch Loss: {loss.item():.4f}") 
        
        # Backward pass
        loss.backward()
        grad_np = decrypted_activations.grad.detach().cpu().numpy().astype(np.float32)

        logging.debug(f"[gRPC Server] Sending gradients of shape: {list(activations.shape)} with total elements: {grad_np.size}, Accuracy: {accuracy:.4f}")

        # Encrypt gradients before sending (simulated)
        encrypted_grads = crypten.cryptensor(grad_np, src=0)
        decrypted_grads = encrypted_grads.get_plain_text()

        # Log metrics
        logger.log_all(start_time, {"gradients": grad_np}, event="SendActivations")
        
        return split_pb2.GradientResponse(
            gradients=decrypted_grads.flatten().tolist(),
            shape=list(decrypted_grads.shape),
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
                    ('grpc.max_send_message_length', 1024 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 1024 * 1024 * 1024),
                ]
            )
            grpc_server_instance = server
            split_pb2_grpc.add_SplitLearningServiceServicer_to_server(SplitLearningService(), server)
            server.add_insecure_port('[::]:50052')
            server.start()
            print("[gRPC Server] Started on port 50052")
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
