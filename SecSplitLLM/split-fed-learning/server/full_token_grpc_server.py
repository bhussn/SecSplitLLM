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

from models.split_bert_model import BertSplitConfig, BertModel_Server
from grpc_generated import split_pb2, split_pb2_grpc
from flwr.common import ndarrays_to_parameters 

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

available_gpus = GPUtil.getAvailable(order='memory', limit=1)
class SplitLearningService(split_pb2_grpc.SplitLearningServiceServicer):
    def __init__(self):
        # self.device = torch.device(f"cuda:{available_gpus[0]}" if available_gpus else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        split_config = BertSplitConfig(split_layer=5)
        self.model = BertModel_Server(split_config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def SendActivations(self, request, context):
        # Deserialize tensors
        act_shape = tuple(request.shape)
        mask_shape = tuple(request.mask_shape)
        # activations = torch.tensor(request.activations, dtype=torch.float32).view(act_shape).to(self.device).requires_grad_()
        # attention_mask = torch.tensor(request.attention_mask, dtype=torch.float32).view(mask_shape).to(self.device)
        # labels = torch.tensor(request.labels, dtype=torch.long).to(self.device)
        activations_np = np.array(request.activations, dtype=np.float32).reshape(act_shape)
        activations = torch.tensor(activations_np, dtype=torch.float32, device=self.device, requires_grad=True)
        attention_mask_np = np.array(request.attention_mask, dtype=np.float32).reshape(mask_shape)
        attention_mask = torch.tensor(attention_mask_np, dtype=torch.float32, device=self.device)
        labels = torch.tensor(list(request.labels), dtype=torch.long).to(self.device)

        # print(f"[gRPC Server] Received activations with shape: {activations.shape}") # DO_NOT_COMMIT

        # Forward pass
        outputs = self.model(activations, attention_mask)
        loss = self.criterion(outputs, labels)

        # Compute accuracy
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels).sum().item()
        accuracy = correct / labels.size(0)

        # Backward pass
        loss.backward()
        # gradients = activations.grad.detach().cpu().numpy().astype(np.float32).flatten().tolist()
        grad_np = activations.grad.detach().cpu().numpy().astype(np.float32)
        
        # Serialize gradients using Flower's Array serialization
        # grad_parameters = ndarrays_to_parameters([grad_np])

        # print(f"[gRPC Server] Sending gradients of shape: {list(activations.shape)} with total elements: {grad_np.size}")
        # print(f"[gRPC Server] Batch Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}") # DO_NOT_COMMIT

        # Log loss and accuracy
        return split_pb2.GradientResponse(  
            gradients=grad_np.astype(np.float32).flatten().tolist(),          
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
                    ('grpc.max_send_message_length', 2_000_000_000),
                    ('grpc.max_receive_message_length', 2_000_000_000),
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