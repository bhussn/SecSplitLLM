import os
import torch
import grpc
import torch.nn as nn
import numpy as np
from concurrent import futures
import threading
from splitfed.models.split_model import BertServer
from splitfed.grpc import split_pb2, split_pb2_grpc
import traceback
from transformers import get_linear_schedule_with_warmup

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Constant for gradient clipping
MAX_GRAD_NORM = 1.0

class SplitLearningServicer(split_pb2_grpc.SplitLearningServicer):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertServer(cut_layer=4).to(self.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6)
        self.criterion = nn.CrossEntropyLoss()
        self.lock = threading.Lock()  # lock for thread-safe optimizer updates

        samples_per_client = 67349 // 15
        batches_per_epoch = samples_per_client // 2
        steps_per_round = batches_per_epoch * 2 * 7 # (batches * local_epochs * clients_per_round)
        num_training_steps = steps_per_round * 5 # (steps_per_round * num_server_rounds)

        print(f"[Server Scheduler] Estimated total training steps: {num_training_steps}")

        # Scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
            )

    def ForwardPass(self, request, context):
        try:
            # Reconstruct activation tensor
            activation_shape = tuple(request.activation_shape)
            buf = request.activation_bytes
            arr = np.frombuffer(buf, dtype=np.float32).reshape(activation_shape).copy()
            activation = torch.from_numpy(arr).to(self.device).requires_grad_()

            # --- NaN/inf CHECK 1 ---
            if torch.isnan(activation).any() or torch.isinf(activation).any():
                print("!!!!!! [SERVER] WARNING: Received activation with NaN/inf values.")
            
            print("--- [SERVER] Received Activation ---")
            print(f"  - Shape: {activation.shape}, Mean: {activation.mean():.4f}, Std: {activation.std():.4f}")

            # Reconstruct attention_mask tensor from bytes
            attention_mask_shape = tuple(request.attention_mask_shape)
            mask_buf = request.attention_mask_bytes # Use the new field name
            mask_arr = np.frombuffer(mask_buf, dtype=np.int32).reshape(attention_mask_shape).copy()
            attention_mask = torch.from_numpy(mask_arr).to(self.device)

            labels = torch.tensor(request.labels, dtype=torch.long).to(self.device)

            with self.lock:  # ensure only one thread updates model at a time
                self.optimizer.zero_grad()
                logits, loss = self.model(activation, attention_mask, labels)
                loss.backward()
                # Graddient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()
                self.scheduler.step()

            grad_tensor = activation.grad.detach()
            # Manually clip the gradient being sent back to the client
                # --- NaN/inf CHECK 2 ---
            if grad_tensor is not None:
                if torch.isnan(grad_tensor).any() or torch.isinf(grad_tensor).any():
                    print("!!!!!! [SERVER] WARNING: Calculated gradient has NaN/inf values BEFORE clipping.")
                torch.nn.utils.clip_grad_norm_(grad_tensor, MAX_GRAD_NORM)

            grad_arr = grad_tensor.cpu().numpy()
            print("--- [SERVER] Sending Gradient ---")
            print(f"  - Shape: {grad_arr.shape}, Mean: {grad_arr.mean():.4f}, Std: {grad_arr.std():.4f}")
            print("-----------------------------------")

            grad_payload = grad_arr.tobytes()
            grad_shape = list(grad_arr.shape)

            return split_pb2.ForwardReply( # type: ignore
                grad_smashed_bytes=grad_payload,
                grad_shape=grad_shape,
                logits=logits.detach().cpu().view(-1).tolist(),
                logits_shape=list(logits.shape),
                loss=loss.item(),
            )

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[Server] Exception in ForwardPass: {e}\nTraceback:\n{tb}")
            context.set_details(f"{e}\n{tb}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return split_pb2.ForwardReply() # type: ignore

    def Inference(self, request, context):
        try:
            activation = torch.tensor(request.activation, dtype=torch.float32)\
                .reshape(tuple(request.activation_shape))\
                .to(self.device)
            

            attention_mask_shape = tuple(request.attention_mask_shape)
            mask_buf = request.attention_mask_bytes
            mask_arr = np.frombuffer(mask_buf, dtype=np.int32).reshape(attention_mask_shape).copy()
            attention_mask = torch.from_numpy(mask_arr).to(self.device)

            logits, _ = self.model(activation, attention_mask)
            return split_pb2.InferenceReply( # type: ignore
                logits=logits.cpu().view(-1).tolist(),
                logits_shape=list(logits.shape),
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[Server] Exception in Inference: {e}\nTraceback:\n{tb}")
            context.set_details(f"{e}\n{tb}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return split_pb2.InferenceReply() # type: ignore

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=5),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    )
    split_pb2_grpc.add_SplitLearningServicer_to_server(SplitLearningServicer(), server)
    server.add_insecure_port("[::]:50052")
    print("[gRPC Server] Listening on port 50052…")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()