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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class SplitLearningServicer(split_pb2_grpc.SplitLearningServicer):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertServer(cut_layer=4).to(self.device)
        self.model.train()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.lock = threading.Lock()  # lock for thread-safe optimizer updates

    def ForwardPass(self, request, context):
        try:
            activation_shape = tuple(request.activation_shape)
            buf = request.activation_bytes
            arr = np.frombuffer(buf, dtype=np.float32).reshape(activation_shape).copy()
            activation = torch.from_numpy(arr).to(self.device).requires_grad_()

            attention_mask_shape = tuple(request.attention_mask_shape)
            mask_list = request.attention_mask
            mask_arr = np.array(mask_list, dtype=np.float32).reshape(attention_mask_shape)
            attention_mask = torch.from_numpy(mask_arr).to(self.device)

            labels = torch.tensor(request.labels, dtype=torch.long).to(self.device)

            with self.lock:  # ensure only one thread updates model at a time
                self.optimizer.zero_grad()
                logits, loss = self.model(activation, attention_mask, labels)
                loss.backward()
                self.optimizer.step()

            grad_arr = activation.grad.detach().cpu().numpy()
            grad_payload = grad_arr.tobytes()
            grad_shape = list(grad_arr.shape)

            return split_pb2.ForwardReply(
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
            return split_pb2.ForwardReply()

    def Inference(self, request, context):
        try:
            activation = torch.tensor(request.activation, dtype=torch.float32)\
                        .reshape(tuple(request.activation_shape))\
                        .to(self.device)
            attention_mask = torch.tensor(request.attention_mask, dtype=torch.float32)\
                            .reshape(tuple(request.attention_mask_shape))\
                            .to(self.device)
            logits, _ = self.model(activation, attention_mask)
            return split_pb2.InferenceReply(
                logits=logits.cpu().view(-1).tolist(),
                logits_shape=list(logits.shape),
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[Server] Exception in Inference: {e}\nTraceback:\n{tb}")
            context.set_details(f"{e}\n{tb}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return split_pb2.InferenceReply()

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=5),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    )
    split_pb2_grpc.add_SplitLearningServicer_to_server(SplitLearningServicer(), server)
    server.add_insecure_port("[::]:50051")
    print("[gRPC Server] Listening on port 50051…")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
