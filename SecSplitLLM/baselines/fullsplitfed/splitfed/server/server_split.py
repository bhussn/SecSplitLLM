# import multiprocessing as mp
# mp.set_start_method("spawn", force=True)
import torch
import os
import grpc 
import torch.nn as nn
from concurrent import futures
from splitfed.models.split_model import BertServer
from splitfed.grpc import split_pb2, split_pb2_grpc

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class SplitLearningServicer(split_pb2_grpc.SplitLearningServicer):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertServer(cut_layer=4).to(self.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def ForwardPass(self, request, context):
        try:
            activation_shape = tuple(request.activation_shape)
            attention_mask_shape = tuple(request.attention_mask_shape)

            activation = (
                torch.tensor(request.activation, dtype=torch.float32)
                .reshape(activation_shape)
                .to(self.device)
                .clone()
                .detach()
                .requires_grad_(True)
            )
            # activation.retain_grad() 
            attention_mask = torch.tensor(request.attention_mask, dtype=torch.float32).reshape(attention_mask_shape).to(self.device)
            labels = torch.tensor(request.labels, dtype=torch.long).to(self.device)

            self.optimizer.zero_grad()
            logits, loss = self.model(activation, attention_mask, labels)

            loss.backward()
            self.optimizer.step()
            
            grad_smashed = activation.grad.detach().cpu().view(-1).tolist()
            grad_shape = list(activation.shape)

            return split_pb2.ForwardReply(
                logits=logits.detach().cpu().view(-1).tolist(),
                logits_shape=list(logits.shape),
                loss=loss.item(),
                grad_smashed=grad_smashed,
                grad_shape=grad_shape,
            )
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return split_pb2.ForwardReply()

    def Inference(self, request, context):
        self.model.eval()
        with torch.no_grad():
            activation_shape = tuple(request.activation_shape)
            attention_mask_shape = tuple(request.attention_mask_shape)

            activation = torch.tensor(request.activation, dtype=torch.float32).reshape(activation_shape).to(self.device)
            attention_mask = torch.tensor(request.attention_mask, dtype=torch.float32).reshape(attention_mask_shape).to(self.device)

            logits, _ = self.model(activation, attention_mask)

        return split_pb2.InferenceReply(
            logits=logits.cpu().view(-1).tolist(),
            logits_shape=list(logits.shape)
        )

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),      
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),   
        ]
    )
    servicer = SplitLearningServicer()
    split_pb2_grpc.add_SplitLearningServicer_to_server(servicer, server)
    server.add_insecure_port("[::]:50051")
    print("[gRPC Server] Listening on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
