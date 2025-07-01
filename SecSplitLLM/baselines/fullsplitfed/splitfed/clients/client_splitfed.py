import os
import torch
import torch.nn as nn
import grpc
import numpy as np
from splitfed.models.split_model import BertClient
from splitfed.grpc import split_pb2, split_pb2_grpc
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from splitfed.training.split_trainer import load_data
import traceback  
from sklearn.metrics import f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class FlowerClient(NumPyClient):
    def __init__(self, client_model, trainloader, valloader, local_epochs, grpc_stub):
        self.client_model = client_model
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.grpc_stub = grpc_stub

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_model.to(self.device)

        self.optimizer = torch.optim.SGD(self.client_model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.client_model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = dict(zip(self.client_model.state_dict().keys(), parameters))
        self.client_model.load_state_dict({k: torch.tensor(v) for k, v in params_dict.items()})

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        client_id = config.get("client_id", 0)
        round_num = config.get("round_num", 0)
        comm_cost = self.train(client_id, round_num)
        updated_params = self.get_parameters()
        return updated_params, len(self.trainloader.dataset), {"communication_cost": comm_cost}

    def train(self, client_id, round_num):
        self.client_model.train()
        total_comm_cost = 0
        for epoch in range(self.local_epochs):
            for batch_idx, batch in enumerate(self.trainloader, start=1):
                try:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    smashed = self.client_model(input_ids, attention_mask)
                    smashed.retain_grad()

                    arr = smashed.cpu().detach().numpy()
                    payload = arr.tobytes()
                    activation_shape = list(arr.shape)

                    # Track sent activation size
                    total_comm_cost += len(payload)

                    request = split_pb2.ForwardRequest(
                        activation_bytes=payload,
                        activation_shape=activation_shape,
                        attention_mask=attention_mask.cpu().numpy().flatten().astype(np.float32).tolist(),
                        attention_mask_shape=list(attention_mask.shape),
                        labels=labels.cpu().numpy().astype(np.int32).tolist(),
                        client_id=client_id,
                        round=round_num,
                        batch_id=batch_idx,
                    )

                    response = self.grpc_stub.ForwardPass(request, timeout=200.0)

                    grad_buf = response.grad_smashed_bytes
                    total_comm_cost += len(grad_buf)
                    grad_arr = np.frombuffer(grad_buf, dtype=np.float32).reshape(response.grad_shape).copy()
                    grads = torch.from_numpy(grad_arr).to(self.device)

                    self.optimizer.zero_grad()
                    smashed.backward(grads)
                    self.optimizer.step()

                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"[Client] Exception in training batch {batch_idx}: {e}\nTraceback:\n{tb}")
                    raise  
        return total_comm_cost

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.client_model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.valloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                smashed = self.client_model(input_ids, attention_mask)

                inference_request = split_pb2.InferenceRequest(
                    activation=smashed.cpu().numpy().flatten().astype(np.float32).tolist(),
                    activation_shape=list(smashed.shape),
                    attention_mask=attention_mask.cpu().numpy().flatten().astype(np.float32).tolist(),
                    attention_mask_shape=list(attention_mask.shape),
                )
                inference_response = self.grpc_stub.Inference(inference_request)

                logits = torch.tensor(inference_response.logits, dtype=torch.float32).reshape((smashed.shape[0], -1)).to(self.device)
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        f1 = f1_score(all_labels, all_preds, average="weighted")
        return avg_loss, total_samples, {"loss": avg_loss, "accuracy": accuracy, "f1": f1}


def client_fn(context: Context):
    cut_layer = context.node_config.get("cut-layer", 4)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]

    client_model = BertClient(cut_layer=cut_layer)

    trainloader = load_data(partition_id, num_partitions, split="train")
    valloader = load_data(partition_id, num_partitions, split="validation")

    channel = grpc.insecure_channel(
        "localhost:50051",
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    )
    grpc_stub = split_pb2_grpc.SplitLearningStub(channel)

    return FlowerClient(client_model, trainloader, valloader, local_epochs, grpc_stub).to_client()

app = ClientApp(client_fn)
