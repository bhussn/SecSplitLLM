# FILE: client_split.py

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import os

from splitfed.models.split_model import BertClient, BertServer
from splitfed.training.split_trainer import load_data, train, test, get_weights
from splitfed.training.utils import weights_dict_to_list, set_model_weights, weights_list_to_dict
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class FlowerClient(NumPyClient):
    def __init__(self, client_model, server_model, trainloader, valloader, local_epochs):
        self.client_model = client_model
        self.server_model = server_model
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_model.to(self.device)
        self.server_model.to(self.device)

    def fit(self, parameters, config):
        # Load server-passed parameters into the client model
        weights_dict = weights_list_to_dict(parameters, self.client_model)
        set_model_weights(self.client_model, weights_dict)

        round_num = config.get("server_round", -1)

        training_loss = None
        avg_fwd = None
        avg_bwd = None
        for _ in range(self.local_epochs):
            training_loss, avg_fwd, avg_bwd = train(
                self.client_model, self.server_model, self.trainloader, self.device, round_num=round_num
            )

        updated_weights = get_weights(self.client_model)
        weights_list = weights_dict_to_list(updated_weights)

        metrics = {
            "train_loss": training_loss,
            "forward_time": avg_fwd,
            "backward_time": avg_bwd,
        }

        return weights_list, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        # Load server-passed parameters into the client model
        weights_dict = weights_list_to_dict(parameters, self.client_model)
        set_model_weights(self.client_model, weights_dict)

        loss, accuracy, f1 = test(self.client_model, self.server_model, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {
            "loss": loss,
            "accuracy": accuracy,
            "f1": f1,
        }


def client_fn(context: Context):
    cut_layer = context.node_config.get("cut-layer", 4)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]

    client_model = BertClient(cut_layer=cut_layer)
    server_model = BertServer(cut_layer=cut_layer)

    trainloader = load_data(partition_id, num_partitions, split="train")
    valloader = load_data(partition_id, num_partitions, split="validation")

    return FlowerClient(client_model, server_model, trainloader, valloader, local_epochs).to_client()


app = ClientApp(client_fn)
