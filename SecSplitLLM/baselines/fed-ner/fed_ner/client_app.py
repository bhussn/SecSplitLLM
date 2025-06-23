# fed_ner/client_app.py

import torch
import logging
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fed_ner.task import get_weights, set_weights, train, test, load_data
from fed_ner.ner_model import Net

logger = logging.getLogger(__name__)

class FlowerClient(NumPyClient):
    """
    Federated learning client that handles local training and evaluation for the NER model.
    """

    def __init__(self, net, trainloader, valloader, local_epochs):
        """
        Initialize the client with model, data loaders, and local training configuration.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net.to(self.device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs

        logger.debug(f"[INIT] Using device: {self.device} | CUDA available: {torch.cuda.is_available()}")

    def fit(self, parameters, config):
        """
        Train the model locally using provided weights and return updated weights and metrics.
        """
        set_weights(self.net, parameters)
        loss = train(self.net, self.trainloader, self.local_epochs, self.device)
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": loss}

    def evaluate(self, parameters, config):
        """
        Evaluate the model locally and return loss, number of samples, and metrics.
        """
        set_weights(self.net, parameters)
        loss, acc, f1 = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {
            "val_loss": loss,
            "val_accuracy": acc,
            "val_f1": f1,
        }

def client_fn(context: Context):
    """
    Instantiate and return a federated Flower client for a specific data partition.
    """
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]

    net = Net()
    trainloader, valloader = load_data(partition_id, num_partitions)
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()

# Initialize the Flower ClientApp
app = ClientApp(client_fn)
