"""fed-ner: A Flower / PyTorch app."""
# fed_ner/client_app.py

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fed_ner.task import Net, get_weights, load_data, set_weights, test, train


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        # Load global weights into model
        set_weights(self.net, parameters)

        # Train locally
        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)

        # Return updated weights and training stats
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        # Load global weights into model
        set_weights(self.net, parameters)

        # Evaluate on local validation data
        loss, accuracy, f1 = test(self.net, self.valloader, self.device)

        # Return loss, number of validation samples, and metrics expected by server
        return loss, len(self.valloader.dataset), {"accuracy": accuracy, "f1": f1}


def client_fn(context: Context):
    # Load model and partitioned data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloader, valloader = load_data(partition_id, num_partitions)

    local_epochs = context.run_config["local-epochs"]

    # Create and return FlowerClient instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


app = ClientApp(client_fn)
