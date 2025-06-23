# fed_ner/server_app.py

import logging
import torch
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context, ndarrays_to_parameters
from fed_ner.task import get_weights, set_weights, load_data, test
from fed_ner.ner_model import Net
from fed_ner.metrics import init_csv, append_csv
from fed_ner.fedavg_strategy import CustomFedAvg

logger = logging.getLogger(__name__)

def server_fn(context: Context) -> ServerAppComponents:
    """
    Create and configure the federated learning server components.
    Includes centralized evaluation using validation data.
    """
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Load centralized validation data
    _, val_loader = load_data(partition_id=0, num_partitions=1)

    # Initialize model and weights
    net = Net()
    parameters = ndarrays_to_parameters(get_weights(net))

    # Define centralized evaluation function
    def evaluate(server_round, parameters_ndarray, _config):
        net = Net()
        set_weights(net, parameters_ndarray)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)
        loss, precision, recall, f1 = test(net, val_loader, device)
        logger.info(
            f"[Centralized Eval] Round {server_round} | "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )
        append_csv(server_round, precision, recall, f1)
        # Optionally add wandb logging here if enabled
        return loss, {"precision": precision, "recall": recall, "f1": f1}

    # Initialize CSV file before training
    init_csv()

    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_fn=evaluate,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )

# Initialize and export Flower server app
app = ServerApp(server_fn=server_fn)
