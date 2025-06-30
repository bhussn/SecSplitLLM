import wandb
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context, ndarrays_to_parameters

from splitfed.models.split_model import BertClient
from splitfed.training.split_trainer import get_weights
from splitfed.training.utils import weights_dict_to_list, init_csv
from splitfed.server.fedavg_strategy import CustomFedAvg

def server_fn(context: Context):
    wandb.init(project="splitfed", name="flower-split-fed", reinit=True)

    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    cut_layer = 4

    wandb.config.update({
        "num_rounds": num_rounds,
        "fraction_fit": fraction_fit,
        "cut_layer": cut_layer,
    })

    model = BertClient(cut_layer=cut_layer)
    init_weights_dict = get_weights(model)
    weights_list = weights_dict_to_list(init_weights_dict)
    initial_parameters = ndarrays_to_parameters(weights_list)

    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        csv_file="server_metrics.csv",
    )

    config = ServerConfig(num_rounds=num_rounds)
    init_csv("server_metrics.csv")

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
