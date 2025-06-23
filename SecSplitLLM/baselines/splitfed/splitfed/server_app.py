import wandb
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context, ndarrays_to_parameters
from splitfed.task import get_weights
from splitfed.split_model import BertClient
from splitfed.metrics import init_csv
from splitfed.utils import weights_dict_to_list
from splitfed.fedavg_strategy import CustomFedAvg


def server_fn(context: Context):
    wandb.init(project="splitfed", name="flower-split-fed", reinit=True)

    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    wandb.config.update({
        "num_rounds": num_rounds,
        "fraction_fit": fraction_fit,
        "cut_layer": 4,
    })

    init_weights = get_weights(BertClient(cut_layer=4))
    weights_list = weights_dict_to_list(init_weights)
    parameters = ndarrays_to_parameters(weights_list)

    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=1,
        initial_parameters=parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)

    init_csv()

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
