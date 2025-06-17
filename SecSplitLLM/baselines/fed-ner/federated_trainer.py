import torch
from flwr.simulation import start_simulation
from flwr.server.server import ServerConfig
from fed_ner import client_app, server_app

def main():
    num_clients = 5
    num_rounds = 10

    client_config = {
        "local-epochs": 1,
        "num-partitions": num_clients,
    }
    server_config = {
        "num-server-rounds": num_rounds,
        "fraction-fit": 1.0,
    }

    server = server_app.app

    def client_fn(cid: str):
        partition_id = int(cid)
        context = {
            "node_name": f"client_{cid}",
            "node_config": {"partition-id": partition_id, "num-partitions": num_clients},
            "run_config": client_config,
        }
        return client_app.client_fn(context)

    start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        client_resources={"num_gpus": 1 if torch.cuda.is_available() else 0},
        server=server,
        config=ServerConfig(num_rounds=num_rounds),
    )

if __name__ == "__main__":
    main()
