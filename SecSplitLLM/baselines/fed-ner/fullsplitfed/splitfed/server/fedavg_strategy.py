import logging
import csv
import os
import wandb
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays

def append_csv(round_num, metrics: dict, csv_file: str):
    """Append metrics as a new row to the CSV file."""
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round"] + list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        row = {"round": round_num}
        row.update(metrics)
        writer.writerow(row)


def log_metrics_wandb(metrics: dict):
    wandb.log(metrics)


class CustomFedAvg(FedAvg):
    def __init__(self, initial_parameters, csv_file=None, *args, **kwargs):
        super().__init__(initial_parameters=initial_parameters, *args, **kwargs)
        self.current_round = 0
        self._prev_params = parameters_to_ndarrays(initial_parameters)
        self.csv_file = csv_file

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure the next round of training."""
        
        # Get the list of client instructions from the parent FedAvg strategy
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        
        # Add the current round number to the config for each client
        new_client_instructions = []
        for client, fit_ins in client_instructions:
            fit_ins.config["round_num"] = server_round
            new_client_instructions.append((client, fit_ins))
            
        return new_client_instructions

    def aggregate_fit(self, rnd, results, failures):
        # Handle empty or invalid results safely
        if not results or all(res is None for _, res in results):
            return self.initial_parameters, {}

        agg = super().aggregate_fit(rnd, results, failures)

        if agg is None or agg[0] is None:
            return self.initial_parameters, {}

        train_losses = []
        fwd_times = []
        bwd_times = []
        comm_costs = []

        for _, fit_res in results:
            if fit_res is None:
                continue
            metrics = fit_res.metrics or {}
            train_losses.append(metrics.get("train_loss", 0))
            fwd_times.append(metrics.get("forward_time", 0))
            bwd_times.append(metrics.get("backward_time", 0))
            comm_costs.append(metrics.get("communication_cost", 0))

        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0
        avg_fwd_time = sum(fwd_times) / len(fwd_times) if fwd_times else 0
        avg_bwd_time = sum(bwd_times) / len(bwd_times) if bwd_times else 0
        avg_comm_cost = sum(comm_costs) / len(comm_costs) if comm_costs else 0
        avg_comm_cost_mb = avg_comm_cost / (1024 * 1024)  # Convert bytes to MB

        self.current_round = rnd

        # Update stored parameters
        self.initial_parameters = agg[0]
        self._prev_params = parameters_to_ndarrays(agg[0])

        metrics_to_log = {
            "round": rnd,
            "train_loss": avg_train_loss,
            "forward_time": avg_fwd_time,
            "backward_time": avg_bwd_time,
            "communication_cost": avg_comm_cost_mb,
        }

        if self.csv_file:
            append_csv(rnd, metrics_to_log, self.csv_file)
            log_metrics_wandb(metrics_to_log)

        return agg

    def aggregate_evaluate(self, rnd, results, failures):
        if not results or all(res is None for _, res in results):
            return None, {}

        val_losses = []
        accuracies = []
        f1s = []

        for _, eval_res in results:
            if eval_res is None:
                continue
            metrics = eval_res.metrics or {}
            val_losses.append(metrics.get("loss", 0))
            accuracies.append(metrics.get("accuracy", 0))
            f1s.append(metrics.get("f1", 0))

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0

        metrics_to_log = {
            "round": rnd,
            "val_loss": avg_val_loss,
            "val_accuracy": avg_accuracy,
            "val_f1": avg_f1,
        }

        if self.csv_file:
            append_csv(rnd, metrics_to_log, self.csv_file)
            log_metrics_wandb(metrics_to_log)

        return super().aggregate_evaluate(rnd, results, failures)
