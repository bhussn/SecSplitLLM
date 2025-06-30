import logging
import csv
import os
import wandb
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays

logger = logging.getLogger(__name__)


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
    """Log metrics to wandb if initialized."""
    if wandb.run is not None:
        wandb.log(metrics)
    else:
        logger.warning("WandB not initialized. Metrics not logged to WandB.")


class CustomFedAvg(FedAvg):
    def __init__(self, initial_parameters, csv_file=None, *args, **kwargs):
        super().__init__(initial_parameters=initial_parameters, *args, **kwargs)
        self.current_round = 0
        self._prev_params = parameters_to_ndarrays(initial_parameters)
        self.csv_file = csv_file

    def aggregate_fit(self, rnd, results, failures):
        agg = super().aggregate_fit(rnd, results, failures)

        train_losses = []
        fwd_times = []
        bwd_times = []

        for _, fit_res in results:
            metrics = fit_res.metrics or {}
            train_losses.append(metrics.get("train_loss", 0))
            fwd_times.append(metrics.get("forward_time", 0))
            bwd_times.append(metrics.get("backward_time", 0))

        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0
        avg_fwd_time = sum(fwd_times) / len(fwd_times) if fwd_times else 0
        avg_bwd_time = sum(bwd_times) / len(bwd_times) if bwd_times else 0

        self.current_round = rnd

        if agg:
            self.initial_parameters = agg[0]
            self._prev_params = parameters_to_ndarrays(agg[0])

        metrics_to_log = {
            "round": rnd,
            "train_loss": avg_train_loss,
            "forward_time": avg_fwd_time,
            "backward_time": avg_bwd_time,
        }

        if self.csv_file:
            try:
                append_csv(rnd, metrics_to_log, self.csv_file)
            except Exception as e:
                logger.error(f"Failed to write training metrics to CSV: {e}")

        try:
            log_metrics_wandb(metrics_to_log)
        except Exception as e:
            logger.error(f"Failed to log training metrics to WandB: {e}")

        return agg

    def aggregate_evaluate(self, rnd, results, failures):
        val_losses = []
        accuracies = []
        f1s = []

        for _, eval_res in results:
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
            try:
                append_csv(rnd, metrics_to_log, self.csv_file)
            except Exception as e:
                logger.error(f"Failed to write validation metrics to CSV: {e}")

        try:
            log_metrics_wandb(metrics_to_log)
        except Exception as e:
            logger.error(f"Failed to log validation metrics to WandB: {e}")

        return super().aggregate_evaluate(rnd, results, failures)
