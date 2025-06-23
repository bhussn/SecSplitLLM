# metrics.py

import csv
import os
import wandb

CSV_FILE = "sfl_metrics.csv"

def init_csv(csv_path=CSV_FILE):
    if not os.path.isfile(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "accuracy", "f1", "communication_cost_bytes"])

def append_csv(round_num, accuracy, f1_score, communication_cost, csv_path=CSV_FILE):
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([round_num, accuracy, f1_score, communication_cost])

def log_metrics_wandb(rnd, accuracy, f1, communication_cost):
    wandb.log({
        "round": rnd,
        "accuracy": accuracy,
        "f1": f1,
        "communication_cost_bytes": communication_cost,
    })

def append_timing(round_num, avg_fwd, avg_bwd, path="timing_metrics.csv"):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["round", "client_forward_ms", "server_backward_ms"])
        writer.writerow([round_num, avg_fwd, avg_bwd])

def log_timing_wandb(round_num, avg_fwd, avg_bwd):
    wandb.log({
        "round": round_num,
        "client_forward_ms": avg_fwd,
        "server_backward_ms": avg_bwd,
    })