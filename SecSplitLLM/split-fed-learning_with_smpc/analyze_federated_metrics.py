import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Load data
client_df = pd.read_csv("metrics/client_metrics.csv")
server_df = pd.read_csv("metrics/server_metrics_cleaned.csv")
flower_df = pd.read_csv("metrics/flower_server_metrics.csv")

# Convert timestamps
client_df["timestamp"] = pd.to_datetime(client_df["timestamp"], errors="coerce")
server_df["timestamp"] = pd.to_datetime(server_df["timestamp"], errors="coerce")
flower_df["timestamp"] = pd.to_datetime(flower_df["timestamp"], errors="coerce")

# Drop invalid timestamps
client_df.dropna(subset=["timestamp"], inplace=True)
server_df.dropna(subset=["timestamp"], inplace=True)
flower_df.dropna(subset=["timestamp"], inplace=True)

# Separate flower server events
train_events = flower_df[flower_df["event"].str.contains("fit", case=False)]
eval_events = flower_df[flower_df["event"].str.contains("evaluation", case=False)]

# Plotting function with shaded overlays
def plot_metric(df_client, df_server, metric, ylabel, filename):
    plt.figure(figsize=(14, 6))

    # Plot client metrics
    for client_id, group in df_client.groupby("role"):
        plt.plot(group["timestamp"], group[metric], label=f"{client_id}")

    # Plot server metrics
    plt.plot(server_df["timestamp"], server_df[metric], label="Server", color="black", linestyle="--")

    # Add shaded overlays for training
    for ts in train_events["timestamp"]:
        plt.axvspan(ts, ts + pd.Timedelta(seconds=30), color="green", alpha=0.1)

    # Add shaded overlays for evaluation
    for ts in eval_events["timestamp"]:
        plt.axvspan(ts, ts + pd.Timedelta(seconds=30), color="red", alpha=0.1)

    # Custom legend
    legend_elements = [
        Patch(facecolor='green', edgecolor='green', alpha=0.2, label='Training Round'),
        Patch(facecolor='red', edgecolor='red', alpha=0.2, label='Evaluation Event')
    ]
    plt.legend(handles=legend_elements + plt.gca().get_legend_handles_labels()[0])

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Over Time with Training and Evaluation Overlays")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("results", filename))
    plt.close()

# Generate plots
plot_metric(client_df, server_df, "latency_sec", "Latency (seconds)", "latency_overlay_shaded.png")
plot_metric(client_df, server_df, "memory_mb", "Memory Usage (MB)", "memory_overlay_shaded.png")
plot_metric(client_df, server_df, "comm_kb", "Communication Overhead (KB)", "comm_overlay_shaded.png")

print("Improved plots saved in the 'results/' directory.")
