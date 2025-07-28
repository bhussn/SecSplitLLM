import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary statistics
df = pd.read_csv("results/summary_statistics.csv", header=[0, 1], index_col=0)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Plotting function
def plot_metric(metric_name, filename):
    metric_df = df[metric_name]
    metric_df.plot(kind="bar", figsize=(12, 6))
    plt.title(f"{metric_name.replace('_', ' ').title()} Summary Statistics by Role")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.xlabel("Role")
    plt.xticks(rotation=0)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join("results", filename))
    plt.close()

# Generate plots
plot_metric("latency_sec", "latency_summary.png")
plot_metric("memory_mb", "memory_summary.png")
plot_metric("comm_kb", "comm_summary.png")

print("Summary statistic plots saved in the 'results/' directory.")
