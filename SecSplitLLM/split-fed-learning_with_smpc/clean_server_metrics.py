import pandas as pd
import os

# Define paths
input_path = os.path.join("metrics", "server_metrics.csv")
output_path = os.path.join("metrics", "server_metrics_cleaned.csv")

# Load the file
df = pd.read_csv(input_path, names=["timestamp", "role", "event", "latency_sec", "memory_mb", "comm_kb"], skiprows=1)

# Convert timestamps, coercing errors
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Drop rows with invalid timestamps
df_cleaned = df.dropna(subset=["timestamp"])

# Save cleaned file
df_cleaned.to_csv(output_path, index=False)

print(f"Cleaned file saved to: {output_path}")
