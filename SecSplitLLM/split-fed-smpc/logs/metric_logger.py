
import time
import os
import psutil
import sys
import csv

class MetricsLogger:
    def __init__(self, role="client", log_file=None):
        self.role = role
        self.process = psutil.Process(os.getpid())
        self.log_file = log_file
        if log_file:
            with open(log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "role", "event", "latency_sec", "memory_mb", "comm_kb"])

    def log_latency(self, start_time, event=""):
        latency = time.time() - start_time
        # print(f"[{self.role}] {event} Latency: {latency:.4f} seconds")
        return latency

    def log_memory(self, event=""):
        mem_mb = self.process.memory_info().rss / 1024 / 1024
        # print(f"[{self.role}] {event} Memory Usage: {mem_mb:.2f} MB")
        return mem_mb

    def log_communication(self, data_dict, event=""):
        total_bytes = sum(sys.getsizeof(v) for v in data_dict.values())
        total_kb = total_bytes / 1024
        # print(f"[{self.role}] {event} Communication Overhead: {total_kb:.2f} KB")
        return total_kb

    def log_all(self, start_time, data_dict, event=""):
        latency = self.log_latency(start_time, event)
        memory = self.log_memory(event)
        comm_kb = self.log_communication(data_dict, event)
        if self.log_file:
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), self.role, event, latency, memory, comm_kb])
        return latency, memory, comm_kb
