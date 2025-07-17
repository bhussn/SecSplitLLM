import os
import csv
import time
from datetime import datetime

def log_to_csv(file_path, headers, row_data):
    """Safely logs data to CSV, creating file if needed"""
    write_header = not os.path.exists(file_path)
    
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerow(row_data)