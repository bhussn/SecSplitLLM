#!/bin/bash

# Cleanup previous results
rm -rf fed_learning/results/*.csv
rm -rf data_cache/*.pkl

# Pre-cache datasets
echo "===== PRE-CACHING DATASETS ====="
python -c "
from fed_learning.training.smpc_trainer import load_data
for i in range(10):
    print(f'Caching dataset for client {i}')
    load_data(i, num_partitions=10)
"

# Set environment variables
export FLWR_SIMULATION="true"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"
export WORLD_SIZE=4  # Server + 3 clients

# Start the server
echo "===== STARTING SERVER ====="
flwr run --app fed_learning.server_side.server_smpc:app > server.log 2>&1 &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"
sleep 15  # Give server time to initialize

# Start clients
echo "===== STARTING CLIENTS ====="
for i in {0..2}; do
    # Set rank for each client (1,2,3)
    RANK=$((i+1))
    echo "Starting client $i with RANK=$RANK"
    
    flwr run --app fed_learning.client_side.client_lora_smpc:app --node-id $i > client_$i.log 2>&1 &
    CLIENT_PIDS[$i]=$!
    echo "Client $i started with PID: ${CLIENT_PIDS[$i]}"
    
    sleep 5  # Stagger client startup
done

# Monitor processes
echo "===== SIMULATION RUNNING ====="
echo "Server PID: $SERVER_PID"
echo "Client PIDs: ${CLIENT_PIDS[@]}"

# Wait for server to complete
echo "Waiting for simulation to complete..."
wait $SERVER_PID
SERVER_EXIT=$?

# Cleanup client processes if needed
for pid in "${CLIENT_PIDS[@]}"; do
    if ps -p $pid > /dev/null; then
        echo "Terminating client PID $pid"
        kill $pid
    fi
done

# Final status
if [ $SERVER_EXIT -eq 0 ]; then
    echo "===== SIMULATION COMPLETED SUCCESSFULLY ====="
else
    echo "===== SIMULATION FAILED (Exit code: $SERVER_EXIT) ====="
fi

echo "Server log: server.log"
echo "Client logs: client_*.log"
exit $SERVER_EXIT