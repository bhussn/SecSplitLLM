#!/bin/bash

# Configuration
LOGFILE="smpc_federated_run.log"
NUM_CLIENTS=1
SERVER_PORT=8082 
SMPC_PORT=29500
export WORLD_SIZE=2  # Server + clients

# Initialize log
{
echo "Starting SMPC Federated Learning at $(date)"
echo "World Size: $WORLD_SIZE, SMPC Port: $SMPC_PORT, Flower Port: $SERVER_PORT"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
} | tee -a $LOGFILE

# Cleanup previous results
echo "Cleaning up previous results..." | tee -a $LOGFILE
rm -rf fed_learning/results/*.csv data_cache/*.pkl /tmp/server.ready /tmp/client_*.ready

# Set environment variables
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$SMPC_PORT
export SMpc_WORLD_SIZE=$WORLD_SIZE

# Pre-cache datasets
echo "===== PRE-CACHING DATASETS =======" | tee -a $LOGFILE
python -c "
from fed_learning.training.smpc_trainer import load_data
for i in range(10):
    print(f'Caching dataset for client {i}')
    load_data(i, num_partitions=10)
" 2>&1 | tee -a $LOGFILE
echo "Dataset caching complete." | tee -a $LOGFILE

# Step 1: Start the Flower server
echo "Launching Flower server..." | tee -a $LOGFILE
python -m fed_learning.server_side.server_smpc 2>&1 | tee server_smpc.log &
SERVER_PID=$!
echo "Flower server started with PID: $SERVER_PID" | tee -a $LOGFILE

# Wait for Flower server to be ready
echo "Waiting for Flower server on port $SERVER_PORT..." | tee -a $LOGFILE
SERVER_READY=false
for i in {1..120}; do
    if nc -z localhost $SERVER_PORT; then
        SERVER_READY=true
        break
    fi
    sleep 1
    if (( i % 10 == 0 )); then
        echo "Waited $i seconds for Flower server..." | tee -a $LOGFILE
    fi
done

if ! $SERVER_READY; then
    echo "ERROR: Flower server did not start within 120 seconds" | tee -a $LOGFILE
    echo "Server log:" | tee -a $LOGFILE
    cat server_smpc.log | tee -a $LOGFILE
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
echo "Flower server is ready." | tee -a $LOGFILE

# Step 2: Launch clients in parallel
echo "Launching $NUM_CLIENTS clients..." | tee -a $LOGFILE
CLIENT_PIDS=()
for ((i=0; i<$NUM_CLIENTS; i++)); do
    echo "Starting client $i..." | tee -a $LOGFILE
    python -m fed_learning.client_side.client_lora_smpc $i 2>&1 | tee "client_$i.log" &
    CLIENT_PID=$!
    CLIENT_PIDS+=($CLIENT_PID)
    echo "Client $i started with PID: $CLIENT_PID" | tee -a $LOGFILE
    sleep 2
done

# Step 3: Wait for Flower server to finish
echo "Waiting for Flower server to complete..." | tee -a $LOGFILE
wait $SERVER_PID
SERVER_EXIT=$?
echo "Flower server process completed with exit code $SERVER_EXIT." | tee -a $LOGFILE

# Step 4: Wait for all clients to finish
echo "Waiting for clients to complete..." | tee -a $LOGFILE
for pid in "${CLIENT_PIDS[@]}"; do
    if ps -p $pid > /dev/null; then
        wait $pid
        CLIENT_EXIT=$?
        if [ $CLIENT_EXIT -eq 0 ]; then
            echo "Client $pid completed successfully." | tee -a $LOGFILE
        else
            echo "Client $pid exited with error $CLIENT_EXIT." | tee -a $LOGFILE
        fi
    else
        echo "Client $pid already exited." | tee -a $LOGFILE
    fi
done

# Step 5: Final status and resource collection
if [ $SERVER_EXIT -eq 0 ]; then
    echo "===== SIMULATION COMPLETED SUCCESSFULLY =====" | tee -a $LOGFILE
else
    echo "===== SIMULATION FAILED (Server exit: $SERVER_EXIT) =====" | tee -a $LOGFILE
    echo "Error analysis:" | tee -a $LOGFILE
    
    echo "----- SERVER ERRORS -----" | tee -a $LOGFILE
    grep -i "error" server_smpc.log | grep -v "0 errors" | tee -a $LOGFILE || true
    
    for i in {0..2}; do
        if [ -f "client_$i.log" ]; then
            echo "----- CLIENT $i ERRORS -----" | tee -a $LOGFILE
            grep -i "error" "client_$i.log" | grep -v "0 errors" | tee -a $LOGFILE || true
        else
            echo "----- CLIENT $i LOG MISSING -----" | tee -a $LOGFILE
        fi
    done
fi

echo "Simulation completed at $(date)." | tee -a $LOGFILE
echo "Final results stored in fed_learning/results/" | tee -a $LOGFILE
echo "Full log available at $LOGFILE" | tee -a $LOGFILE

exit $SERVER_EXIT
