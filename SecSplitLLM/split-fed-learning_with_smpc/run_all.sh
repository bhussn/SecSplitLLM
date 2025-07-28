#!/bin/bash

# Ensure metrics directory exists
mkdir -p metrics

# Default values
NUM_CLIENTS=${1:-2}
LOCAL_EPOCHS=${2:-1}
SMPC_SCHEME=${3:-"additive"} 
NUM_ROUNDS=${4:-3}            

LOGFILE="federated_run.log"

echo "Starting federated learning run at $(date)" > $LOGFILE
echo "Using NUM_CLIENTS=$NUM_CLIENTS and LOCAL_EPOCHS=$LOCAL_EPOCHS" | tee -a $LOGFILE
echo "Using SMPC_SCHEME=$SMPC_SCHEME" | tee -a $LOGFILE  
echo "Using NUM_ROUNDS=$NUM_ROUNDS" | tee -a $LOGFILE

# Trap to handle Ctrl+C and clean up background processes
cleanup() {
  echo "Caught interrupt signal. Cleaning up..." | tee -a $LOGFILE
  for pid in "${CLIENT_PIDS[@]}"; do
    kill $pid 2>/dev/null
  done
  kill $FLOWER_PID 2>/dev/null
  kill $GRPC_PID 2>/dev/null
  echo "All background processes terminated." | tee -a $LOGFILE
  exit 1
}
trap cleanup SIGINT SIGTERM

# Step 1: Start the gRPC server
echo "Launching gRPC server..." | tee -a $LOGFILE
python server/grpc_server.py >> $LOGFILE 2>&1 &
GRPC_PID=$!
echo "gRPC server PID: $GRPC_PID" | tee -a $LOGFILE

# Wait for gRPC server to be ready with timeout
echo "Waiting for gRPC server to be ready on port 50052..." | tee -a $LOGFILE
timeout=30
elapsed=0
while ! nc -z localhost 50052; do
  sleep 1
  elapsed=$((elapsed+1))
  if [ $elapsed -ge $timeout ]; then
    echo "Timeout waiting for gRPC server." | tee -a $LOGFILE
    cleanup
  fi
done
echo "gRPC server is ready." | tee -a $LOGFILE

# Step 2: Start the Flower server
echo "Launching Flower server..." | tee -a $LOGFILE
python server/server_fl.py --smpc_scheme $SMPC_SCHEME --num_rounds $NUM_ROUNDS >> $LOGFILE 2>&1 &
FLOWER_PID=$!
echo "Flower server PID: $FLOWER_PID" | tee -a $LOGFILE

# Wait for Flower server to be ready with timeout
echo "Waiting for Flower server to be ready on port 8082..." | tee -a $LOGFILE
timeout=30
elapsed=0
while ! nc -z localhost 8082; do
  sleep 1
  elapsed=$((elapsed+1))
  if [ $elapsed -ge $timeout ]; then
    echo "Timeout waiting for Flower server." | tee -a $LOGFILE
    cleanup
  fi
done
echo "Flower server is ready." | tee -a $LOGFILE

# Step 3: Launch clients in parallel
CLIENT_PIDS=()
for ((i=0; i<$NUM_CLIENTS; i++)); do
  echo "Launching client $i..." | tee -a $LOGFILE
  NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  GPU_ID=$((i % NUM_GPUS))
  CUDA_VISIBLE_DEVICES=$GPU_ID python clients/client_fl.py $i --local_epochs $LOCAL_EPOCHS >> $LOGFILE 2>&1 &
  pid=$!
  CLIENT_PIDS+=($pid)
  echo "Client $i PID: $pid" | tee -a $LOGFILE
done

# Step 4: Wait for Flower server to finish
wait $FLOWER_PID
echo "Flower server process completed." | tee -a $LOGFILE

# Step 5: Wait for all clients to finish and summarize exit codes
for pid in "${CLIENT_PIDS[@]}"; do
  wait $pid
  exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "Client process $pid completed successfully." | tee -a $LOGFILE
  else
    echo "Client process $pid exited with error code $exit_code." | tee -a $LOGFILE
  fi
done

# Step 7: Run visualization scripts
echo "Running analysis and visualization scripts..." | tee -a metrics/federated_run.log

# Ensure results directory exists
mkdir -p results

# Run analysis with overlays and summary statistics
python analyze_federated_metrics.py >> metrics/federated_run.log 2>&1

# Visualize summary statistics
python visualize_summary_statistics.py >> metrics/federated_run.log 2>&1

echo "Visualization completed. Results saved in 'results/'." | tee -a metrics/federated_run.log

# Step 6: Kill gRPC server
kill $GRPC_PID
echo "gRPC server terminated." | tee -a $LOGFILE

echo "Federated learning session completed at $(date)." | tee -a $LOGFILE
