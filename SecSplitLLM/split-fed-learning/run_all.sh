#!/bin/bash

# Usage: ./script.sh <run_number> <grpc_port> <flower_port>
RUN_NUMBER=${1:-1}         # default to 1 if not provided
GRPC_PORT=${2:-50055}      # default to 50055 if not provided
FLOWER_PORT=${3:-8085}     # default to 8085 if not provided

# Log file
LOGFILE="federated_run_${RUN_NUMBER}.log"
LOCAL_EPOCHS=3
echo "------ RUN $RUN_NUMBER ------"
echo "Starting federated learning run at $(date)" > $LOGFILE

# Step 1: Start the gRPC server (server_fl.py)
echo "Launching gRPC server on port $GRPC_PORT..." | tee -a $LOGFILE
python server/baseline_grpc.py --port $GRPC_PORT >> $LOGFILE 2>&1 &
GRPC_PID=$!

# Wait for gRPC server to be ready
echo "Waiting for gRPC server to be ready on port $GRPC_PORT..." | tee -a $LOGFILE
while ! nc -z localhost $GRPC_PORT; do
  sleep 1
done
echo "gRPC server is ready." | tee -a $LOGFILE

# Step 2: Start the Flower server (in background)
echo "Launching Flower server on port $FLOWER_PORT..." | tee -a $LOGFILE
python server/baseline_fl.py --port $FLOWER_PORT >> $LOGFILE 2>&1 &
FLOWER_PID=$!

# Wait for Flower server to be ready
echo "Waiting for Flower server to be ready on port $FLOWER_PORT..." | tee -a $LOGFILE
while ! nc -z localhost $FLOWER_PORT; do
  sleep 1
done
echo "Flower server is ready." | tee -a $LOGFILE

# Step 3: Launch clients in parallel
NUM_CLIENTS=5
CLIENT_PIDS=()

for ((i=0; i<$NUM_CLIENTS; i++)); do
  echo "Launching client $i..." | tee -a $LOGFILE
  python clients/baseline_client.py $i --local_epochs $LOCAL_EPOCHS >> $LOGFILE 2>&1 &
  CLIENT_PIDS+=($!)
done

# Step 4: Wait for Flower server to finish
wait $FLOWER_PID
echo "Flower server process completed." | tee -a $LOGFILE

# Step 5: Wait for all clients to finish
for pid in "${CLIENT_PIDS[@]}"; do
  wait $pid
  if [ $? -eq 0 ]; then
    echo "Client process $pid completed successfully." | tee -a $LOGFILE
  else
    echo "Client process $pid exited with error." | tee -a $LOGFILE
  fi
done

# Step 6: Kill gRPC server
kill $GRPC_PID
echo "gRPC server terminated." | tee -a $LOGFILE

echo "Federated learning session completed at $(date)." | tee -a $LOGFILE
