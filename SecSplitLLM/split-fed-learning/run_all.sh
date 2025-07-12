#!/bin/bash

# Log file
LOGFILE="federated_run.log"
LOCAL_EPOCHS=3
echo "Starting federated learning run at $(date)" > $LOGFILE

# Step 1: Start the gRPC server (server_fl.py)
echo "Launching gRPC server..." | tee -a $LOGFILE
python server/grpc_server.py >> $LOGFILE 2>&1 &
GRPC_PID=$!

# Wait for gRPC server to be ready
echo "Waiting for gRPC server to be ready on port 50051..." | tee -a $LOGFILE
while ! nc -z localhost 50051; do
  sleep 1
done
echo "gRPC server is ready." | tee -a $LOGFILE

# Step 2: Start the Flower server (in background)
echo "Launching Flower server..." | tee -a $LOGFILE
python server/server_fl.py >> $LOGFILE 2>&1 &
FLOWER_PID=$!

# Wait for Flower server to be ready
echo "Waiting for Flower server to be ready on port 8081..." | tee -a $LOGFILE
while ! nc -z localhost 8081; do
  sleep 1
done
echo "Flower server is ready." | tee -a $LOGFILE

# Step 3: Launch clients in parallel
NUM_CLIENTS=5
CLIENT_PIDS=()

for ((i=0; i<$NUM_CLIENTS; i++)); do
  echo "Launching client $i..." | tee -a $LOGFILE
python clients/client_fl.py $i --local_epochs $LOCAL_EPOCHS >> $LOGFILE 2>&1 &
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



