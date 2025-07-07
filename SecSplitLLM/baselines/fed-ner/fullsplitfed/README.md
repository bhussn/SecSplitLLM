# Instructions 
1. pull the new code
2. The gRPC files may be corrupted
    - In the grpc directory delete all .py files in the folder (rm -r *.py) 
    - Make a new conda environment with the following dependencies "pip install grpcio grpcio-tools"
    - Make sure to use the 4.25.1 version of protobuf, install it with this command "pip install protobuf==4.25.1"
    - Inside that environemnt, regenerate the needed files with:
        - Be in the fullsplited directory
        - regenerated with: "python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. splitfed/grpc/split.proto"
    - Leave the environment and go back to the original flwr environment 
3. If things are still not working, create a new conda environment and 
   download requirements.txt with "pip install -r requirements.txt"
4. Running the program: (ensure you are in the fullsplitfed folder)
    - "python -m splitfed.server.server_split &" <--- run server
    - "flwr run ." <--- run program
5. Use "lsof -i :50052" to see running servers and be sure to do "kill PID" to
   clean up runs.
