# Instructions 
1. pull the new code
2. The gRPC files may be corrupted
    - Delete all .py files in the folder (rm -r *.py) 
    - Make a new flwr env with the following dependencies "pip install grpcio grpcio-tools"
    - Inside that environemnt, regenerate the needed files with:
        - Be in the fullsplited directory
        - regenerated with: "python -m grpc_tools.protoc -I=grpc --python_out=grpc --grpc_python_out=grpc grpc/split.proto"
3. If things are still not working, create a new conda environment and 
   download requirements.txt with "pip install -r requirements.txt"
4. Running the program: (ensure you are in the fullsplitfed folder)
    - "python splitfed/server/sever_split.py &" <--- run server
    - "flwr run ." <--- run program
5. Use "lsof -i :50051" to see running servers and be sure to do "kill PID" to
   clean up runs.