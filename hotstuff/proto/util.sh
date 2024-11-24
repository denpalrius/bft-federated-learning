
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/hotstuff.proto



# Generate python code from proto file
# protoc -I . --python_betterproto_out=src proto/hotstuff.proto

# python -m grpc_tools.protoc -I . --python_betterproto_out=proto proto/hotstuff.proto