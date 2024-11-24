import grpc
from concurrent import futures
import time
import hotstuff_pb2
import hotstuff_pb2_grpc

class HotStuffServicer(hotstuff_pb2_grpc.HotStuffServicer):
    def ProposeBlock(self, request, context):
        # Handle block proposal logic
        print(f"Received proposal: Block ID {request.block_id}, Data: {request.data}")
        return hotstuff_pb2.BlockResponse(success=True)

    def VoteOnBlock(self, request, context):
        # Handle voting logic
        print(f"Received vote: Block ID {request.block_id}, Vote: {request.vote}")
        return hotstuff_pb2.VoteResponse(success=True)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    hotstuff_pb2_grpc.add_HotStuffServicer_to_server(HotStuffServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051...")
    try:
        while True:
            time.sleep(86400)  # Keep the server running
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
