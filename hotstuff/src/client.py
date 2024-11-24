import grpc
import hotstuff_pb2
import hotstuff_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = hotstuff_pb2_grpc.HotStuffStub(channel)

    # Create a proposal message
    proposal = hotstuff_pb2.Proposal(block_id=1, data="Block Data")
    response = stub.ProposeBlock(proposal)
    print(f"ProposeBlock Response: Success={response.success}")

    # Create a vote message
    vote = hotstuff_pb2.Vote(block_id=1, vote=True)
    response = stub.VoteOnBlock(vote)
    print(f"VoteOnBlock Response: Success={response.success}")

if __name__ == '__main__':
    run()
