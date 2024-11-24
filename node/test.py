import struct
from tendermint.abci.types_pb2 import (
    ResponseInfo,
    ResponseInitChain,
    ResponseCheckTx,
    ResponseDeliverTx,
    ResponseQuery,
    ResponseCommit,
)
from abci.server import ABCIServer
from abci.application import BaseApplication, OkCode, ErrorCode


# Tx encoding/decoding
def encode_number(value):
    return struct.pack(">I", value)


def decode_number(raw):
    return int.from_bytes(raw, byteorder="big")


class SimpleCounter(BaseApplication):
    def __init__(self):
        # Initialize state variable for counter
        self.txCount = 0
        self.last_block_height = 0

    def info(self, req) -> ResponseInfo:
        """
        Since this will always respond with height=0, Tendermint
        will resync this app from the beginning.
        """
        r = ResponseInfo()
        r.version = req.version
        r.last_block_height = self.last_block_height
        r.last_block_app_hash = b""
        return r

    def init_chain(self, req) -> ResponseInitChain:
        """Set initial state on first run"""
        self.txCount = 0
        self.last_block_height = 0
        print("Initialized chain with state:", self.txCount)
        return ResponseInitChain()

    def check_tx(self, tx) -> ResponseCheckTx:
        """
        Validate the Tx before entry into the mempool.
        Checks that txs are submitted in order 1,2,3...
        """
        value = decode_number(tx)
        print(f"Checking transaction: {value}, expecting {self.txCount + 1}")
        if value != (self.txCount + 1):
            print(f"Transaction failed, expected {self.txCount + 1}, got {value}")
            return ResponseCheckTx(code=ErrorCode)
        return ResponseCheckTx(code=OkCode)

    def deliver_tx(self, tx) -> ResponseDeliverTx:
        """
        We have a valid tx, increment the state.
        """
        self.txCount += 1
        self.last_block_height += 1  # Increment height on each successful transaction
        print(f"Transaction delivered, updated state to {self.txCount}")
        return ResponseDeliverTx(code=OkCode)

    def query(self, req) -> ResponseQuery:
        """Return the last tx count"""
        print(f"Querying current state: {self.txCount}")
        v = encode_number(self.txCount)
        return ResponseQuery(
            code=OkCode, value=v, height=self.last_block_height
        )

    def commit(self) -> ResponseCommit:
        """Return the current encoded state value to Tendermint"""
        state = struct.pack(">Q", self.txCount)
        print(f"Committing state: {self.txCount}")
        return ResponseCommit(data=state)


def main():
    app = ABCIServer(app=SimpleCounter())
    print("Starting the BFT app with Tendermint...")
    app.run()


if __name__ == "__main__":
    main()
