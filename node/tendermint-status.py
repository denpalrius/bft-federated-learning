import requests
import json

# Tendermint RPC URL
TENDERMINT_RPC_URL = "http://localhost:26657"

# Function to get Tendermint status
def get_status():
    try:
        response = requests.get(f"{TENDERMINT_RPC_URL}/status")
        response.raise_for_status()  # Raise an exception for bad responses (4xx, 5xx)
        status = response.json()
        return status
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Tendermint: {e}")
        return None

# Function to query the Tendermint node
def query_tendermint(method, params=None):
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or [],
        "id": 1
    }
    
    try:
        response = requests.post(f"{TENDERMINT_RPC_URL}/rpc", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Tendermint: {e}")
        return None

# Main function to test
def main():
    # Fetch status
    status = get_status()
    if status:
        print("Tendermint Status:")
        print(json.dumps(status, indent=2))

    # Example of querying the Tendermint node for the latest block
    block = query_tendermint("block", {"height": "latest"})
    if block:
        print("Latest Block Information:")
        print(json.dumps(block, indent=2))

if __name__ == "__main__":
    main()
