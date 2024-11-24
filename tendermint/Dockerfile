# Tendermint official image
FROM tendermint/tendermint:latest

# Set working directory
WORKDIR /tendermint

# Expose ports
EXPOSE 26656 26657 26658

# Tendermint initialization and execution
ENTRYPOINT [ "tendermint", "node", "--proxy_app=kvstore" ]
