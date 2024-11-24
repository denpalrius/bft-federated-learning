# Initialize tendermint node with default configuration
docker run --rm \
  -v ./tendermint/tendermint_data:/tendermint \
  tendermint/tendermint init

# docker-compose up -d
