docker run --rm \
  -v $(pwd)/tendermint_data:/tendermint \
  tendermint/tendermint:v0.35.0 init
