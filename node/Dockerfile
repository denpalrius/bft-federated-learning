FROM python:3.9-slim

# Set working directory
WORKDIR /node

# Copy application files
COPY . /node

# Install dependencies
RUN pip install --no-cache-dir torch torchvision abci numpy cryptography

# Expose node's gRPC port
EXPOSE 50051

# Command to run the node
CMD ["python", "node_app.py", "--node-id=node"]
