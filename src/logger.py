import logging


# Global logging configuration
logging.basicConfig(
    filename='output.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set log level for specific third-party packages
logging.getLogger("flwr").setLevel(logging.DEBUG)

# Your code
logging.debug("This is a debug message.")
logging.info("This is an info message.")
