import logging
import os

def setup_logger(name: str = None, log_level: int = logging.INFO) -> logging.Logger:
    log_format = "%(asctime)s - %(name)s - %(levelname)s: %(message)s"
    
    log_file = os.getenv("LOG_FILE", "output.log") 

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    # Avoid duplicate logs by disabling propagation
    logger.propagate = False

    return logger
