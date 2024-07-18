import logging


def setup_logging(log_file_name):
    # Create a custom logger
    logger = logging.getLogger("my_app")
    logger.setLevel(logging.DEBUG)  # Set minimum level of logs to handle

    # Create handlers (console and file handler for example)
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(f"{log_file_name}.log")
    c_handler.setLevel(logging.WARNING)  # Console handles only warnings and above
    f_handler.setLevel(logging.DEBUG)  # File handles all debug logs

    # Create formatters and add it to handlers
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger
