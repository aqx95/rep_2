import os
import logging

def log(config, name):
    log_file = os.path.join(config.LOG_PATH, 'log.txt')
    if not os.path.isfile(log_file):
        os.makedirs(config.LOG_PATH)
        open(log_file, 'w+').close()

    console_log_format = "%(levelname)s %(message)s"
    file_log_format = "%(levelname)s: %(asctime)s: %(message)s"

    #Configure logger
    logging.basicConfig(level=logging.INFO, format=console_log_format)
    logger = logging.getLogger(name)

    #File handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(file_log_format)
    handler.setFormatter(formatter)

    #Add handler to logger
    logger.addHandler(handler)

    return logger
