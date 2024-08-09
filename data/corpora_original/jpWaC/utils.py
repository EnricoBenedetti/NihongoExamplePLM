import os
import random
import numpy as np
from transformers import set_seed
import logging
import sys

RNG_SEED = 42


# Create a logger
def get_logger():
    """Returns logger for the script"""

    program_name = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program_name)

    # Set the logging level (choose one of: DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.DEBUG)

    # Create a file handler to save logs to a file
    file_handler = logging.FileHandler(f'{program_name}logs.log')

    # Create a formatter to specify the log message format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger


def fix_reproducibility():
   # Set a seed value
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(RNG_SEED)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(RNG_SEED)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(RNG_SEED)
    set_seed(RNG_SEED)

def __init__():
    fix_reproducibility()