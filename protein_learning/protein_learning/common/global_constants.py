"""Global constants"""
from datetime import datetime
import logging
import sys

TRAINING, TESTING, VALIDATION = "training", 'testing', 'validation'
STATS, MODELS, PARAMS, LOGS, CHECKPOINTS = 'stats', 'models', 'params', 'logs', 'checkpoints'
NAME, PATH, EXT = 'name', 'path', 'ext'
REZERO_INIT = 0.01
MAX_SEQ_LEN = 800
START_TIME = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")

# Creating and Configuring Logger
logging.basicConfig(
    stream=sys.stdout,
    level=logging.WARNING
)
get_logger = lambda name: logging.getLogger(name)
