import logging
import sys

# Define a detailed format string with maximum information
detailed_format = (
    "%(asctime)s.%(msecs)03d | "
    "%(levelname)-8s | "
    "%(filename)s:%(lineno)d | "
    "%(message)s"
)

# Date format with full timestamp
date_format = "%Y-%m-%d %H:%M:%S"

# Configure root logger for terminal output only
logging.basicConfig(
    level=logging.INFO,  # INFO level - filters out DEBUG from libraries
    format=detailed_format,
    datefmt=date_format,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)

# Log initialization
logger.info("Logger initialized.")