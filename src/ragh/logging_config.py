from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="DEBUG", backtrace=True, diagnose=True)
logger.add("logs/ragh_{time}.log", rotation="10 MB", retention="14 days", level="INFO")
