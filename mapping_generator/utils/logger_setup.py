import logging
import sys

class CleanFormatter(logging.Formatter):
    """
    Custom log formatter.
    INFO: Standard terminal text (no colors, no emojis).
    WARN/ERROR: Colored for visibility.
    """
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMAT_DEFAULT = "[%(levelname)s] %(message)s"
    FORMAT_INFO = "%(message)s"

    FORMATS = {
        logging.DEBUG: grey + FORMAT_DEFAULT + reset,
        logging.INFO: FORMAT_INFO,
        logging.WARNING: yellow + "[WARNING] %(message)s" + reset,
        logging.ERROR: red + "[ERROR] %(message)s" + reset,
        logging.CRITICAL: bold_red + "[CRITICAL] %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(verbose: bool = False):
    """
    Configures the root logger.
    
    Args:
        verbose (bool): If True, sets level to DEBUG. Otherwise, INFO.
    """
    root_logger = logging.getLogger()
    
    if root_logger.handlers:
        root_logger.handlers = []
        
    level = logging.DEBUG if verbose else logging.INFO
    root_logger.setLevel(level)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CleanFormatter())
    root_logger.addHandler(console_handler)
