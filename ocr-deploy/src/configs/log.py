import os
import logging
from logging.handlers import TimedRotatingFileHandler

def setup_logger(name: str, log_file: str, level=logging.INFO):
    logger = logging.getLogger(name)
    
    # Configure logging level based on DEBUG env variable
    debug_mode = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes", "y")
    log_level = logging.DEBUG if debug_mode else level
    logger.setLevel(log_level)

    # Avoid duplicate handlers if initialized multiple times
    if logger.hasHandlers():
        return logger

    # Log line formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    if debug_mode:
        # Ensure parent directories exist
        if os.path.dirname(log_file):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Daily rotating file handler
        file_handler = TimedRotatingFileHandler(
            log_file, when='midnight', interval=1, backupCount=7, encoding='utf-8', utc=True
        )
        file_handler.suffix = '%Y%m%d'
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console output handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Pre-initialize a standard logger for the OCR service
log_file_path = os.environ.get('WORKER_LOG_FILE', 'logs/worker.log')
logger = setup_logger("ocr-worker", log_file_path)
