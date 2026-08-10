#!/usr/bin/env python3
import threading
import signal
import sys
from dotenv import load_dotenv

# Load environment variables from .env file if present at startup
load_dotenv()

from src.configs.log import logger
from src.database.db_access import DBAccess
from src.tasks.task_runner import TaskRunner
from src.messaging.kafka_consumer import KafkaTaskListener

def main():
    logger.info("====================================================")
    logger.info("Starting Football Goal/Replay Detector Worker Service")
    logger.info("====================================================")

    # Initialize database client
    db = DBAccess()

    # Create coordination stop event
    stop_event = threading.Event()

    # 1. Initialize TaskRunner
    runner = TaskRunner()

    # 2. Initialize and start Kafka message listener thread
    listener = KafkaTaskListener(db, runner)
    kafka_thread = threading.Thread(
        target=listener.start_listening,
        args=(stop_event,),
        name="ocr-kafka-listener",
        daemon=True
    )
    kafka_thread.start()
    logger.info("Kafka Task Listener thread started.")

    # Graceful shutdown handler
    def shutdown_handler(signum, frame):
        logger.info("⛔ Received termination signal (%d), starting graceful shutdown...", signum)
        stop_event.set()
        
        # Shutdown TaskRunner
        logger.info("Waiting for TaskRunner executor to finish active tasks...")
        runner.shutdown(wait=True)
        
        # Join Kafka thread
        logger.info("Waiting for Kafka Listener thread to shutdown...")
        kafka_thread.join(timeout=10)
        
        logger.info("✅ Graceful shutdown complete. Exiting.")
        sys.exit(0)

    # Bind OS signals
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    # Keep main thread alive until stop event is set
    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    except KeyboardInterrupt:
        shutdown_handler(signal.SIGINT, None)

if __name__ == '__main__':
    main()
