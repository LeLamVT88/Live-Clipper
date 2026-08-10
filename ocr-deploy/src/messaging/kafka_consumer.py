import json
import os
import time
import threading
from confluent_kafka import Consumer, KafkaError
from src.config import WORKER_DEFAULT_CONFIG as config
from src.configs.log import logger
from src.database.db_access import DBAccess
from src.utils.AES_cipher import AESCipher

class KafkaTaskListener:
    def __init__(self, db_access: DBAccess, runner):
        self.db = db_access
        self.runner = runner
        self.cipher = AESCipher()
        self.bootstrap_servers = config.get('kafka_bootstrap_servers', 'localhost:9092')
        self.group_id = config.get('kafka_consumer_group_id', 'ocr-detector-group')
        self.topic = config.get('kafka_task_topic', 'ocr-video-tasks')
        self.consumer = None

    def _init_consumer(self):
        conf = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
        }

        sasl_mechanism = os.environ.get('KAFKA_SASL_MECHANISM')
        if sasl_mechanism:
            conf['sasl.mechanism'] = sasl_mechanism
            conf['security.protocol'] = os.environ.get('KAFKA_SECURITY_PROTOCOL', 'SASL_PLAINTEXT')
            conf['sasl.username'] = os.environ.get('KAFKA_SASL_USERNAME', '')
            raw_pass = os.environ.get('KAFKA_SASL_PASSWORD', '')
            conf['sasl.password'] = self.cipher.decrypt(raw_pass)
            logger.info("Kafka SASL authentication configured | mechanism=%s | user=%s",
                        sasl_mechanism, conf['sasl.username'])

        consumer = Consumer(conf)
        consumer.subscribe([self.topic])
        return consumer

    def _close_consumer(self):
        if self.consumer:
            try:
                self.consumer.close()
            except Exception:
                pass
            self.consumer = None

    def start_listening(self, stop_event: threading.Event):
        logger.info("Starting Kafka Task Listener thread...")

        retry_delay = 1.0
        max_delay = 30.0

        while not stop_event.is_set():
            try:
                self.consumer = self._init_consumer()
                logger.info("Kafka consumer subscribed successfully to topic='%s'", self.topic)
                retry_delay = 1.0
            except Exception as e:
                logger.error("Failed to initialize Kafka consumer: %s. Retrying in %.0fs...", e, retry_delay)
                if stop_event.wait(retry_delay):
                    return
                retry_delay = min(retry_delay * 2, max_delay)
                continue

            while not stop_event.is_set():
                try:
                    msg = self.consumer.poll(1.0)
                    if msg is None:
                        continue

                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            continue
                        logger.error("Kafka consumer error: %s", msg.error())
                        self._close_consumer()
                        break

                    retry_delay = 1.0

                    val = msg.value()
                    if not val:
                        continue

                    payload_str = val.decode('utf-8')
                    logger.info("Kafka consumed message value: %s", payload_str)

                    try:
                        payload = json.loads(payload_str)
                    except Exception as ex:
                        logger.error("Failed to parse Kafka message JSON payload: %s", ex)
                        continue

                    file_path = payload.get('filePath') or payload.get('file_path')
                    start_time = payload.get('startTime')
                    if start_time is None:
                        start_time = payload.get('start_time', 0.0)
                    end_time = payload.get('endTime')
                    if end_time is None:
                        end_time = payload.get('end_time', 0.0)

                    if not file_path:
                        logger.warning("Kafka message discarded: missing 'filePath'. Payload: %s", payload_str)
                        continue

                    details = self.db.get_file_details(file_path)
                    if details:
                        schedule_id = details['schedule_id']
                        duration = details['duration']
                    else:
                        logger.warning("filePath '%s' not found in MySQL full_path_contents. Creating a default entry with schedule_id=101.", file_path)
                        schedule_id = 101
                        duration = 0
                        self.db.insert_file_placeholder(schedule_id, file_path, duration)

                    task_config = {
                        'start_time': float(start_time),
                        'end_time': float(end_time)
                    }

                    self.runner.submit_task(
                        final_file=file_path,
                        schedule_id=schedule_id,
                        duration=duration,
                        config_overrides=task_config
                    )

                except Exception as e:
                    logger.error("Exception in Kafka Task Listener loop: %s", e, exc_info=True)
                    time.sleep(2.0)

        self._close_consumer()
        logger.info("Kafka consumer connection closed.")
