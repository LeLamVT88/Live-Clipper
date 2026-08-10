import pymysql
import pymysql.cursors
import os
import time
from src.config import WORKER_DEFAULT_CONFIG as config
from src.configs.log import logger

class DBAccess:
    def __init__(self):
        # MySQL parameters
        self.mysql_host = config.get('mysql_host', '127.0.0.1')
        self.mysql_port = config.get('mysql_port', 3306)
        self.mysql_user = config.get('mysql_user', 'clipper_user')
        self.mysql_password = config.get('mysql_password', 'clipper_pass')
        self.mysql_database = config.get('mysql_database', 'live_clipper')

        self._init_db()

    def _get_mysql_connection(self):
        # Return a PyMySQL connection that uses DictCursor
        return pymysql.connect(
            host=self.mysql_host,
            port=self.mysql_port,
            user=self.mysql_user,
            password=self.mysql_password,
            database=self.mysql_database,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False
        )

    def _init_db(self):
        """Initializes MySQL connection and verifies connectivity."""
        try:
            with self._get_mysql_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
            logger.info("MySQL database connected successfully at %s:%d/%s (schema unchanged)", 
                        self.mysql_host, self.mysql_port, self.mysql_database)
        except Exception as e:
            logger.error("Failed to connect to MySQL database: %s", e)
            raise

    def get_file_details(self, final_file):
        """Queries the MySQL database for schedule_id and duration matching final_file."""
        sql = "SELECT schedule_id, duration FROM full_path_contents WHERE final_file = %s LIMIT 1"
        try:
            with self._get_mysql_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (final_file,))
                    row = cursor.fetchone()
                    if row:
                        return dict(row)
            return None
        except Exception as e:
            logger.error("Failed to query file details from MySQL for '%s': %s", final_file, e)
            return None

    def insert_file_placeholder(self, schedule_id, final_file, duration=0):
        """Inserts a default placeholder entry in MySQL for development and testing support."""
        sql = """
        INSERT INTO full_path_contents (schedule_id, final_file, duration)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            schedule_id = VALUES(schedule_id),
            duration = VALUES(duration);
        """
        try:
            with self._get_mysql_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (schedule_id, final_file, duration))
                conn.commit()
            logger.info("Inserted placeholder row in MySQL full_path_contents for '%s'", final_file)
            return True
        except Exception as e:
            logger.error("Failed to insert placeholder in MySQL for '%s': %s", final_file, e)
            return False

    def complete_task(self, schedule_id, goals):
        """Saves goal events to MySQL match_events."""
        sql_event = """
        INSERT INTO match_events (schedule_id, time, type, start_time, end_time, status, full_path, final_path, comment)
        VALUES (%s, %s, %s, %s, %s, 2, %s, %s, %s);
        """
        try:
            with self._get_mysql_connection() as mysql_conn:
                mysql_conn.begin()
                with mysql_conn.cursor() as mysql_cursor:
                    # Batch insert events
                    for g in goals:
                        mysql_cursor.execute(sql_event, (
                            schedule_id,
                            g['time'],
                            g['type'],
                            g['start_time'],
                            g['end_time'],
                            g['full_path'],
                            g['final_path'],
                            g['comment']
                        ))
                mysql_conn.commit()
            logger.info("Successfully completed task (schedule_id=%s), recorded %d events in MySQL", schedule_id, len(goals))
            return True
        except Exception as e:
            logger.error("Failed to complete task details in MySQL: %s", e)
            return False
