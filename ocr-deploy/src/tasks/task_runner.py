import time
import json
import os
from concurrent.futures import ThreadPoolExecutor
from src.config import WORKER_DEFAULT_CONFIG as config
from src.configs.log import logger
from src.database.db_access import DBAccess
from src.sweep.detector import GoalDetector

class TaskRunner:
    def __init__(self):
        self.db = DBAccess()
        self.max_workers = config.get('max_workers', 2)
        
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="ocr-worker-"
        )
        logger.info("TaskRunner initialized | max_workers=%d", self.max_workers)

    def submit_task(self, final_file, schedule_id, duration=0, config_overrides=None):
        """Submits task execution to the in-memory thread pool."""
        logger.info("Submitting task for file='%s' | schedule_id=%s to ThreadPoolExecutor", final_file, schedule_id)
        self.executor.submit(
            self._process_task, 
            final_file, 
            schedule_id, 
            duration, 
            config_overrides
        )

    def shutdown(self, wait=True):
        """Shuts down the in-memory ThreadPoolExecutor."""
        logger.info("Shutting down TaskRunner executor...")
        self.executor.shutdown(wait=wait)

    def _process_task(self, video_path: str, schedule_id: int, duration: int, config_overrides: dict) -> None:
        logger.info("Starting processing for file='%s' | schedule_id=%s", video_path, schedule_id)

        if not config_overrides:
            config_overrides = {}

        # Resolve output directory
        # By default, use directory of the video path + '/ocr_output' or fallback to default
        video_dir = os.path.dirname(video_path)
        output_dir = config_overrides.get('output_dir') or os.path.join(video_dir, 'ocr_output')

        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found at: {video_path}")

            # Instantiate and run GoalDetector
            detector = GoalDetector(video_path, output_dir, config_overrides)
            detector.calibrate()
            goals = detector.track()

            # Format outputs matching Java database match_events schema
            db_events = []
            for g in goals:
                # Resolve start/end timings based on replay results
                replay = g.get('replay')
                if replay:
                    start_time = int(round(replay.get('true_start_sec', g['timestamp_sec'] - 10)))
                    end_time = int(round(replay.get('true_end_sec', g['timestamp_sec'] + 15)))
                    clip_path = replay.get('clip_path')
                else:
                    start_time = int(round(g['timestamp_sec'] - 10))
                    end_time = int(round(g['timestamp_sec'] + 15))
                    clip_path = None

                comment = f"Goal {g['before_score']} -> {g['after_score']} (Scorer: {g['scorer_side']})"
                
                db_events.append({
                    'schedule_id': schedule_id,
                    'time': g['timestamp_formatted'],
                    'type': 'GOAL',
                    'start_time': start_time,
                    'end_time': end_time,
                    'full_path': video_path,
                    'final_path': clip_path,
                    'comment': comment
                })

            # Save results to MySQL
            self.db.complete_task(schedule_id, db_events)
            logger.info("Finished processing file='%s' | schedule_id=%s successfully.", video_path, schedule_id)

        except Exception as err:
            logger.error("Error processing file='%s' | schedule_id=%s: %s", video_path, schedule_id, err, exc_info=True)
