from ml_goodput_measurement import goodput
import time
from datetime import datetime
import os
import time
from google.api import metric_pb2
from google.cloud import compute_v1
from google.cloud import monitoring_v3
from absl import logging

import jax

def get_run_name(dir):
    return dir.split("/")[-3]

class GoodPutManager():
    _recorder = None
    _calculator = None
    _run_name = None

    def __init__(self, run_name):
        """Returns GoodPut Calculator instance."""
        self._run_name = run_name
        goodput_logger_name = f'goodput_{run_name}'
        self._recorder = goodput.GoodputRecorder(
        job_name=run_name,
        logger_name=goodput_logger_name,
        logging_enabled=(jax.process_index() == 0)
        )
        self._calculator = goodput.GoodputCalculator(
            job_name=run_name,
            logger_name=goodput_logger_name
        )

    def record_step_start_time(self, step):
        self._recorder.record_step_start_time(step)

    def record_job_start_time(self):
        self._recorder.record_job_start_time()
        logging.info(f"Recorded job start time for: {self._run_name}")

    def record_job_end_time(self):
        self._recorder.record_job_end_time()
        logging.info(f"Recorded job end time for: {self._run_name}")

    def get_goodput(self):
        return self._calculator.get_job_goodput()

    def write_goodput_to_cloud(self):
        """Writes current GoodPut to Cloud Monitoring"""
