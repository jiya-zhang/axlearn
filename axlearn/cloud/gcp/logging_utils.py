from ml_goodput_measurement import goodput
import time
from datetime import datetime
import os
import time
import sys
from google.api import metric_pb2
from google.cloud import compute_v1
from google.cloud import monitoring_v3
from google.cloud import bigquery
from absl import logging

import jax

def get_run_name(dir):
    return dir.split("/")[-3]

def write_restore_time_to_bq(use_orbax, step, restore_time):
    # TODO: consider only writing restore time if jax.process_index()==0
    # Currently this will likely record one restore time per worker
    project_id = os.environ['PROJECT_ID']
    run_name = os.environ['RUN_NAME']
    client = bigquery.Client()
    bigquery_table = client.get_table(project_id + '.axlearn.orbax_testing_results')
    row_data = {
        'run_name': run_name,
        'use_orbax': use_orbax,
        'step': step,
        'restore_time': restore_time
    }
    client.insert_rows(
        table = bigquery_table,
        rows = [row_data]
    )

class GoodPutManager():
    _recorder = None
    _calculator = None
    _run_name = None
    _bigquery_client = None
    _bigquery_table = None

    def __init__(self, run_name, project_name):
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
        self._bigquery_client = bigquery.Client()
        self._bigquery_table = self._bigquery_client.get_table(project_name + '.axlearn.orbax_testing_results')

    def record_step_start_time(self, step):
        if jax.process_index() == 0:
            self._recorder.record_step_start_time(step)

    def record_job_start_time(self):
        # TODO(maggiejz): record only if job start time doesn't exist already
        if jax.process_index() == 0:
            self._recorder.record_job_start_time()
            logging.info(f"Recorded job start time for: {self._run_name}")
        else:
            logging.info("Process index non zero. Did not record job start time.")

    def record_job_end_time(self):
        if jax.process_index() == 0:
            self._recorder.record_job_end_time()
            logging.info(f"Recorded job end time for: {self._run_name}")
        else:
            logging.info("Process index non zero. Did not record job end time.")

    def get_goodput(self):
        if jax.process_index() == 0:
            return self._calculator.get_job_goodput()[0]
        else:
            return None

    def write_metrics_to_bq(self, run_name, use_orbax, step, step_time, goodput):
        row_data = {
            'run_name': run_name,
            'use_orbax': use_orbax,
            'step': step,
            'running_avg_step_time': step_time,
            'running_goodput': goodput
        }
        self._bigquery_client.insert_rows(
            table = self._bigquery_table,
            rows = [row_data]
        )
