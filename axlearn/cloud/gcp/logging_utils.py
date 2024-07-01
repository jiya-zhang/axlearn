from ml_goodput_measurement import goodput
import time
from datetime import datetime
import os
import time
from google.api import metric_pb2
from google.cloud import compute_v1
from google.cloud import monitoring_v3

import jax

def create_recorder(run_name):
    """Returns GoodPut Recorder instance."""
    goodput_logger_name = f'goodput_{run_name}'
    return goodput.GoodputRecorder(
        job_name=run_name,
        logger_name=goodput_logger_name,
        logging_enabled=(jax.process_index() == 0)
    )


def write_goodput():
    """Writes current GoodPut to Cloud Monitoring"""
