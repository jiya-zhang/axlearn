from ml_goodput_measurement import goodput
import time
from datetime import datetime
import os
import time
import sys
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
    _metric_name = None
    _project = None

    def __init__(self, run_name, project_name):
        """Returns GoodPut Calculator instance."""
        self._project = project_name
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

    def write_goodput_to_cloud(self, step, goodput):
        """Writes current GoodPut to Cloud Monitoring"""
        client = monitoring_v3.MetricServiceClient()

        seconds_since_epoch_utc = time.time()
        nanos_since_epoch_utc = int(
            (seconds_since_epoch_utc - int(seconds_since_epoch_utc)) * 10**9
        )
        interval = monitoring_v3.types.TimeInterval({
            "end_time": {
                "seconds": int(seconds_since_epoch_utc),
                "nanos": nanos_since_epoch_utc,
            }
        })
        event_time = time.strftime(
            "%d %b %Y %H:%M:%S UTC", time.gmtime(seconds_since_epoch_utc)
        )

        series = monitoring_v3.types.TimeSeries()
        series.metric.type = "custom.googleapis.com/" + self._metric_name
        series.resource.type = "generic_node"
        series.resource.labels["location"] = "us-central2-b"
        series.resource.labels["namespace"] = "namespace"
        series.resource.labels["node_id"] = "node_id"
        series.metric.labels["goodput"] = str(goodput)
        series.metric.labels["job_name"] = step
        series.metric.labels["event_time"] = event_time
        series.points = [
            monitoring_v3.types.Point(
                interval=interval,
                value=monitoring_v3.types.TypedValue(double_value=goodput),
            )
        ]

        client.create_time_series(name=self._project, time_series=[series])


    def set_up_metric(self):
        """Create a custom metric for Goodput"""
        if self._run_name is None:
            logging.info("Run name is not specified. Exiting...")
            sys.exit()
        self._metric_name = f'goodput_metric_{self._run_name}'
        client = monitoring_v3.MetricServiceClient()

        descriptor = metric_pb2.MetricDescriptor()
        descriptor.type = "custom.googleapis.com/" + self._metric_name
        descriptor.metric_kind = metric_pb2.MetricDescriptor.MetricKind.GAUGE
        descriptor.value_type = metric_pb2.MetricDescriptor.ValueType.DOUBLE
        descriptor.description = "Goodput of the job."

        descriptor = client.create_metric_descriptor(
            name=self._project, metric_descriptor=descriptor
        )
        logging.info(f"Created custom metric: {descriptor.name}.")
