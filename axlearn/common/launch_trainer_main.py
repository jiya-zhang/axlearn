# Copyright © 2023 Apple Inc.

"""Main function for launching the trainer."""
from absl import app

from axlearn.common import launch, launch_trainer
from ml_goodput_measurement import goodput
from axlearn.cloud.gcp import logging_utils
import jax

def main(_):
    launch.setup()

    # create Goodput Manager
    goodput_manager = logging_utils.GoodPutManager(run_name='20240701-01')
    # record job's overall start time
    goodput_manager.record_job_start_time()

    trainer_config = launch_trainer.get_trainer_config()
    launch_trainer.run_trainer(trainer_config)

    # record job's overall stop time
    goodput_manager.record_job_end_time()


if __name__ == "__main__":
    app.run(main)
