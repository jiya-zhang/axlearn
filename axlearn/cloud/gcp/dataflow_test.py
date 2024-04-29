import logging
import warnings
from typing import Any, Dict, Optional, Sequence

import apache_beam as beam
import jax
from absl import app, flags
from absl.flags import argparse_flags
from apache_beam.ml.inference.base import ModelHandler, PredictionResult, RunInference
from apache_beam.options.pipeline_options import PipelineOptions

import axlearn.common.input_fake as input_fake
import axlearn.common.launch_trainer as trainer_utils
from axlearn.common.inference import InferenceRunner, MethodRunner
from axlearn.common.utils import NestedTensor

warnings.filterwarnings("ignore")

class RunInferenceRunner(beam.DoFn):
    def __init__(self, method_runner):
        self.method_runner = method_runner

    def process(self, element):
        logging.info("RUNNING PROCESS")
        logging.info(f"Type of element:{type(element)}")
        output_list = []
        for el in element:
            output_list.append(self.method_runner(el))

        return output_list


def get_examples() -> Sequence[NestedTensor]:
    """Returns a list of fake input. You can edit this function to return your desired input.
    Fake input: https://github.com/apple/axlearn/blob/main/axlearn/common/input_fake.py#L49

    Returns:
        A list of examples of type FakeLmInput.
        Must be a Sequence since Beam expects a Sequence of examples.
        A Sequence of NestedTensor, Tensor, or other types should all work.
    """
    cfg = input_fake.FakeLmInput.default_config()
    cfg.is_training = False
    cfg.global_batch_size = 1
    cfg.total_num_batches = 1

    fake_input = input_fake.FakeLmInput(cfg)
    example_list = []
    for _ in range(cfg.total_num_batches):
        example_list.append(fake_input.__next__())

    return example_list

def parse_flags(argv):
    """Parse out arguments in addition to the defined absl flags
    (can be found in axlearn/common/launch_trainer.py).
    Addition arguments are returned to the 'main' function by 'app.run'.
    """
    parser = argparse_flags.ArgumentParser(
        description="Parser to parse additional arguments other than defiend ABSL flags."
    )
    parser.add_argument(
        "--save_main_session",
        default=True,
    )
    parser.add_argument(
        "--pickle_library",
        default="cloudpickle",
        help="Pickler library to use: dill or cloudpickle."
    )
    # Assume all remaining unknown arguments are Dataflow Pipeline options
    known_args , pipeline_args = parser.parse_known_args(argv[1:])
    pipeline_args.append(f"--pickle_library={known_args.pickle_library}")
    if known_args.save_main_session is True:
        pipeline_args.append("--save_main_session")
    print(f"pipeline_args: {pipeline_args}")
    return pipeline_args

def main(args):
    FLAGS = flags.FLAGS

    # get pipeline input
    pipeline_input = get_examples()

    # run pipeline
    print(f"args:{args}")
    pipeline_options = PipelineOptions(args)

    pipeline = beam.Pipeline(options=pipeline_options)

    module_config = trainer_utils.get_trainer_config(flag_values=FLAGS)

    # get InferenceRunner Config from Trainer Config and instantiate InferenceRunner
    inference_runner_cfg = InferenceRunner.config_from_trainer(module_config)
    inference_runner_cfg.init_state_builder.set(dir=FLAGS.trainer_dir)
    inference_runner = InferenceRunner(cfg=inference_runner_cfg, parent=None)

    # create Method Runner only once
    method_runner = inference_runner.create_method_runner(
        method="predict", prng_key=jax.random.PRNGKey(1)
    )

    with pipeline as p:
        (
            p
            | "CreateInput" >> beam.Create(pipeline_input)
            | "RunInferenceRunner" >> beam.ParDo(RunInferenceRunner(method_runner=method_runner))
            | "PrintOutput" >> beam.Map(print)
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    app.run(main, flags_parser=parse_flags)
