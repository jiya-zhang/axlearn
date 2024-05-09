from absl import app, flags
from absl.flags import argparse_flags
import logging

from typing import Dict
from typing import Iterable
from typing import Tuple

import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFAutoModelForMaskedLM

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.huggingface_inference import HuggingFaceModelHandlerKeyedTensor


import warnings
warnings.filterwarnings("ignore")

class CustomProcessFn(beam.DoFn):
    def process(self, element):
        if tf.config.list_physical_devices('GPU'):
            print("A GPU is attached and available.")
        else:
            print("No GPU is attached or available.")
        yield

def parse_flags(argv):
    """Parse out arguments in addition to the defined absl flags
    (can be found in axlearn/common/launch_trainer.py).
    Addition arguments are returned to the 'main' function by 'app.run'.
    """
    parser = argparse_flags.ArgumentParser(
        description="Parser to parse additional arguments other than defined ABSL flags."
    )
    parser.add_argument("--save_main_session", default=True)

    # The default pickler is dill and cannot pickle absl FlagValues. Use cloudpickle instead.
    parser.add_argument("--pickle_library", default="cloudpickle")

    # Assume all remaining unknown arguments are Dataflow Pipeline options
    known_args , pipeline_args = parser.parse_known_args(argv[1:])
    if known_args.save_main_session is True:
        pipeline_args.append("--save_main_session")
    pipeline_args.append(f"--pickle_library={known_args.pickle_library}")
    return pipeline_args

def main(args):
    #logging.info(f"Pipeline options: {args}")
    FLAGS = flags.FLAGS

    text = ["example text."]

    # run pipeline
    pipeline_options = PipelineOptions(args)

    with beam.Pipeline(options=pipeline_options) as p:
        (
            p
            | "CreateExamples" >> beam.Create(text)
            | "GetJaxDevices" >> beam.ParDo(CustomProcessFn())
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    app.run(main, flags_parser=parse_flags)