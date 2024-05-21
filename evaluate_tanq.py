# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Calculates RMS metrics for TANQ experiments."""

import json
from typing import Any, Mapping, Sequence

from absl import app
from absl import flags
import tensorflow as tf

from tanq import tanq_utils


_INPUT_PATH = flags.DEFINE_string("input_path", None, "Input path.")
_PREDICTION_KEY = flags.DEFINE_string(
    "prediction_key", None, "Key of the prediction table to be evaluated."
)
_OUTPUT_PATH = flags.DEFINE_string("output_path", None, "Output path.")
_PATH_SAMPLE_RESULTS = flags.DEFINE_string(
    "sample_output_path", None, "Samples output path."
)
_GOLD_TABLE_KEY = "answer_table"


def _run_eval(
    tanq_samples: list[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
  return tanq_utils.evaluate_tables(
      tanq_samples,
      table_key=_PREDICTION_KEY.value,
      gold_table_key=_GOLD_TABLE_KEY,
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  tanq = tanq_utils.load_jsonl_file(_INPUT_PATH.value)
  eval_results, samplewise_results = _run_eval(tanq)

  with tf.io.gfile.GFile(_OUTPUT_PATH.value, "w") as file:
    json.dump(eval_results, file, indent=3)

  with tf.io.gfile.GFile(_PATH_SAMPLE_RESULTS.value, "w") as file:
    json.dump(samplewise_results, file, indent=3)


if __name__ == "__main__":
  app.run(main)
