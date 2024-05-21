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

"""Utils file to calculate RMS metrics for TANQ experiments."""

import dataclasses
import itertools
import json
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from pix2struct import metrics as pix2struct_metrics
from scipy import optimize
import tensorflow as tf


def _to_float(text: str) -> Optional[float]:
  try:
    if text.endswith("%"):
      # Convert percentages to floats.
      return float(text.rstrip("%")) / 100.0
    else:
      return float(text)
  except ValueError:
    return None


def _get_relative_distance(
    target: float, prediction: float, theta: float = 1.0
) -> float:
  """Returns min(1, |target-prediction|/|target|)."""
  if not target:
    return int(not prediction)
  distance = min(abs((target - prediction) / target), 1)
  return distance if distance < theta else 1


def _permute(values: tuple[str, ...], indexes: list[int]) -> tuple[str, ...]:
  return tuple(values[i] if i < len(values) else "" for i in indexes)


@dataclasses.dataclass(frozen=True)
class Table:
  """Helper class for the content of a markdown table."""

  title: Optional[str] = None
  headers: tuple[str, ...] = dataclasses.field(default_factory=tuple)
  rows: tuple[tuple[str, ...], ...] = dataclasses.field(default_factory=tuple)

  def permuted(self, indexes: list[int]) -> "Table":
    """Builds a version of the table changing the column order."""
    return Table(
        title=self.title,
        headers=_permute(self.headers, indexes),
        rows=tuple(_permute(row, indexes) for row in self.rows),
    )

  def aligned(
      self, headers: tuple[str, ...], text_theta: float = 0.5
  ) -> tuple["Table", float]:
    """Builds a column permutation with headers in the most correct order."""
    if len(headers) != len(self.headers):
      raise ValueError(f"Header length {headers} must match {self.headers}.")
    distance = []
    for h2 in self.headers:
      distance.append([
          1 - pix2struct_metrics.anls_metric(h1, h2, text_theta)
          for h1 in headers
      ])
    cost_matrix = np.array(distance)
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
    permutation = [idx for _, idx in sorted(zip(col_ind, row_ind))]
    score = (1 - cost_matrix)[permutation[1:], range(1, len(row_ind))].prod()
    return self.permuted(permutation), score


def _table_as_str(
    tanq_samples: list[Mapping[str, Any]], table_key: str
) -> list[str]:
  return [" | ".join(entry[table_key].split("|")) for entry in tanq_samples]


def evaluate_tables(
    tanq_samples: list[Mapping[str, Any]],
    table_key: str,
    gold_table_key: str,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
  """Evaluates the predicted table against the tanq gold table."""
  targets = [[t] for t in _table_as_str(tanq_samples, gold_table_key)]
  predictions = _table_as_str(tanq_samples, table_key)
  score_dict = table_datapoints_precision_recall_per_point(
      targets,
      predictions,
  )
  samples = {}
  for i, target in enumerate(targets):
    samples[tanq_samples[i]["init_qid"]] = {
        "question": tanq_samples[i]["ext_question_rephrased"],
        "target": target,
        "prediction": predictions[i],
        "precision": score_dict["precision"][i],
        "recall": score_dict["recall"][i],
        "f1": score_dict["f1"][i],
    }

  return {
      "table_datapoints_precision": (
          100.0 * sum(score_dict["precision"]) / len(targets)
      ),
      "table_datapoints_recall": (
          100.0 * sum(score_dict["recall"]) / len(targets)
      ),
      "table_datapoints_f1": 100.0 * sum(score_dict["f1"]) / len(targets),
  }, samples


def table_datapoints_precision_recall_per_point(
    targets: Sequence[Sequence[str]],
    predictions: Sequence[str],
    text_theta: float = 0.5,
    number_theta: float = 0.1,
) -> dict[str, Sequence[float]]:
  """Computes precisin recall and F1 metrics given two flattened tables.

  Parses each string into a dictionary of keys and values using row and column
  headers. Then we match keys between the two dicts as long as their relative
  levenshtein distance is below a threshold. Values are also compared with
  ANLS if strings or relative distance if they are numeric.

  Args:
    targets: list of list of strings.
    predictions: list of strings.
    text_theta: relative edit distance above this is set to the maximum of 1.
    number_theta: relative error rate above this is set to the maximum of 1.

  Returns:
    Dictionary with per-point precision, recall and F1
  """
  assert len(targets) == len(predictions)
  per_point_scores = {"precision": [], "recall": [], "f1": []}
  for pred, target in zip(predictions, targets):
    all_metrics = []
    for transposed in [True, False]:
      pred_table = _parse_table(pred, transposed=transposed)
      # pylint:disable=g-complex-comprehension
      all_metrics.extend([
          _table_datapoints_precision_recall_f1(
              _parse_table(t),
              pred_table,
              text_theta,
              number_theta,
          )
          for t in target
      ])
      # pylint:enable=g-complex-comprehension
    p, r, f = max(all_metrics, key=lambda x: x[-1])
    per_point_scores["precision"].append(p)
    per_point_scores["recall"].append(r)
    per_point_scores["f1"].append(f)
  return per_point_scores


def _parse_table(text: str, transposed: bool = False) -> Table:
  """Builds a table from a markdown representation."""
  lines = text.lower().splitlines()
  if not lines:
    return Table()
  if lines[0].startswith("title |"):
    title = lines[0][len("title |") :].strip()
    offset = 1
  else:
    title = None
    offset = 0
  if len(lines) < offset + 1:
    return Table(title=title)
  rows = []
  for line in lines[offset:]:
    rows.append(tuple(v.strip() for v in line.split(" | ")))
  if transposed:
    rows = [tuple(row) for row in itertools.zip_longest(*rows, fillvalue="")]
  return Table(title=title, headers=rows[0], rows=tuple(rows[1:]))


def _get_table_datapoints(table: Table) -> dict[str, str]:
  """Extracts a dict of datapoints from a table."""
  datapoints = {}
  if table.title is not None:
    datapoints["title"] = table.title
  if not table.rows or len(table.headers) <= 1:
    return datapoints
  for row in table.rows:
    for header, cell in zip(table.headers[1:], row[1:]):
      datapoints[f"{row[0]} {header}"] = cell
  return datapoints


def _get_datapoint_metric(
    target: tuple[str, str],
    prediction: tuple[str, str],
    text_theta=0.5,
    number_theta=0.1,
) -> float:
  """Computes a metric that scores how similar two datapoint pairs are."""
  key_metric = pix2struct_metrics.anls_metric(
      target[0], prediction[0], text_theta
  )
  pred_float = _to_float(prediction[1])
  target_float = _to_float(target[1])
  if pred_float is not None and target_float:
    return key_metric * (
        1 - _get_relative_distance(target_float, pred_float, number_theta)
    )
  elif target[1] == prediction[1]:
    return key_metric
  else:
    return key_metric * pix2struct_metrics.anls_metric(
        target[1], prediction[1], text_theta
    )


def _table_datapoints_precision_recall_f1(
    target_table: Table,
    prediction_table: Table,
    text_theta: float = 0.5,
    number_theta: float = 0.1,
) -> tuple[float, float, float]:
  """Calculates matching similarity between two tables as dicts."""
  target_datapoints = list(_get_table_datapoints(target_table).items())
  prediction_datapoints = list(_get_table_datapoints(prediction_table).items())
  if not target_datapoints and not prediction_datapoints:
    return 1, 1, 1
  if not target_datapoints:
    return 0, 1, 0
  if not prediction_datapoints:
    return 1, 0, 0
  distance = []
  for t, _ in target_datapoints:
    distance.append([
        1 - pix2struct_metrics.anls_metric(t, p, text_theta)
        for p, _ in prediction_datapoints
    ])
  cost_matrix = np.array(distance)
  row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
  score = 0
  for r, c in zip(row_ind, col_ind):
    score += _get_datapoint_metric(
        target_datapoints[r], prediction_datapoints[c], text_theta, number_theta
    )
  if score == 0:
    return 0, 0, 0
  precision = score / len(prediction_datapoints)
  recall = score / len(target_datapoints)
  return precision, recall, 2 * precision * recall / (precision + recall)


def load_jsonl_file(path: str) -> list[dict[str, Any]]:
  """Opens and parses line in a JSONL file."""
  with tf.io.gfile.GFile(path) as f:
    return [json.loads(l) for l in f]
