"""
LongBench-style local benchmark for performance evaluation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_simple_sgl_function


def _render_options(options: Any) -> str:
    if isinstance(options, dict):
        ordered = []
        for key in ["A", "B", "C", "D"]:
            value = str(options.get(key, "")).strip()
            if value:
                ordered.append(f"{key}. {value}")
        if ordered:
            return "\n".join(ordered)
        return ""

    if isinstance(options, list):
        rendered = []
        for idx, option in enumerate(options):
            label = chr(ord("A") + idx)
            value = str(option).strip()
            if value:
                rendered.append(f"{label}. {value}")
        return "\n".join(rendered)

    return ""


def _format_longbench_prompt(record: Dict[str, Any]) -> str:
    context = str(record.get("context", "")).strip()
    question = str(record.get("question", record.get("input", ""))).strip()
    if not context:
        raise ValueError("LongBench record is missing context")
    if not question:
        raise ValueError("LongBench record is missing question/input")

    prompt = f"{context}\n\nQuestion: {question}"
    options_text = _render_options(record.get("options"))
    if options_text:
        prompt += f"\n{options_text}"
    return prompt


def _load_local_records(dataset_path: str) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON list")
        return records

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number} must contain JSON objects")
                records.append(record)
        return records

    raise ValueError(f"Unsupported LongBench dataset format: {path.suffix}")


@BENCHMARKS.register("longbench")
class LongBenchBenchmarker(Benchmarker):
    """LongBench benchmark for speedup and accept length only."""

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        dataset_path: Optional[str] = None,
    ):
        super().__init__(num_samples, subset)
        self.dataset_path = dataset_path

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Any]]:
        if not self.dataset_path:
            raise ValueError(
                "longbench requires a local dataset path via --benchmark-data-paths longbench=<path>"
            )

        records = _load_local_records(self.dataset_path)
        questions = []
        for idx, record in enumerate(records):
            if self.num_samples is not None and idx >= self.num_samples:
                break
            questions.append({"question": _format_longbench_prompt(record)})

        return questions, []

    def create_sgl_function(self):
        return create_simple_sgl_function(function_name="get_longbench_answer")
