"""
Alpaca-style local benchmark for performance evaluation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_simple_sgl_function


def _format_alpaca_prompt(record: Dict[str, Any]) -> str:
    instruction = str(record.get("instruction", "")).strip()
    input_text = str(record.get("input", "")).strip()
    if instruction:
        if input_text:
            return f"{instruction}\n\nInput:\n{input_text}"
        return instruction

    conversations = record.get("conversations")
    if isinstance(conversations, list) and conversations:
        rendered_messages = []
        for message in conversations:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user")).strip().lower()
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role == "system":
                rendered_messages.append(f"System: {content}")
            elif role == "assistant":
                rendered_messages.append(f"Assistant: {content}")
            else:
                rendered_messages.append(f"User: {content}")
        if rendered_messages:
            return "\n\n".join(rendered_messages)

    text = str(record.get("text", "")).strip()
    if text:
        return text

    raise ValueError(
        "Unsupported alpaca record. Expected instruction/input, conversations, or text."
    )


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

    raise ValueError(f"Unsupported alpaca dataset format: {path.suffix}")


@BENCHMARKS.register("alpaca")
class AlpacaBenchmarker(Benchmarker):
    """Alpaca benchmark for speedup and accept length only."""

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        dataset_path: Optional[str] = None,
    ):
        super().__init__(num_samples, subset)
        self.dataset_path = dataset_path

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Any]]:
        if self.dataset_path:
            records = _load_local_records(self.dataset_path)
        else:
            dataset = load_dataset("tatsu-lab/alpaca", split="train")
            records = [dict(item) for item in dataset]

        questions = []
        for idx, record in enumerate(records):
            if self.num_samples is not None and idx >= self.num_samples:
                break
            questions.append({"question": _format_alpaca_prompt(record)})

        return questions, []

    def create_sgl_function(self):
        return create_simple_sgl_function(function_name="get_alpaca_answer")
