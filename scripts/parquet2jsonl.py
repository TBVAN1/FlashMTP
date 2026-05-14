"""Convert parquet files to JSONL for regenerate_train_data.py.

Output schema (same for all modes):
{
	"id": int,
	"conversations": [
		{"role": str, "content": str}
	]
}

Modes:
- Nemotron: --input-dir with multiple parquet files; balanced sampling by ``category``.
- Orca Math: --input-parquet (e.g. train-00000-of-00001.parquet); random sample by
  ``question``; gold ``answer`` is dropped so the target model can regenerate responses.

Default output names: Nemotron_{num_samples}.jsonl / OrcaMath_{num_samples}.jsonl
"""

import argparse
import glob
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import pyarrow.parquet as pq
from tqdm import tqdm


VALID_ROLES = {"system", "user", "assistant"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	src = parser.add_mutually_exclusive_group(required=True)
	src.add_argument(
		"--input-dir",
		type=str,
		help="Directory of Nemotron parquet files (*.parquet).",
	)
	src.add_argument(
		"--input-parquet",
		type=str,
		help="Single Orca Math parquet path (columns: question, answer).",
	)
	parser.add_argument("--output-dir", type=str, required=True)
	parser.add_argument("--num-samples", type=int, required=True)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--output-stem",
		type=str,
		default=None,
		help="Output basename without .jsonl (default: Nemotron_* or OrcaMath_*).",
	)
	return parser.parse_args()


def iter_rows(parquet_path: str) -> Iterable[Dict[str, Any]]:
	pf = pq.ParquetFile(parquet_path)
	for batch in pf.iter_batches():
		for row in batch.to_pylist():
			yield row


def normalize_conversations(messages: Any) -> Optional[List[Dict[str, str]]]:
	if not isinstance(messages, list):
		return None

	convs: List[Dict[str, str]] = []
	for m in messages:
		if not isinstance(m, dict):
			continue
		role = str(m.get("role", "")).strip().lower()
		if role not in VALID_ROLES:
			continue
		content = m.get("content", "")
		if not isinstance(content, str):
			content = json.dumps(content, ensure_ascii=False)
		if role == "system" and content.strip() == "":
			continue
		convs.append({"role": role, "content": content})

	if not convs:
		return None

	# regenerate_train_data expects the first non-system role to be user.
	start = 0
	if convs[0]["role"] == "system":
		start = 1
	if start >= len(convs) or convs[start]["role"] != "user":
		return None

	has_assistant = any(x["role"] == "assistant" for x in convs)
	if not has_assistant:
		return None
	return convs


def orca_row_to_conversations(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
	"""Single-turn user message from Orca Math question (matches regenerate_train_data)."""
	q = row.get("question")
	if not isinstance(q, str) or not q.strip():
		return None
	return [{"role": "user", "content": q.strip()}]


def sample_orca_math(parquet_path: str, num_samples: int, seed: int) -> List[List[Dict[str, str]]]:
	"""Reservoir sample valid rows in one pass (memory-friendly on large shards)."""
	rng = random.Random(seed)
	if not os.path.isfile(parquet_path):
		raise FileNotFoundError(parquet_path)

	reservoir: List[List[Dict[str, str]]] = []
	seen_valid = 0
	for row in tqdm(iter_rows(parquet_path), desc="Reading Orca Math parquet"):
		convs = orca_row_to_conversations(row)
		if convs is None:
			continue
		seen_valid += 1
		if len(reservoir) < num_samples:
			reservoir.append(convs)
		else:
			j = rng.randint(0, seen_valid - 1)
			if j < num_samples:
				reservoir[j] = convs

	if not reservoir:
		raise RuntimeError("No valid Orca Math rows (non-empty question)")
	if seen_valid < num_samples:
		print(
			f"warning: only {seen_valid} valid rows (< num_samples {num_samples}); "
			"using all available"
		)

	rng.shuffle(reservoir)
	return reservoir


def sample_nemotron(input_dir: str, num_samples: int, seed: int) -> tuple[List[Dict[str, Any]], List[str]]:
	random.seed(seed)
	parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
	if not parquet_files:
		raise FileNotFoundError(f"No parquet files found in {input_dir}")

	buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
	sample_key = 0

	for parquet_path in tqdm(parquet_files, desc="Reading parquet"):
		for row in iter_rows(parquet_path):
			task_type = row.get("category")
			if not isinstance(task_type, str) or not task_type:
				continue

			convs = normalize_conversations(row.get("messages"))
			if convs is None:
				continue

			buckets[task_type].append({"_key": sample_key, "conversations": convs})
			sample_key += 1

	task_types = sorted([k for k, v in buckets.items() if len(v) > 0])
	if not task_types:
		raise RuntimeError("No valid samples found after filtering")

	# Balanced allocation: average random sampling among task types.
	base = num_samples // len(task_types)
	rem = num_samples % len(task_types)

	selected: List[Dict[str, Any]] = []
	for i, t in enumerate(task_types):
		need = base + (1 if i < rem else 0)
		pool = buckets[t]
		if len(pool) <= need:
			chosen = pool
		else:
			chosen = random.sample(pool, need)
		selected.extend(chosen)

	# If total is still short (some tasks have too few samples), fill from leftovers.
	if len(selected) < num_samples:
		used_keys = {x["_key"] for x in selected}
		leftovers: List[Dict[str, Any]] = []
		for t in task_types:
			for item in buckets[t]:
				if item["_key"] not in used_keys:
					leftovers.append(item)
		random.shuffle(leftovers)
		need_more = num_samples - len(selected)
		selected.extend(leftovers[:need_more])

	random.shuffle(selected)
	if len(selected) > num_samples:
		selected = selected[:num_samples]

	return selected, task_types


def main() -> None:
	args = parse_args()
	os.makedirs(args.output_dir, exist_ok=True)

	if args.input_parquet:
		selected_convs = sample_orca_math(args.input_parquet, args.num_samples, args.seed)
		stem = args.output_stem or f"OrcaMath_{args.num_samples}"
		task_types: Optional[List[str]] = None
	else:
		assert args.input_dir is not None
		selected, task_types = sample_nemotron(args.input_dir, args.num_samples, args.seed)
		selected_convs = [x["conversations"] for x in selected]
		stem = args.output_stem or f"Nemotron_{args.num_samples}"

	output_path = os.path.join(args.output_dir, f"{stem}.jsonl")
	with open(output_path, "w", encoding="utf-8") as f:
		for idx, convs in enumerate(selected_convs):
			out_row = {"id": idx, "conversations": convs}
			f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

	if task_types is not None:
		print(f"task_types: {task_types}")
	print(f"written: {len(selected_convs)}")
	print(f"output: {output_path}")


if __name__ == "__main__":
	main()
