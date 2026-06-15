"""Shared JSONL I/O helpers."""

import json
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict]:
    """Load records from a JSONL file. Each line is {"input": {...}, "output": {...}}.

    Blank lines are skipped; a torn final line from a crashed writer is tolerated.
    """
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
