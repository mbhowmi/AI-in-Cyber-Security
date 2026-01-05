#!/usr/bin/env python3
"""
Preprocess wardbeck alerts to remove obvious noise and shorten prompts.

Reads:
  /project/6000603/mbhowmi/ML_CS/output/wardbeck_privacy.jsonl

Writes:
  /project/6000603/mbhowmi/ML_CS/output/wardbeck_privacy_filtered.jsonl

Current filters (conservative):
- Drop ClamAV update noise:
    rule_id == "52507"
    OR "ClamAV database update" in rule_description
"""

import json
from pathlib import Path

DEFAULT_INPUT_JSONL = "/project/6000603/mbhowmi/ML_CS/output/wardbeck_privacy.jsonl"
DEFAULT_OUTPUT_JSONL = "/project/6000603/mbhowmi/ML_CS/output/wardbeck_privacy_filtered.jsonl"


def is_noise(alert: dict) -> bool:
    """
    Return True if this alert is considered noise and should be dropped.
    """
    rule_id = str(alert.get("rule_id", "") or "")
    rule_desc = str(alert.get("rule_description", "") or "")

    # ClamAV database update noise (very frequent, clearly benign).
    if rule_id == "52507":
        return True
    if "clamav database update" in rule_desc.lower():
        return True

    return False


def preprocess(input_path: str, output_path: str) -> None:
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    dropped = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                alert = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                dropped += 1
                continue

            if is_noise(alert):
                dropped += 1
                continue

            fout.write(json.dumps(alert) + "\n")
            kept += 1

    print(f"[INFO] Total alerts read   : {total}")
    print(f"[INFO] Alerts kept        : {kept}")
    print(f"[INFO] Alerts dropped     : {dropped}")
    print(f"[INFO] Filtered alerts -> {out_path}")


def main() -> None:
    preprocess(DEFAULT_INPUT_JSONL, DEFAULT_OUTPUT_JSONL)


if __name__ == "__main__":
    main()
