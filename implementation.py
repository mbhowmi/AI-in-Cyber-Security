#!/usr/bin/env python3
"""
Batch-level attack detection using local LLaMA (llama.cpp), text output.

For each chronological batch of wardbeck alerts, LLaMA returns a single line:
  ATTACK: yes/no; TITLE: short title; STAGES: stage1, stage2, ...

Stages are from the wardbeck/MAD-LLM set:
  [network_scans, service_scans, wpscan, dirb, webshell,
   cracking, reverse_shell, privilege_escalation, service_stop, dnsteal]

Input:
- /project/6000603/mbhowmi/ML_CS/output/wardbeck_privacy.jsonl

Output:
- /project/6000603/mbhowmi/ML_CS/output/wardbeck_batch_text_detection_v2.csv
"""

import argparse
import json
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

LLAMA_BIN = "/home/mbhowmi/projects/def-bauer/mbhowmi/ML_CS/llama.cpp/build/bin/llama-cli"
DEFAULT_MODEL_PATH = "/home/mbhowmi/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

DEFAULT_INPUT_JSONL = "/project/6000603/mbhowmi/ML_CS/output/wardbeck_privacy_filtered.jsonl"
DEFAULT_OUTPUT_CSV  = "/project/6000603/mbhowmi/ML_CS/output/wardbeck_batch_text_detection_v2.csv"

DEFAULT_CTX_SIZE = 4096
DEFAULT_N_PREDICT = 96
DEFAULT_THREADS = 16
DEFAULT_N_GPU_LAYERS = 999

DEFAULT_BATCH_SIZE = 60
MAX_CHARS_PER_ALERT = 120
MAX_PROMPT_CHARS = 4000


STAGE_LIST = [
    "network_scans",
    "service_scans",
    "wpscan",
    "dirb",
    "webshell",
    "cracking",
    "reverse_shell",
    "privilege_escalation",
    "service_stop",
    "dnsteal",
]


def _shorten(text: str, max_len: int) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 10] + "...[TRUNC]"


def build_compact_line(idx: int, alert: Dict[str, Any]) -> str:
    ts = alert.get("timestamp", "")
    src = alert.get("source", "")
    rule_id = alert.get("rule_id", "")
    rule_desc = _shorten(alert.get("rule_description", ""), 40)
    msg = _shorten(alert.get("message", ""), 40)

    base = f"[{idx}] ts={ts} src={src} rule_id={rule_id}"
    extras = []
    if rule_desc:
        extras.append(f"rule_desc={rule_desc}")
    if msg:
        extras.append(f"msg={msg}")

    line = base
    if extras:
        line += " " + " ".join(extras)

    if len(line) > MAX_CHARS_PER_ALERT:
        line = line[: MAX_CHARS_PER_ALERT - 10] + "...[TRUNC]"
    return line


def build_batch_prompt(batch_idx: int, batch_alerts: List[Dict[str, Any]]) -> str:
    header = f"""
You are a cyber security analyst.

You will see a chronological batch of IDS alerts from the wardbeck scenario.
Each alert has an INDEX from 0 to {len(batch_alerts) - 1}.

Your task for THIS BATCH ONLY:
1. Decide whether these alerts indicate a multi-stage attack (yes/no).
2. If yes, give a short ATTACK TITLE for the overall pattern
   (for example: "webshell + cracking" or "scans only").
3. If yes, list which ATTACK STAGES from this list appear in this batch:
   [network_scans, service_scans, wpscan, dirb, webshell,
    cracking, reverse_shell, privilege_escalation, service_stop, dnsteal].
4. Use only the stage names from the list above.

IMPORTANT OUTPUT FORMAT:
- Output exactly ONE line of plain text.
- Use this format, replacing the values:

  ATTACK: yes; TITLE: webshell + cracking; STAGES: network_scans, dirb, webshell, cracking

If there is no attack, write:

  ATTACK: no; TITLE: none; STAGES: none

Do NOT write anything else.

Below are the alerts in this batch (batch index = {batch_idx}).
"""

    parts: List[str] = [textwrap.dedent(header).strip(), ""]

    for idx, alert in enumerate(batch_alerts):
        parts.append(build_compact_line(idx, alert))

    prompt = "\n".join(parts)

    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = (
            prompt[: MAX_PROMPT_CHARS - 100]
            + "\n...[TRUNCATED ALERTS]\nRemember to output one line: ATTACK: yes/no; TITLE: ...; STAGES: ..."
        )

    return prompt


def run_llama_cli(
    model_path: str,
    prompt: str,
    ctx_size: int = DEFAULT_CTX_SIZE,
    n_predict: int = DEFAULT_N_PREDICT,
    threads: int = DEFAULT_THREADS,
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
) -> str:
    cmd = [
        LLAMA_BIN,
        "-m", model_path,
        "-c", str(ctx_size),
        "-n", str(n_predict),
        "-t", str(threads),
        "-ngl", str(n_gpu_layers),
        "-p", prompt,
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="ignore",
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-cli failed with exit code {proc.returncode}\n"
            f"STDERR:\n{proc.stderr}"
        )

    return proc.stdout.strip()


def parse_attack_line(text_out: str) -> Dict[str, Any]:
    """
    Parse a line like:
      ATTACK: yes; TITLE: webshell + cracking; STAGES: network_scans, dirb, webshell, cracking
    Robust to extra lines: uses the LAST line starting with 'ATTACK:'.
    """
    result = {"attack": None, "title": "", "stages": []}

    if not text_out:
        return result

    lines = [ln.strip() for ln in text_out.splitlines() if ln.strip()]
    attack_line = None
    for ln in reversed(lines):
        if ln.lower().startswith("attack:"):
            attack_line = ln
            break

    if attack_line is None:
        return result

    line = attack_line

    try:
        # Split into segments separated by ';'
        segments = [seg.strip() for seg in line.split(";")]

        attack_val = None
        title_val = ""
        stages_val = ""

        for seg in segments:
            lower = seg.lower()
            if lower.startswith("attack"):
                # ATTACK: yes / no
                if ":" in seg:
                    attack_str = seg.split(":", 1)[1].strip().lower()
                else:
                    attack_str = seg.strip().lower()
                if attack_str.startswith("y"):
                    attack_val = True
                elif attack_str.startswith("n"):
                    attack_val = False
            elif lower.startswith("title"):
                # TITLE: ...
                if ":" in seg:
                    title_val = seg.split(":", 1)[1].strip()
            elif lower.startswith("stages"):
                # STAGES: ...
                if ":" in seg:
                    stages_val = seg.split(":", 1)[1].strip().lower()

        result["attack"] = attack_val
        result["title"] = title_val

        # Parse stages
        if stages_val and stages_val != "none":
            raw_stages = [s.strip() for s in stages_val.split(",")]
            norm_stages = []
            for s in raw_stages:
                for ref in STAGE_LIST:
                    if s == ref.lower():
                        norm_stages.append(ref)
                        break
            result["stages"] = list(dict.fromkeys(norm_stages))
        else:
            result["stages"] = []

    except Exception:
        pass

    return result


def load_alerts_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                alerts.append(json.loads(line))
    return alerts


def build_batches(alerts: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    for i in range(0, len(alerts), batch_size):
        batches.append(alerts[i : i + batch_size])
    return batches


def batch_time_bounds(batch_alerts: List[Dict[str, Any]]) -> (float, float):
    """
    Return (start_ts, end_ts) in epoch seconds for this batch,
    using the first and last alert timestamps.
    """
    if not batch_alerts:
        return None, None

    # Alerts are already chronological.
    ts_first = batch_alerts[0].get("timestamp")
    ts_last = batch_alerts[-1].get("timestamp")

    def to_epoch(ts):
        if ts is None:
            return None
        try:
            return float(ts)
        except (TypeError, ValueError):
            dt = pd.to_datetime(ts, utc=True)
            return dt.timestamp()

    return to_epoch(ts_first), to_epoch(ts_last)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        default=DEFAULT_INPUT_JSONL,
        help="Input JSONL file with wardbeck alerts (privacy-preserved).",
    )
    parser.add_argument(
        "--output_csv",
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV for batch-level attack detection (text).",
    )
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help="GGUF model path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Alerts per batch (default 60).",
    )
    parser.add_argument(
        "--ctx_size",
        type=int,
        default=DEFAULT_CTX_SIZE,
        help="llama.cpp context size (default 4096).",
    )
    parser.add_argument(
        "--n_predict",
        type=int,
        default=DEFAULT_N_PREDICT,
        help="Number of tokens to generate (default 96).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="CPU threads for llama.cpp.",
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=DEFAULT_N_GPU_LAYERS,
        help="Number of layers to offload to GPU.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    print(f"[INFO] Loading alerts from {input_path}")
    alerts = load_alerts_from_jsonl(input_path)
    print(f"[INFO] Loaded {len(alerts)} alerts")

    batches = build_batches(alerts, args.batch_size)
    print(f"[INFO] Built {len(batches)} batches (batch_size={args.batch_size})")

    rows: List[Dict[str, Any]] = []

    for b_idx, batch_alerts in enumerate(batches):
        print(f"[INFO] Processing batch {b_idx}/{len(batches)-1} (size={len(batch_alerts)})")
        prompt = build_batch_prompt(b_idx, batch_alerts)
        print(f"[DEBUG] Prompt length (chars) for batch {b_idx}: {len(prompt)}")

        batch_start_ts, batch_end_ts = batch_time_bounds(batch_alerts)

        try:
            raw = run_llama_cli(
                model_path=args.model_path,
                prompt=prompt,
                ctx_size=args.ctx_size,
                n_predict=args.n_predict,
                threads=args.threads,
                n_gpu_layers=args.n_gpu_layers,
            )
            parsed = parse_attack_line(raw)
            attack = parsed["attack"]
            title = parsed["title"]
            stages = parsed["stages"]
            status = "ok" if attack is not None else "parse_error"
            error_msg = "" if status == "ok" else "could_not_parse_attack_line"
        except Exception as e:
            raw = ""
            attack = None
            title = ""
            stages = []
            status = "llama_error"
            error_msg = str(e)

        rows.append(
            {
                "batch_index": b_idx,
                "batch_size": len(batch_alerts),
                "batch_start_ts": batch_start_ts,
                "batch_end_ts": batch_end_ts,
                "prompt_chars": len(prompt),
                "llm_raw_output": raw,
                "attack": attack,
                "attack_title": title,
                "stages_list": ",".join(stages),
                "status": status,
                "error_msg": error_msg,
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved batch detection to {out_path}")
    

if __name__ == "__main__":
    main()
