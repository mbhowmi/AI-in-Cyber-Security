#!/usr/bin/env python3
"""
MAD-LLM-style evaluation for wardbeck using batch-level LLM outputs.

Inputs:
- /home/mbhowmi/projects/def-bauer/mbhowmi/ML_CS/ait_Dataset/labels.csv
    Ground-truth attack stages for all AIT scenarios (including wardbeck).
- /project/6000603/mbhowmi/ML_CS/output/wardbeck_privacy.jsonl
    Privacy-preserved wardbeck alerts, chronological.
- /project/6000603/mbhowmi/ML_CS/output/wardbeck_batch_text_detection_v2.csv
    LLM batch predictions (attack yes/no, title, stages_list, status).

Outputs:
- /project/6000603/mbhowmi/ML_CS/output/wardbeck_stage_metrics_edited.csv
    Per-stage precision/recall/F1 for wardbeck.
- Prints macro-averaged metrics (over the 10 stages).
"""

import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd


LABELS_CSV = "/home/mbhowmi/projects/def-bauer/mbhowmi/ML_CS/ait_Dataset/labels.csv"
WARD_BECK_ALERTS_JSONL = "/project/6000603/mbhowmi/ML_CS/output/wardbeck_privacy_filtered.jsonl"
WARD_BECK_BATCH_CSV = "/project/6000603/mbhowmi/ML_CS/output/wardbeck_batch_text_detection_v2.csv"
OUT_STAGE_METRICS_CSV = "/project/6000603/mbhowmi/ML_CS/output/wardbeck_stage_metrics_v2_edited.csv"

SCENARIO_NAME = "wardbeck"


def load_ground_truth(labels_csv: str, scenario: str) -> pd.DataFrame:
    """
    Load ground-truth stages for the given scenario from AIT labels.csv.

    Expected columns in labels.csv: [scenario, attack, start, end]
    where 'attack' is the stage name and 'start','end' are epoch seconds.
    """
    df = pd.read_csv(labels_csv)
    df = df[df["scenario"] == scenario].copy()
    df.rename(columns={"attack": "stage"}, inplace=True)
    return df[["stage", "start", "end"]]


def load_alert_timestamps(jsonl_path: str) -> List[float]:
    """
    Load wardbeck alerts and return list of timestamps in epoch seconds.

    Assuming each JSON line has a 'timestamp' field that is either:
    - numeric epoch seconds (string or number), or
    - ISO8601 timestamp convertible by pandas.to_datetime.
    """
    timestamps: List[float] = []
    jsonl_path = Path(jsonl_path)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ts = obj.get("timestamp", None)
            if ts is None:
                continue
            # Try numeric epoch first
            try:
                t_val = float(ts)
                timestamps.append(t_val)
            except (TypeError, ValueError):
                # Assume ISO8601 string
                dt = pd.to_datetime(ts, utc=True)
                timestamps.append(dt.timestamp())

    return timestamps


def derive_batch_time_windows(
    alert_timestamps: List[float],
    batch_size: int,
) -> List[Tuple[float, float]]:
    """
    Given per-alert timestamps (chronological) and batch_size,
    derive [start, end] time window for each batch index.

    For batch b:
      alerts indices [b*batch_size : (b+1)*batch_size)
      batch_start = timestamp of first alert in batch
      batch_end   = timestamp of last alert in batch
    """
    n_alerts = len(alert_timestamps)
    windows: List[Tuple[float, float]] = []

    for b_start in range(0, n_alerts, batch_size):
        b_end = min(b_start + batch_size, n_alerts) - 1
        if b_start >= n_alerts:
            break
        start_ts = alert_timestamps[b_start]
        end_ts = alert_timestamps[b_end]
        windows.append((start_ts, end_ts))

    return windows


def load_batch_predictions(batch_csv: str) -> pd.DataFrame:
    """
    Load LLM batch predictions CSV for wardbeck.

    Expected columns:
      batch_index, batch_size, prompt_chars, llm_raw_output,
      attack (True/False), attack_title, stages_list, status, error_msg

    stages_list is a comma-separated list of stages or empty string.
    Only rows with status == "ok" are used.
    """
    df = pd.read_csv(batch_csv)

    # Keeping only successful rows
    df_ok = df[df["status"] == "ok"].copy()

    # Normalize stages_list -> list
    def parse_stage_list(s: str):
        if pd.isna(s) or not s:
            return []
        return [x.strip() for x in str(s).split(",") if x.strip()]

    df_ok["stages"] = df_ok["stages_list"].apply(parse_stage_list)
    return df_ok


def intervals_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """
    Closed interval overlap check.
    """
    return (a_start <= b_end) and (b_start <= a_end)


def compute_stage_metrics(
    gt: pd.DataFrame,
    batch_df: pd.DataFrame,
    batch_windows: List[Tuple[float, float]],
) -> pd.DataFrame:
    """
    Computing per-stage precision/recall/F1 using time overlap, inspired by MAD-LLM.

    For each ground-truth stage S:
      - gt_interval(s) = [start, end] rows in gt where stage == S.
      - predicted_S_batches = all batches whose 'stages' list contains S.

    Recall for S:
      - TP_recall_S: number of gt intervals for S that are covered by at least
        one predicted_S_batch whose time window overlaps that interval.
      - FN_recall_S: number of gt intervals for S not covered by any such batch.

    Precision for S:
      - For each predicted_S_batch, check if its time window overlaps any gt
        interval for S.
      - TP_precision_S: number of predicted_S_batches that overlap >= 1 gt interval.
      - FP_precision_S: number of predicted_S_batches that do not overlap any gt interval.

    Then:
      recall_S = TP_recall_S / (TP_recall_S + FN_recall_S)
      precision_S = TP_precision_S / (TP_precision_S + FP_precision_S)
      F1_S is the harmonic mean of precision_S and recall_S.
    """
    # Ensuring batch_windows length matches max batch_index + 1
    max_idx = int(batch_df["batch_index"].max())
    if len(batch_windows) <= max_idx:
        raise ValueError(
            f"Not enough batch windows ({len(batch_windows)}) for max batch_index {max_idx}"
        )

    stages = sorted(gt["stage"].unique())
    records = []

    for stage in stages:
        # Ground truth intervals for this stage
        gt_s = gt[gt["stage"] == stage]

        # Batches that predicted this stage
        pred_s = batch_df[batch_df["stages"].apply(lambda lst: stage in lst)]

        # --- Recall ---
        tp_recall = 0
        fn_recall = 0
        for _, row in gt_s.iterrows():
            g_start = row["start"]
            g_end = row["end"]

            detected = False
            for _, brow in pred_s.iterrows():
                b_idx = int(brow["batch_index"])
                b_start, b_end = batch_windows[b_idx]
                if intervals_overlap(g_start, g_end, b_start, b_end):
                    detected = True
                    break

            if detected:
                tp_recall += 1
            else:
                fn_recall += 1

        # --- Precision ---
        tp_prec = 0
        fp_prec = 0
        for _, brow in pred_s.iterrows():
            b_idx = int(brow["batch_index"])
            b_start, b_end = batch_windows[b_idx]

            hits = False
            for _, row in gt_s.iterrows():
                g_start = row["start"]
                g_end = row["end"]
                if intervals_overlap(g_start, g_end, b_start, b_end):
                    hits = True
                    break

            if hits:
                tp_prec += 1
            else:
                fp_prec += 1

        recall = tp_recall / (tp_recall + fn_recall) if (tp_recall + fn_recall) > 0 else 0.0
        precision = tp_prec / (tp_prec + fp_prec) if (tp_prec + fp_prec) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        records.append(
            {
                "stage": stage,
                "tp_recall_count": tp_recall,
                "fn_recall_count": fn_recall,
                "tp_precision_count": tp_prec,
                "fp_precision_count": fp_prec,
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }
        )

    metrics_df = pd.DataFrame(records)
    return metrics_df


def main() -> None:
    # 1) Load ground-truth for wardbeck from AIT labels
    gt_df = load_ground_truth(LABELS_CSV, SCENARIO_NAME)
    print(f"[INFO] Loaded {len(gt_df)} ground-truth stage intervals for {SCENARIO_NAME}.")

    # 2) Load wardbeck alert timestamps and derive batch time windows
    alert_ts = load_alert_timestamps(WARD_BECK_ALERTS_JSONL)
    print(f"[INFO] Loaded {len(alert_ts)} wardbeck alert timestamps.")

    # For batch_size, r
    batch_df_raw = pd.read_csv(WARD_BECK_BATCH_CSV)
    if batch_df_raw.empty:
        raise RuntimeError("Batch CSV is empty.")
    batch_size = int(batch_df_raw["batch_size"].iloc[0])
    print(f"[INFO] Using batch_size={batch_size} for time windows.")

    batch_windows = derive_batch_time_windows(alert_ts, batch_size)
    print(f"[INFO] Derived {len(batch_windows)} batch time windows.")

    # 3) Load batch predictions (status == ok, parse stage lists)
    batch_df = load_batch_predictions(WARD_BECK_BATCH_CSV)
    print(f"[INFO] Loaded {len(batch_df)} successful batch predictions (status == 'ok').")

    # 4) Compute per-stage metrics
    metrics_df = compute_stage_metrics(gt_df, batch_df, batch_windows)

    # 5) Macro-averages across stages (MAD-LLM-style overall metrics)
    macro_precision = metrics_df["precision"].mean()
    macro_recall = metrics_df["recall"].mean()
    macro_f1 = metrics_df["f1"].mean()

    print("\nPer-stage metrics for wardbeck:")
    print(metrics_df.to_string(index=False))

    print("\nMacro-averaged metrics over wardbeck stages:")
    print(f"  Macro Precision: {macro_precision:.3f}")
    print(f"  Macro Recall   : {macro_recall:.3f}")
    print(f"  Macro F1       : {macro_f1:.3f}")

    # 6) Save per-stage metrics to CSV
    out_path = Path(OUT_STAGE_METRICS_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved per-stage metrics to {out_path}")


if __name__ == "__main__":
    main()
