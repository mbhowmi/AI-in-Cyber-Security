#!/usr/bin/env python3
"""
Privacy-preserving preprocessing for the AIT wardbeck scenario.


- Wazuh alerts (wardbeck_wazuh.json) are JSONL, each line like:
    {
      "predecoder": {...},
      "agent": {"ip": "10.132.56.171", "name": "wazuh-client", "id": "25"},
      "manager": {"name": "wazuh.manager"},
      "rule": {...},
      "decoder": {...},
      "full_log": "...",
      "input": {"type": "log"},
      "@timestamp": "2022-01-19T00:38:02.000000Z",
      "location": "/var/log/syslog",
      "id": "1688374126.662"
    }

- AMiner alerts (wardbeck_aminer.json) are JSONL, each line like:
    {
      "AnalysisComponent": {...},
      "LogData": {
        "RawLogData": ["..."],
        "Timestamps": [1642550401],
        "DetectionTimestamp": [1642550401],
        "LogLinesCount": 1,
        "LogResources": ["/var/log/auth.log"]
      },
      "AMiner": {"ID": "10.132.56.204"}
    }

The Objective:
- Preserve attack-relevant fields (time, rule info, messages).
- Pseudonymize or anonymize identifiers (hosts, agent names/IDs, IPs, AMiner.ID).
- Redact emails, file paths, literal IPs in text.
- Output one sanitized JSON alert per line to a JSONL file.
"""

import argparse
import json
import hmac
import hashlib
import ipaddress
import re
from pathlib import Path
from typing import Any, Dict, List

# HMAC key for pseudonymization
HMAC_KEY = b"CHANGE_ME_SECRET_KEY"

# Regex patterns for sensitive substrings in free text.
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
UNIX_PATH_RE = re.compile(r"/[^\s\"]+")           # approx. UNIX paths
WIN_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s\"]+")  # approx. Windows paths
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")  # IPv4 literals


def hmac_pseudo(value: str) -> str:
    """Return a stable HMAC-based pseudonym for any identifier-like string."""
    if not value:
        return value
    digest = hmac.new(HMAC_KEY, value.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"ID_{digest[:16]}"


def anonymize_ip(ip: str) -> str:
    """
    Anonymize an IP-like string while preserving basic structure.
    If parsing fails, return the original string.
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip

    if isinstance(addr, ipaddress.IPv4Address):
        octets = ip.split(".")
        anon_octets: List[str] = []
        for o in octets:
            d = hmac.new(HMAC_KEY, o.encode("utf-8"), hashlib.sha1).hexdigest()
            anon_octets.append(str(int(d[:2], 16) % 256))
        return ".".join(anon_octets)
    else:
        d = hmac.new(HMAC_KEY, ip.encode("utf-8"), hashlib.sha1).hexdigest()
        groups = [d[i : i + 4] for i in range(0, 32, 4)]
        return ":".join(groups)


def redact_text(text: str) -> str:
    """
    Redact sensitive patterns in free-text:
    - email addresses
    - file paths
    - literal IPv4 addresses
    """
    if not isinstance(text, str):
        return text
    text = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = UNIX_PATH_RE.sub("[PATH_REDACTED]", text)
    text = WIN_PATH_RE.sub("[PATH_REDACTED]", text)
    text = IP_RE.sub("[IP_REDACTED]", text)
    return text


def sanitize_wazuh_alert(alert: Dict[str, Any]) -> Dict[str, Any]:
    """
    Kept (sanitized) fields:
    - timestamp:    from @timestamp (ISO string)
    - hostname:     from predecoder.hostname (HMAC)
    - program_name: from predecoder.program_name
    - agent_ip:     from agent.ip (IP anonymized)
    - agent_name:   from agent.name (HMAC)
    - agent_id:     from agent.id (HMAC)
    - manager_name: from manager.name (HMAC)
    - rule_level:       from rule.level
    - rule_description: from rule.description (redacted)
    - rule_id:          from rule.id
    - decoder_name: from decoder.name
    - message:      from full_log (redacted)
    - log_location: from location (redacted)
    - alert_id:     from id (HMAC)
    - source:       literal "wazuh"
    """
    out: Dict[str, Any] = {}

    # Timestamp
    if "@timestamp" in alert:
        out["timestamp"] = alert["@timestamp"]

    # Predecoder host/program
    pre = alert.get("predecoder", {})
    if isinstance(pre, dict):
        if "hostname" in pre:
            out["hostname"] = hmac_pseudo(str(pre["hostname"]))
        if "program_name" in pre:
            out["program_name"] = pre["program_name"]

    # Agent info
    agent = alert.get("agent", {})
    if isinstance(agent, dict):
        if "ip" in agent:
            out["agent_ip"] = anonymize_ip(str(agent["ip"]))
        if "name" in agent:
            out["agent_name"] = hmac_pseudo(str(agent["name"]))
        if "id" in agent:
            out["agent_id"] = hmac_pseudo(str(agent["id"]))

    # Manager name
    manager = alert.get("manager", {})
    if isinstance(manager, dict) and "name" in manager:
        out["manager_name"] = hmac_pseudo(str(manager["name"]))

    # Rule details
    rule = alert.get("rule", {})
    if isinstance(rule, dict):
        if "level" in rule:
            out["rule_level"] = rule["level"]
        if "description" in rule:
            out["rule_description"] = redact_text(str(rule["description"]))
        if "id" in rule:
            out["rule_id"] = rule["id"]

    # Decoder
    decoder = alert.get("decoder", {})
    if isinstance(decoder, dict) and "name" in decoder:
        out["decoder_name"] = decoder["name"]

    # Message and location
    if "full_log" in alert:
        out["message"] = redact_text(str(alert["full_log"]))
    if "location" in alert:
        out["log_location"] = redact_text(str(alert["location"]))

    # Alert ID pseudonymized
    if "id" in alert:
        out["alert_id"] = hmac_pseudo(str(alert["id"]))

    # Label the source
    out["source"] = "wazuh"
    return out


def sanitize_aminer_alert(alert: Dict[str, Any]) -> Dict[str, Any]:
    """
    Kept (sanitized) fields:
    - timestamp:    first from LogData.DetectionTimestamp or LogData.Timestamps
    - message:      first string from LogData.RawLogData (redacted)
    - log_location: first value from LogData.LogResources (redacted)
    - ac_type:      AnalysisComponent.AnalysisComponentType
    - ac_name:      AnalysisComponent.AnalysisComponentName
    - ac_message:   AnalysisComponent.Message (redacted)
    - aminer_ip:    AMiner.ID (anonymized as IP)
    - source:       literal "aminer"
    """
    out: Dict[str, Any] = {}

    logdata = alert.get("LogData", {})
    if isinstance(logdata, dict):
        det_ts = logdata.get("DetectionTimestamp") or logdata.get("Timestamps")
        if isinstance(det_ts, list) and det_ts:
            out["timestamp"] = det_ts[0]

        raw = logdata.get("RawLogData")
        if isinstance(raw, list) and raw:
            out["message"] = redact_text(str(raw[0]))

        resources = logdata.get("LogResources")
        if isinstance(resources, list) and resources:
            out["log_location"] = redact_text(str(resources[0]))

    ac = alert.get("AnalysisComponent", {})
    if isinstance(ac, dict):
        if "AnalysisComponentType" in ac:
            out["ac_type"] = ac["AnalysisComponentType"]
        if "AnalysisComponentName" in ac:
            out["ac_name"] = ac["AnalysisComponentName"]
        if "Message" in ac:
            out["ac_message"] = redact_text(str(ac["Message"]))

    am = alert.get("AMiner", {})
    if isinstance(am, dict) and "ID" in am:
        out["aminer_ip"] = anonymize_ip(str(am["ID"]))

    out["source"] = "aminer"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ait_dir",
        required=True,
        help="Directory containing wardbeck_aminer.json, wardbeck_wazuh.json, labels.csv.",
    )
    parser.add_argument(
        "--scenario",
        default="wardbeck",
        help="Scenario name (currently tested with wardbeck).",
    )
    parser.add_argument(
        "--out",
        default="/project/6000603/mbhowmi/ML_CS/output/wardbeck_privacy.jsonl",
        help="Output JSONL file with privacy-preserved alerts.",
    )
    args = parser.parse_args()

    ait_dir = Path(args.ait_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wazuh_path = ait_dir / f"{args.scenario}_wazuh.json"
    aminer_path = ait_dir / f"{args.scenario}_aminer.json"

    with out_path.open("w", encoding="utf-8") as out_f:
        # Process Wazuh alerts
        if wazuh_path.exists():
            with wazuh_path.open("r", encoding="utf-8") as wf:
                for line in wf:
                    line = line.strip()
                    if not line:
                        continue
                    alert = json.loads(line)
                    sa = sanitize_wazuh_alert(alert)
                    out_f.write(json.dumps(sa) + "\n")

        # Process AMiner alerts
        if aminer_path.exists():
            with aminer_path.open("r", encoding="utf-8") as af:
                for line in af:
                    line = line.strip()
                    if not line:
                        continue
                    alert = json.loads(line)
                    sa = sanitize_aminer_alert(alert)
                    out_f.write(json.dumps(sa) + "\n")


if __name__ == "__main__":
    main()
