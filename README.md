# Privacy-Preserving Alert-Based Multi-Stage Attack Detection via LLM
This project is driven by a simple research question: is it possible to protect people’s data in SOC alerts without breaking the patterns that LLMs need to detect attacks. In practice, this means asking whether fields such as usernames, IP addresses, and hostnames can be transformed or removed so that the model only sees an abstract, random-looking scenario, while the underlying behaviour of the attack is still visible in the sequence of events. A second question is whether this privacy layer can be integrated into enterprise SOC workflows. For example, as part of an LLM-powered dashboard, to reduce insider threat risk from both SOC analysts and regular employees by ensuring that no one, including the model, directly sees raw PII. The motivation of this study is to design and evaluate a privacy-preserving pipeline that hides sensitive information in security alerts while keeping the attack detection signal intact. Here the idea is to add a lightweight transformation step before alerts reach the model, rather than changing the model itself, so that deployment in industry stays straightforward and does not require heavy infrastructure changes. The expected outcome is that this approach will show only minimal loss in detection accuracy and multi-stage attack reconstruction, while significantly reducing privacy risk and insider threat exposure in real-world SOC environments.


Dataset: https://zenodo.org/records/8263181/files/ait_ads.zip?download=1

privacy_preserving.py file preprocesses the AIT wardbeck scenario.

privacy_strict.py removes obvious noise and shorten prompts from the first output file

implementation.py file workson Batch-level attack detection using local LLaMA (llama.cpp)

For each chronological batch of wardbeck alerts, LLaMA returns a single line:

  ATTACK: yes/no; TITLE: short title; STAGES: stage1, stage2, ...

Stages are from the wardbeck/MAD-LLM set:

  [network_scans, service_scans, wpscan, dirb, webshell,
   cracking, reverse_shell, privilege_escalation, service_stop, dnsteal]

evaluation.py does MAD-LLM-style evaluation for wardbeck using batch-level LLM outputs  See below for the original MAD-LLM technique:

Dan Du, Xingmao Guan, Yuling Liu, Bo Jiang, Song Liu, Huamin Feng, and Junrong Liu. Mad-llm:
A novel approach for alert-based multi-stage attack detection via llm. In 2024 IEEE International
Symposium on Parallel and Distributed Processing with Applications (ISPA), pages 2046–2053.
IEEE, 2024.
