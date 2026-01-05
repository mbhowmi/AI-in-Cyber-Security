# Ensemble-Based Deep Learning for Unsupervised Insider Threat Detection
This project develops an online, unsupervised deep learning system employing a DNN and LSTM ensemble to model normal user behavior from system logs. The system generates interpretable anomaly scores based on deviations, achieving AUCs of 0.95 (DNN) and 0.93 (LSTM) on the CERT dataset, outperforming baseline methods. The novelty lies in its adaptable ensemble framework with feature-level anomaly decomposition, not requiring labeled data. The research methodology encompasses feature extraction, structured stream neural networks, probability-based anomaly scoring, online training, anomaly detection, and comparison with baselines. Result analysis highlights the superior performance of deep learning, the detrimental impact of categorical features, and the benefit of diagonal covariance. The study concludes the effectiveness of the proposed system for real-time, adaptable, and interpretable insider threat detection. Future work will focus on enhancing categorical feature handling and testing on diverse datasets, with key lessons emphasizing ensemble models and interpretability.

Dataset:
CERT datasetv6.2 has been used directly from https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247 (r6.2.tar.bz2).

Feature Extraction:
The methodology was inspired from "Analyzing Data Granularity Levels for Insider Threat Detection Using Machine Learning"(https://ieeexplore.ieee.org/document/8962316).

Anomaly Detection and Baseline Model Comparision:
This work is inspired by "Anomaly Detection for Insider Threats Using Unsupervised Ensembles"(https://ieeexplore.ieee.org/document/9399116), with significant enhancements in both detection performance and methodological approach. For a detailed understanding of the original framework, please refer to the pdf.
