# Network Intrusion Detection System

A machine learning-based Network Intrusion Detection System (NIDS) that classifies network traffic using the CICIDS2017 dataset. It features an interactive Streamlit dashboard, model comparison, and SHAP-based explainability.

**Tech Stack:** Python · XGBoost · Scikit-learn · Pandas · SHAP · Streamlit

## Key Features

- **Traffic Classification:** Classifies network traffic into seven categories.
- **Threat Scanner:** Analyzes uploaded CSV, Excel, and JSON files.
- **Manual Prediction:** Predicts the category of individual network flows.
- **Model Comparison:** Evaluates four machine learning classifiers.
- **Explainable AI:** Uses SHAP to explain model predictions.
- **Performance Dashboard:** Displays classification metrics and dataset insights.

## Model Performance

Results recorded on the test split:

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | F1 (Weighted) | Attack FPR |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost** | **99.9296%** | 0.9545 | **0.9912** | **0.9700** | 0.9993 | 0.0008 |
| Decision Tree | 99.9084% | 0.9708 | 0.9668 | 0.9687 | 0.9991 | 0.0005 |
| Random Forest | 99.9045% | 0.9815 | 0.9550 | 0.9672 | 0.9990 | 0.0004 |
| Logistic Regression | 86.3547% | 0.5075 | 0.9591 | 0.5580 | 0.9031 | 0.1589 |

XGBoost achieved the highest accuracy and macro F1-score, while Random Forest recorded the highest macro precision and lowest reported Attack FPR.

*Results reflect the recorded experiment and may differ on real-world network traffic.*

## Installation and Usage

**1. Clone the repository**

```bash
git clone https://github.com/krishpatel1220/network-intrusion-detection-system.git
cd network-intrusion-detection-system
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Prepare the dataset and train the models**

Download the [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html) and place the required CSV files in `data/raw/`.

```bash
python prepare_data.py
python train.py
```

**4. Launch the application**

```bash
streamlit run app.py
```

The required model artifacts must be generated before launching the application if they are not included in the repository.

## Dataset

This project uses the **CICIDS2017 dataset** from the Canadian Institute for Cybersecurity. The implementation uses 52 features to classify network traffic into seven categories:

Benign, DoS, DDoS, Port Scanning, Bot, Brute Force, and Web Attack.

## Limitations

- CICIDS2017 may not represent modern network traffic.
- Class imbalance can affect detection performance for less frequent attacks.
- High benchmark scores do not guarantee equivalent performance on unseen networks.

## Author

**Krish Patel**

[GitHub Profile](https://github.com/krishpatel1220)