# 🔍 Trust-Metrix: MLOps Model Reliability & Observability Dashboard

A comprehensive **MLOps Watchdog** system for monitoring deployed machine learning models in production. Trust-Metrix provides real-time monitoring, drift detection, fairness auditing, and explainability for ML models.

## 🎯 Overview

Trust-Metrix acts as a reliability guardian for your ML models, continuously monitoring:
- **Data Drift**: Statistical divergence between training and production data
- **Concept Drift**: Model performance degradation over time
- **Bias & Fairness**: Disparate impact across demographic groups
- **Model Explainability**: SHAP-based predictions explanations

## ✨ Core Features

### 📊 Data Drift Detection
- **KS-test (Kolmogorov-Smirnov)**: Statistical test for distribution changes
- **PSI (Population Stability Index)**: Industry-standard drift metric
- **Real-time Alerts**: Automatic notifications when drift is detected
- **Feature-level Analysis**: Identify which features are drifting

### 📈 Concept Drift Monitoring
- **Performance Tracking**: Monitor accuracy, precision, recall, and F1-score
- **Baseline Comparison**: Compare against training performance
- **Threshold Alerts**: Configurable thresholds (default: 5% degradation)
- **Trend Analysis**: Historical performance visualization

### ⚖️ Bias & Fairness Auditing
- **Disparate Impact Analysis**: 80% rule compliance checking
- **Demographic Parity**: Equal selection rates across groups
- **Equal Opportunity**: Equal TPR across protected groups
- **Group-level Metrics**: Performance breakdown by demographics

### 🔬 Explainability Interface
- **SHAP Integration**: State-of-the-art model explanations
- **Prediction Explanations**: Understand individual predictions
- **Global Feature Importance**: Overall model behavior analysis
- **Alert Explanations**: Why alerts were triggered

## 🏗️ Architecture

```
Trust-Metrix/
├── src/
│   ├── models/
│   │   └── train_model.py          # Model training script
│   ├── monitoring/
│   │   ├── drift_detection.py      # Data drift (KS-test, PSI)
│   │   ├── concept_drift.py        # Performance monitoring
│   │   ├── fairness_audit.py       # Bias & fairness checks
│   │   └── explainability.py       # SHAP-based explanations
├── api/
│   └── main.py                     # FastAPI service
├── dashboard/
│   └── app.py                      # Streamlit dashboard
├── data/                           # Training data
├── models/                         # Trained models
└── requirements.txt                # Dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yadavanujkumar/Trust-Metrix.git
cd Trust-Metrix
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the sample model**
```bash
python src/models/train_model.py
```

This will:
- Generate synthetic credit risk data (10,000 samples)
- Train a Random Forest classifier
- Save the model, baseline statistics, and training data
- Display model performance metrics

### Running the System

#### 1. Start the FastAPI Service
```bash
python api/main.py
```

The API will be available at `http://localhost:8000`

API endpoints:
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Make predictions
- `POST /monitor/drift` - Check data drift
- `POST /monitor/performance` - Check performance drift
- `POST /monitor/fairness` - Run fairness audit
- `POST /explain` - Explain prediction
- `GET /explain/global` - Global feature importance
- `GET /monitor/history` - Get monitoring history

#### 2. Launch the Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📖 Usage Examples

### Using the API

#### Make a Prediction with Explanation
```python
import requests

response = requests.post(
    "http://localhost:8000/predict?include_explanation=true",
    json={
        "Age": 35,
        "Income": 60000,
        "Credit_Score": 650,
        "Loan_Amount": 25000,
        "Employment_Length": 5,
        "Gender": 1
    }
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']}")
print(f"Top contributor: {result['explanation']['top_contributors'][0]}")
```

#### Check Data Drift
```python
import pandas as pd
import requests

# Generate some production data
production_data = pd.DataFrame({
    'Age': [40, 45, 50],
    'Income': [70000, 80000, 90000],
    'Credit_Score': [700, 720, 680],
    'Loan_Amount': [30000, 35000, 28000],
    'Employment_Length': [8, 10, 7],
    'Gender': [1, 0, 1]
})

response = requests.post(
    "http://localhost:8000/monitor/drift",
    json={"data": production_data.to_dict('records')}
)

result = response.json()
if result['alert']:
    print(f"Alert: {result['alert']['message']}")
    print(f"Drifted features: {result['drift_results']['drifted_features']}")
```

#### Run Fairness Audit
```python
response = requests.post(
    "http://localhost:8000/monitor/fairness",
    json={
        "data": production_data.to_dict('records'),
        "labels": [0, 1, 0]
    }
)

result = response.json()
print(f"Overall Fair: {result['audit_results']['overall_fair']}")
```

### Using the Dashboard

1. **Overview Page**: System health and monitoring statistics
2. **Data Drift**: Interactive drift detection with visualization
3. **Performance Monitoring**: Real-time performance metrics
4. **Fairness Audit**: Bias detection across demographic groups
5. **Explainability**: SHAP-based prediction explanations
6. **Testing**: Run comprehensive end-to-end tests

## 🧪 Sample Model

The included sample model is a **Credit Risk Classifier** that predicts loan default risk based on:

| Feature | Description | Range |
|---------|-------------|-------|
| Age | Applicant age | 18-80 years |
| Income | Annual income | $20k-$200k |
| Credit_Score | Credit score | 300-850 |
| Loan_Amount | Requested loan | $1k-$100k |
| Employment_Length | Years employed | 0-40 years |
| Gender | Gender (binary) | 0 or 1 |

**Model Performance** (on test set):
- Accuracy: ~85%
- Precision: ~70%
- Recall: ~65%
- F1-Score: ~67%

## 🔧 Configuration

### Drift Detection Thresholds

Edit `src/monitoring/drift_detection.py`:
```python
DataDriftDetector(
    reference_data,
    threshold_ks=0.05,      # KS-test p-value threshold
    threshold_psi=0.2       # PSI threshold (0.2 = significant drift)
)
```

### Performance Thresholds

Edit `src/monitoring/concept_drift.py`:
```python
ConceptDriftMonitor(
    baseline_metrics,
    thresholds={
        'accuracy': 0.05,    # 5% drop threshold
        'precision': 0.05,
        'recall': 0.05,
        'f1': 0.05
    }
)
```

### Fairness Thresholds

Edit `src/monitoring/fairness_audit.py`:
```python
FairnessAuditor(
    sensitive_features=['Gender', 'Age'],
    fairness_threshold=0.8    # 80% rule for disparate impact
)
```

## 📊 Monitoring Metrics

### Data Drift Metrics
- **KS Statistic**: Measures maximum difference between CDFs
- **P-value**: Statistical significance of drift
- **PSI**: Population Stability Index
  - < 0.1: No significant change
  - 0.1-0.2: Small change
  - ≥ 0.2: Significant drift

### Fairness Metrics
- **Disparate Impact Ratio**: Selection rate ratio between groups
- **Demographic Parity**: Difference in selection rates
- **Equal Opportunity**: Difference in True Positive Rates
- **80% Rule**: Protected group rate ≥ 0.8 × reference group rate

## 🛠️ Tech Stack

- **Python 3.8+**: Core language
- **Scikit-learn**: ML model training
- **FastAPI**: REST API service
- **Streamlit**: Interactive dashboard
- **SHAP**: Model explainability
- **EvidentlyAI**: Data quality monitoring (optional)
- **Pandas/NumPy**: Data manipulation
- **SciPy**: Statistical tests
- **Plotly**: Interactive visualizations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SHAP**: For powerful model interpretability
- **EvidentlyAI**: For inspiration on ML monitoring
- **Fairlearn**: For fairness metrics concepts
- **FastAPI**: For the excellent API framework
- **Streamlit**: For rapid dashboard development

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for trustworthy and reliable ML systems**
