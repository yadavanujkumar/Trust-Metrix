"""
FastAPI service for model serving and monitoring.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Optional, Dict
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.monitoring.drift_detection import DataDriftDetector
from src.monitoring.concept_drift import ConceptDriftMonitor
from src.monitoring.fairness_audit import FairnessAuditor
from src.monitoring.explainability import ModelExplainer

app = FastAPI(title="Trust-Metrix MLOps Monitoring API", version="1.0.0")

# Global variables for model and monitors
model = None
feature_names = None
baseline_stats = None
training_data = None
drift_detector = None
concept_monitor = None
fairness_auditor = None
explainer = None

# Storage for monitoring history
monitoring_history = {
    'predictions': [],
    'drift_checks': [],
    'performance_checks': [],
    'fairness_checks': []
}


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    Age: float
    Income: float
    Credit_Score: float
    Loan_Amount: float
    Employment_Length: float
    Gender: int


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int
    probability: Dict[str, float]
    explanation: Optional[Dict] = None


class MonitoringRequest(BaseModel):
    """Request model for monitoring."""
    data: List[Dict]
    labels: Optional[List[int]] = None


@app.on_event("startup")
async def load_model():
    """Load model and initialize monitors on startup."""
    global model, feature_names, baseline_stats, training_data
    global drift_detector, concept_monitor, fairness_auditor, explainer
    
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    
    # Load model
    model_path = os.path.join(models_dir, 'credit_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
    else:
        print("Warning: Model not found. Please train the model first.")
    
    # Load feature names
    feature_names_path = os.path.join(models_dir, 'feature_names.pkl')
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)
    
    # Load baseline statistics
    baseline_path = os.path.join(models_dir, 'baseline_stats.pkl')
    if os.path.exists(baseline_path):
        with open(baseline_path, 'rb') as f:
            baseline_stats = pickle.load(f)
    
    # Load training data
    training_data_path = os.path.join(data_dir, 'training_data.csv')
    if os.path.exists(training_data_path):
        training_data = pd.read_csv(training_data_path)
        
        # Initialize monitors
        if feature_names:
            drift_detector = DataDriftDetector(training_data[feature_names])
        
        if baseline_stats:
            concept_monitor = ConceptDriftMonitor(baseline_stats)
        
        fairness_auditor = FairnessAuditor(
            sensitive_features=['Gender', 'Age'],
            fairness_threshold=0.8
        )
        
        if model and feature_names:
            explainer = ModelExplainer(model, feature_names)
            explainer.initialize_explainer(training_data)
        
        print("Monitors initialized successfully")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Trust-Metrix MLOps Monitoring API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "monitors_initialized": all([
            drift_detector is not None,
            concept_monitor is not None,
            fairness_auditor is not None,
            explainer is not None
        ])
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, include_explanation: bool = False):
    """Make a prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert request to DataFrame
    data = pd.DataFrame([request.dict()])
    
    # Make prediction
    prediction = model.predict(data[feature_names])[0]
    proba = model.predict_proba(data[feature_names])[0]
    
    response = {
        "prediction": int(prediction),
        "probability": {
            "class_0": float(proba[0]),
            "class_1": float(proba[1])
        }
    }
    
    # Add explanation if requested
    if include_explanation and explainer:
        explanation = explainer.explain_prediction(data)
        response["explanation"] = explanation
    
    # Store prediction
    monitoring_history['predictions'].append({
        'timestamp': datetime.now().isoformat(),
        'data': request.dict(),
        'prediction': int(prediction),
        'probability': float(proba[1])
    })
    
    return response


@app.post("/monitor/drift")
async def check_drift(request: MonitoringRequest):
    """Check for data drift."""
    if drift_detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    
    # Convert request data to DataFrame
    current_data = pd.DataFrame(request.data)
    
    # Check drift
    drift_results = drift_detector.detect_drift(current_data[feature_names])
    
    # Store results
    monitoring_history['drift_checks'].append({
        'timestamp': datetime.now().isoformat(),
        'results': drift_results
    })
    
    # Generate alert if drift detected
    alert = None
    if drift_results['drifted_features']:
        alert = {
            'alert_type': 'DATA_DRIFT',
            'severity': 'HIGH' if drift_results['overall_drift_score'] > 0.5 else 'MEDIUM',
            'message': f"Data drift detected in {len(drift_results['drifted_features'])} features",
            'drifted_features': drift_results['drifted_features']
        }
    
    return {
        'drift_results': drift_results,
        'alert': alert,
        'timestamp': datetime.now().isoformat()
    }


@app.post("/monitor/performance")
async def check_performance(request: MonitoringRequest):
    """Check for concept drift (performance degradation)."""
    if concept_monitor is None:
        raise HTTPException(status_code=503, detail="Concept monitor not initialized")
    
    if request.labels is None:
        raise HTTPException(status_code=400, detail="Labels required for performance monitoring")
    
    # Convert request data to DataFrame
    current_data = pd.DataFrame(request.data)
    
    # Make predictions
    y_pred = model.predict(current_data[feature_names])
    y_true = np.array(request.labels)
    
    # Monitor performance
    monitoring_result = concept_monitor.monitor(y_true, y_pred)
    
    # Store results
    monitoring_history['performance_checks'].append({
        'timestamp': datetime.now().isoformat(),
        'results': monitoring_result
    })
    
    # Generate alert if drift detected
    alert = concept_monitor.generate_alert(monitoring_result['drift_detection'])
    
    return {
        'monitoring_result': monitoring_result,
        'alert': alert,
        'timestamp': datetime.now().isoformat()
    }


@app.post("/monitor/fairness")
async def check_fairness(request: MonitoringRequest):
    """Check for bias and fairness issues."""
    if fairness_auditor is None:
        raise HTTPException(status_code=503, detail="Fairness auditor not initialized")
    
    if request.labels is None:
        raise HTTPException(status_code=400, detail="Labels required for fairness auditing")
    
    # Convert request data to DataFrame
    current_data = pd.DataFrame(request.data)
    
    # Make predictions
    y_pred = model.predict(current_data[feature_names])
    y_true = np.array(request.labels)
    
    # Audit fairness
    audit_results = fairness_auditor.audit(current_data, y_true, y_pred)
    
    # Store results
    monitoring_history['fairness_checks'].append({
        'timestamp': datetime.now().isoformat(),
        'results': audit_results
    })
    
    # Generate alert if fairness issues detected
    alert = fairness_auditor.generate_fairness_alert(audit_results)
    
    return {
        'audit_results': audit_results,
        'alert': alert,
        'timestamp': datetime.now().isoformat()
    }


@app.get("/monitor/history")
async def get_monitoring_history():
    """Get monitoring history."""
    return {
        'total_predictions': len(monitoring_history['predictions']),
        'total_drift_checks': len(monitoring_history['drift_checks']),
        'total_performance_checks': len(monitoring_history['performance_checks']),
        'total_fairness_checks': len(monitoring_history['fairness_checks']),
        'recent_predictions': monitoring_history['predictions'][-10:],
        'recent_drift_checks': monitoring_history['drift_checks'][-5:],
        'recent_performance_checks': monitoring_history['performance_checks'][-5:],
        'recent_fairness_checks': monitoring_history['fairness_checks'][-5:]
    }


@app.post("/explain")
async def explain_prediction(request: PredictionRequest):
    """Explain a specific prediction."""
    if explainer is None:
        raise HTTPException(status_code=503, detail="Explainer not initialized")
    
    # Convert request to DataFrame
    data = pd.DataFrame([request.dict()])
    
    # Get explanation
    explanation = explainer.explain_prediction(data)
    
    return explanation


@app.get("/explain/global")
async def global_importance():
    """Get global feature importance."""
    if explainer is None or training_data is None:
        raise HTTPException(status_code=503, detail="Explainer not initialized")
    
    importance = explainer.get_global_importance(training_data)
    
    return importance


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
