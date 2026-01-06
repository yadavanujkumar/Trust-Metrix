"""
Concept Drift Monitoring Module - Track model performance metrics over time.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Optional
from datetime import datetime
import json


class ConceptDriftMonitor:
    """Monitor model performance and detect concept drift."""
    
    def __init__(self, baseline_metrics: Dict, thresholds: Optional[Dict] = None):
        """
        Initialize concept drift monitor.
        
        Args:
            baseline_metrics: Baseline performance metrics from training
            thresholds: Performance drop thresholds (default: 5% for all metrics)
        """
        self.baseline_metrics = baseline_metrics
        self.thresholds = thresholds or {
            'accuracy': 0.05,  # 5% drop
            'precision': 0.05,
            'recall': 0.05,
            'f1': 0.05
        }
        self.performance_history = []
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         timestamp: Optional[datetime] = None) -> Dict:
        """
        Calculate current performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            timestamp: Optional timestamp for the metrics
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'timestamp': timestamp or datetime.now()
        }
        
        return metrics
    
    def detect_performance_drift(self, current_metrics: Dict) -> Dict:
        """
        Detect if model performance has degraded significantly.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            Dictionary with drift detection results
        """
        drift_detected = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            baseline_value = self.baseline_metrics.get(metric, 0)
            current_value = current_metrics.get(metric, 0)
            
            # Calculate drop
            performance_drop = baseline_value - current_value
            drop_percentage = (performance_drop / baseline_value) * 100 if baseline_value > 0 else 0
            
            # Check if drop exceeds threshold
            is_drifted = performance_drop > self.thresholds[metric]
            
            drift_detected[metric] = {
                'baseline': baseline_value,
                'current': current_value,
                'drop': performance_drop,
                'drop_percentage': drop_percentage,
                'is_drifted': is_drifted,
                'threshold': self.thresholds[metric]
            }
        
        # Overall drift status
        any_drift = any([v['is_drifted'] for v in drift_detected.values()])
        
        return {
            'metrics': drift_detected,
            'overall_drift_detected': any_drift,
            'timestamp': current_metrics.get('timestamp', datetime.now())
        }
    
    def add_to_history(self, metrics: Dict, drift_result: Dict):
        """Add metrics and drift results to performance history."""
        entry = {
            'timestamp': str(metrics.get('timestamp', datetime.now())),
            'metrics': metrics,
            'drift_result': drift_result
        }
        self.performance_history.append(entry)
    
    def get_performance_trend(self, window_size: int = 10) -> Dict:
        """
        Get performance trends over the last N evaluations.
        
        Args:
            window_size: Number of recent evaluations to consider
            
        Returns:
            Dictionary with trend information
        """
        if not self.performance_history:
            return {'status': 'No history available'}
        
        recent_history = self.performance_history[-window_size:]
        
        trends = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            values = [h['metrics'][metric] for h in recent_history]
            
            if len(values) >= 2:
                # Calculate trend (positive = improving, negative = degrading)
                trend = values[-1] - values[0]
                avg_value = np.mean(values)
                std_value = np.std(values)
                
                trends[metric] = {
                    'avg': float(avg_value),
                    'std': float(std_value),
                    'trend': float(trend),
                    'direction': 'improving' if trend > 0 else 'degrading' if trend < 0 else 'stable',
                    'recent_values': [float(v) for v in values]
                }
        
        return trends
    
    def monitor(self, y_true: np.ndarray, y_pred: np.ndarray, 
                timestamp: Optional[datetime] = None) -> Dict:
        """
        Complete monitoring workflow: calculate metrics, detect drift, and update history.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            timestamp: Optional timestamp
            
        Returns:
            Complete monitoring results
        """
        # Calculate current metrics
        current_metrics = self.calculate_metrics(y_true, y_pred, timestamp)
        
        # Detect drift
        drift_result = self.detect_performance_drift(current_metrics)
        
        # Add to history
        self.add_to_history(current_metrics, drift_result)
        
        # Get trends
        trends = self.get_performance_trend()
        
        return {
            'current_metrics': current_metrics,
            'drift_detection': drift_result,
            'trends': trends
        }
    
    def generate_alert(self, drift_result: Dict) -> Optional[Dict]:
        """
        Generate an alert if drift is detected.
        
        Args:
            drift_result: Drift detection results
            
        Returns:
            Alert dictionary or None
        """
        if not drift_result['overall_drift_detected']:
            return None
        
        drifted_metrics = [
            metric for metric, info in drift_result['metrics'].items() 
            if info['is_drifted']
        ]
        
        alert = {
            'alert_type': 'CONCEPT_DRIFT',
            'severity': 'HIGH' if len(drifted_metrics) > 2 else 'MEDIUM',
            'timestamp': str(drift_result['timestamp']),
            'message': f"Performance degradation detected in {len(drifted_metrics)} metric(s): {', '.join(drifted_metrics)}",
            'drifted_metrics': drifted_metrics,
            'details': drift_result['metrics']
        }
        
        return alert
