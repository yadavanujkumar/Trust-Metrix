"""
Explainability Module using SHAP for model interpretability.
"""
import pandas as pd
import numpy as np
import shap
from typing import Dict, Optional, Any
import pickle


class ModelExplainer:
    """Provide model explanations using SHAP."""
    
    def __init__(self, model: Any, feature_names: list):
        """
        Initialize the explainer.
        
        Args:
            model: Trained model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.background_data = None
    
    def initialize_explainer(self, background_data: pd.DataFrame, explainer_type: str = 'tree'):
        """
        Initialize SHAP explainer with background data.
        
        Args:
            background_data: Background dataset for SHAP (typically training data sample)
            explainer_type: Type of explainer ('tree', 'kernel', 'linear')
        """
        self.background_data = background_data[self.feature_names]
        
        if explainer_type == 'tree':
            # Use TreeExplainer for tree-based models (faster)
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == 'kernel':
            # Use KernelExplainer for any model (slower but model-agnostic)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                self.background_data.sample(min(100, len(self.background_data)), random_state=42)
            )
        else:
            # Default to TreeExplainer
            self.explainer = shap.TreeExplainer(self.model)
    
    def explain_prediction(self, instance: pd.DataFrame) -> Dict:
        """
        Explain a single prediction.
        
        Args:
            instance: Single instance to explain (DataFrame with one row)
            
        Returns:
            Dictionary with SHAP values and explanation
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")
        
        # Get prediction
        prediction = self.model.predict(instance[self.feature_names])[0]
        prediction_proba = self.model.predict_proba(instance[self.feature_names])[0]
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(instance[self.feature_names])
        
        # Handle both binary and multi-class cases
        if isinstance(shap_values, list):
            # For binary classification, use class 1 (positive class)
            shap_values_class = shap_values[1]
        else:
            shap_values_class = shap_values
        
        # Get feature contributions
        feature_contributions = {}
        for i, feature in enumerate(self.feature_names):
            feature_contributions[feature] = {
                'value': float(instance[feature].values[0]),
                'shap_value': float(shap_values_class[0][i]),
                'impact': 'positive' if shap_values_class[0][i] > 0 else 'negative'
            }
        
        # Sort by absolute SHAP value
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]['shap_value']),
            reverse=True
        )
        
        # Get top contributing features
        top_features = [
            {
                'feature': feature,
                'value': contrib['value'],
                'shap_value': contrib['shap_value'],
                'impact': contrib['impact']
            }
            for feature, contrib in sorted_features[:5]
        ]
        
        return {
            'prediction': int(prediction),
            'prediction_probability': {
                'class_0': float(prediction_proba[0]),
                'class_1': float(prediction_proba[1])
            },
            'feature_contributions': feature_contributions,
            'top_contributors': top_features,
            'base_value': float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) 
                               else self.explainer.expected_value)
        }
    
    def explain_alert(self, alert_type: str, current_data: pd.DataFrame, 
                     reference_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Explain why an alert was triggered.
        
        Args:
            alert_type: Type of alert ('drift', 'concept_drift', 'fairness')
            current_data: Current data that triggered the alert
            reference_data: Reference data for comparison
            
        Returns:
            Explanation of the alert
        """
        explanation = {
            'alert_type': alert_type,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if alert_type == 'data_drift':
            # Explain which features changed the most
            if reference_data is not None:
                feature_changes = {}
                for feature in self.feature_names:
                    if feature in current_data.columns and feature in reference_data.columns:
                        ref_mean = reference_data[feature].mean()
                        curr_mean = current_data[feature].mean()
                        change = ((curr_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0
                        
                        feature_changes[feature] = {
                            'reference_mean': float(ref_mean),
                            'current_mean': float(curr_mean),
                            'percentage_change': float(change)
                        }
                
                # Sort by absolute change
                sorted_changes = sorted(
                    feature_changes.items(),
                    key=lambda x: abs(x[1]['percentage_change']),
                    reverse=True
                )
                
                explanation['feature_changes'] = dict(sorted_changes[:5])
                explanation['message'] = f"Data drift detected. Top changing feature: {sorted_changes[0][0]} ({sorted_changes[0][1]['percentage_change']:.2f}% change)"
        
        elif alert_type == 'concept_drift':
            # Sample some recent predictions and explain them
            sample_size = min(5, len(current_data))
            sample = current_data.sample(sample_size, random_state=42)
            
            explanations = []
            for idx, row in sample.iterrows():
                instance = row.to_frame().T
                try:
                    pred_explanation = self.explain_prediction(instance)
                    explanations.append({
                        'instance_id': int(idx),
                        'prediction': pred_explanation['prediction'],
                        'top_contributor': pred_explanation['top_contributors'][0]['feature']
                    })
                except Exception as e:
                    continue
            
            explanation['sample_explanations'] = explanations
            explanation['message'] = "Model performance degraded. Sample predictions analyzed for patterns."
        
        elif alert_type == 'fairness':
            explanation['message'] = "Fairness violation detected. Review group-level predictions for bias."
        
        return explanation
    
    def get_global_importance(self, data: pd.DataFrame, sample_size: int = 100) -> Dict:
        """
        Calculate global feature importance using SHAP.
        
        Args:
            data: Data to calculate importance on
            sample_size: Number of samples to use
            
        Returns:
            Dictionary with global feature importance
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")
        
        # Sample data
        sample_data = data[self.feature_names].sample(min(sample_size, len(data)), random_state=42)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(sample_data)
        
        # Handle both binary and multi-class cases
        if isinstance(shap_values, list):
            shap_values_class = shap_values[1]
        else:
            shap_values_class = shap_values
        
        # Calculate mean absolute SHAP value for each feature
        feature_importance = {}
        for i, feature in enumerate(self.feature_names):
            importance = np.mean(np.abs(shap_values_class[:, i]))
            feature_importance[feature] = float(importance)
        
        # Sort by importance
        sorted_importance = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'feature_importance': dict(sorted_importance),
            'top_features': [f[0] for f in sorted_importance[:5]]
        }
