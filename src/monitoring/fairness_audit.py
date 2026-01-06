"""
Bias & Fairness Auditing Module - Check for disparate impact across demographic groups.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix


class FairnessAuditor:
    """Audit model predictions for bias and fairness across demographic groups."""
    
    def __init__(self, sensitive_features: List[str], fairness_threshold: float = 0.8):
        """
        Initialize fairness auditor.
        
        Args:
            sensitive_features: List of demographic features to audit (e.g., ['Gender', 'Age'])
                              Note: For continuous features like Age, the data should be 
                              pre-binned into categories (e.g., age groups) for meaningful analysis.
            fairness_threshold: Disparate impact threshold (default: 0.8, the "80% rule")
        """
        self.sensitive_features = sensitive_features
        self.fairness_threshold = fairness_threshold
    
    def calculate_group_metrics(self, data: pd.DataFrame, y_true: np.ndarray, 
                               y_pred: np.ndarray, sensitive_feature: str) -> Dict:
        """
        Calculate performance metrics for each group within a sensitive feature.
        
        Args:
            data: Input data with demographic features
            y_true: True labels
            y_pred: Predicted labels
            sensitive_feature: The demographic feature to analyze
            
        Returns:
            Dictionary with metrics for each group
        """
        results = {}
        
        # Get unique groups
        groups = data[sensitive_feature].unique()
        
        for group in groups:
            # Get indices for this group
            group_mask = data[sensitive_feature] == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            if len(group_y_true) == 0:
                continue
            
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred, labels=[0, 1]).ravel()
            
            # Positive prediction rate (selection rate)
            selection_rate = (tp + fp) / len(group_y_true) if len(group_y_true) > 0 else 0
            
            # True positive rate (TPR) - also called sensitivity or recall
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # False positive rate (FPR)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Accuracy
            accuracy = (tp + tn) / len(group_y_true)
            
            results[str(group)] = {
                'sample_size': int(len(group_y_true)),
                'selection_rate': float(selection_rate),
                'tpr': float(tpr),
                'fpr': float(fpr),
                'accuracy': float(accuracy),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            }
        
        return results
    
    def calculate_disparate_impact(self, group_metrics: Dict) -> Dict:
        """
        Calculate disparate impact ratio between groups.
        
        Disparate Impact Ratio = (Selection Rate of Protected Group) / (Selection Rate of Reference Group)
        A ratio < 0.8 suggests potential discrimination (80% rule)
        
        Args:
            group_metrics: Metrics for each group
            
        Returns:
            Dictionary with disparate impact analysis
        """
        if len(group_metrics) < 2:
            return {'status': 'Insufficient groups for comparison'}
        
        # Find group with highest selection rate (reference group)
        selection_rates = {group: metrics['selection_rate'] 
                          for group, metrics in group_metrics.items()}
        
        reference_group = max(selection_rates, key=selection_rates.get)
        reference_rate = selection_rates[reference_group]
        
        disparate_impact = {}
        
        for group, metrics in group_metrics.items():
            if group == reference_group:
                disparate_impact[group] = {
                    'ratio': 1.0,
                    'is_reference': True,
                    'passes_80_rule': True
                }
            else:
                ratio = metrics['selection_rate'] / reference_rate if reference_rate > 0 else 0
                passes = ratio >= self.fairness_threshold
                
                disparate_impact[group] = {
                    'ratio': float(ratio),
                    'is_reference': False,
                    'passes_80_rule': passes,
                    'selection_rate': metrics['selection_rate'],
                    'reference_rate': reference_rate
                }
        
        return {
            'reference_group': reference_group,
            'groups': disparate_impact,
            'overall_fair': all([g['passes_80_rule'] for g in disparate_impact.values()])
        }
    
    def calculate_demographic_parity(self, group_metrics: Dict) -> Dict:
        """
        Calculate demographic parity difference.
        
        Demographic parity is achieved when selection rates are equal across groups.
        
        Args:
            group_metrics: Metrics for each group
            
        Returns:
            Dictionary with demographic parity analysis
        """
        selection_rates = [metrics['selection_rate'] for metrics in group_metrics.values()]
        
        if len(selection_rates) < 2:
            return {'status': 'Insufficient data'}
        
        max_rate = max(selection_rates)
        min_rate = min(selection_rates)
        
        parity_difference = max_rate - min_rate
        
        # Typically, a difference < 0.1 (10%) is considered acceptable
        is_fair = parity_difference < 0.1
        
        return {
            'max_selection_rate': float(max_rate),
            'min_selection_rate': float(min_rate),
            'parity_difference': float(parity_difference),
            'is_fair': is_fair
        }
    
    def calculate_equal_opportunity(self, group_metrics: Dict) -> Dict:
        """
        Calculate equal opportunity difference (difference in TPR across groups).
        
        Equal opportunity is achieved when TPR is equal across groups.
        
        Args:
            group_metrics: Metrics for each group
            
        Returns:
            Dictionary with equal opportunity analysis
        """
        tpr_values = {group: metrics['tpr'] for group, metrics in group_metrics.items()}
        
        if len(tpr_values) < 2:
            return {'status': 'Insufficient data'}
        
        max_tpr = max(tpr_values.values())
        min_tpr = min(tpr_values.values())
        
        opportunity_difference = max_tpr - min_tpr
        
        # Typically, a difference < 0.1 (10%) is considered acceptable
        is_fair = opportunity_difference < 0.1
        
        return {
            'max_tpr': float(max_tpr),
            'min_tpr': float(min_tpr),
            'opportunity_difference': float(opportunity_difference),
            'is_fair': is_fair,
            'group_tpr': {k: float(v) for k, v in tpr_values.items()}
        }
    
    def audit(self, data: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Perform comprehensive fairness audit across all sensitive features.
        
        Args:
            data: Input data with demographic features
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Complete fairness audit report
        """
        audit_results = {}
        
        for feature in self.sensitive_features:
            if feature not in data.columns:
                audit_results[feature] = {'error': 'Feature not found in data'}
                continue
            
            # Calculate metrics for each group
            group_metrics = self.calculate_group_metrics(data, y_true, y_pred, feature)
            
            # Calculate fairness metrics
            disparate_impact = self.calculate_disparate_impact(group_metrics)
            demographic_parity = self.calculate_demographic_parity(group_metrics)
            equal_opportunity = self.calculate_equal_opportunity(group_metrics)
            
            audit_results[feature] = {
                'group_metrics': group_metrics,
                'disparate_impact': disparate_impact,
                'demographic_parity': demographic_parity,
                'equal_opportunity': equal_opportunity
            }
        
        # Overall fairness assessment
        overall_fair = all([
            result.get('disparate_impact', {}).get('overall_fair', False)
            for result in audit_results.values()
            if 'error' not in result
        ])
        
        return {
            'features': audit_results,
            'overall_fair': overall_fair,
            'fairness_threshold': self.fairness_threshold
        }
    
    def generate_fairness_alert(self, audit_results: Dict) -> Optional[Dict]:
        """
        Generate an alert if fairness issues are detected.
        
        Args:
            audit_results: Results from fairness audit
            
        Returns:
            Alert dictionary or None
        """
        if audit_results['overall_fair']:
            return None
        
        unfair_features = []
        for feature, results in audit_results['features'].items():
            if 'error' in results:
                continue
            
            if not results.get('disparate_impact', {}).get('overall_fair', True):
                unfair_features.append(feature)
        
        if not unfair_features:
            return None
        
        alert = {
            'alert_type': 'FAIRNESS_VIOLATION',
            'severity': 'HIGH',
            'message': f"Fairness issues detected in {len(unfair_features)} feature(s): {', '.join(unfair_features)}",
            'unfair_features': unfair_features,
            'details': {feature: audit_results['features'][feature] 
                       for feature in unfair_features}
        }
        
        return alert
