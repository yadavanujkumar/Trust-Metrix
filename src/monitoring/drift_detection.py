"""
Data Drift Detection Module using KS-test and PSI (Population Stability Index).
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class DataDriftDetector:
    """Detect data drift using KS-test and PSI."""
    
    def __init__(self, reference_data: pd.DataFrame, threshold_ks=0.05, threshold_psi=0.2):
        """
        Initialize drift detector with reference data.
        
        Args:
            reference_data: Training/reference data
            threshold_ks: P-value threshold for KS test (default: 0.05)
            threshold_psi: PSI threshold for drift detection (default: 0.2)
                          PSI < 0.1: No significant change
                          0.1 <= PSI < 0.2: Small change
                          PSI >= 0.2: Significant drift
        """
        self.reference_data = reference_data
        self.threshold_ks = threshold_ks
        self.threshold_psi = threshold_psi
        self.feature_distributions = {}
        
        # Calculate reference distributions
        for col in reference_data.columns:
            if reference_data[col].dtype in ['int64', 'float64']:
                self.feature_distributions[col] = reference_data[col].values
    
    def ks_test(self, current_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Perform Kolmogorov-Smirnov test for each numerical feature.
        
        Args:
            current_data: Current/production data
            
        Returns:
            Dictionary with KS test results for each feature
        """
        results = {}
        
        for feature in self.feature_distributions.keys():
            if feature in current_data.columns:
                ref_values = self.feature_distributions[feature]
                curr_values = current_data[feature].values
                
                # Perform KS test
                ks_statistic, p_value = stats.ks_2samp(ref_values, curr_values)
                
                is_drifted = p_value < self.threshold_ks
                
                results[feature] = {
                    'ks_statistic': float(ks_statistic),
                    'p_value': float(p_value),
                    'is_drifted': is_drifted,
                    'drift_severity': 'High' if ks_statistic > 0.3 else 'Medium' if ks_statistic > 0.15 else 'Low'
                }
        
        return results
    
    def calculate_psi(self, current_data: pd.DataFrame, bins=10) -> Dict[str, Dict]:
        """
        Calculate Population Stability Index (PSI) for each feature.
        
        PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        
        Args:
            current_data: Current/production data
            bins: Number of bins for discretization
            
        Returns:
            Dictionary with PSI values for each feature
        """
        results = {}
        
        for feature in self.feature_distributions.keys():
            if feature in current_data.columns:
                ref_values = self.feature_distributions[feature]
                curr_values = current_data[feature].values
                
                # Create bins based on reference data
                _, bin_edges = np.histogram(ref_values, bins=bins)
                
                # Calculate expected (reference) distribution
                expected_percents = np.histogram(ref_values, bins=bin_edges)[0] / len(ref_values)
                # Add small epsilon to avoid division by zero
                expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
                
                # Calculate actual (current) distribution
                actual_percents = np.histogram(curr_values, bins=bin_edges)[0] / len(curr_values)
                actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
                
                # Calculate PSI
                psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
                
                # Determine drift status
                if psi < 0.1:
                    drift_status = 'No Drift'
                elif psi < 0.2:
                    drift_status = 'Small Drift'
                else:
                    drift_status = 'Significant Drift'
                
                results[feature] = {
                    'psi': float(psi),
                    'is_drifted': psi >= self.threshold_psi,
                    'drift_status': drift_status
                }
        
        return results
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Perform comprehensive drift detection using both KS-test and PSI.
        
        Args:
            current_data: Current/production data
            
        Returns:
            Dictionary with drift detection results
        """
        ks_results = self.ks_test(current_data)
        psi_results = self.calculate_psi(current_data)
        
        # Combine results
        combined_results = {}
        for feature in ks_results.keys():
            combined_results[feature] = {
                'ks_test': ks_results[feature],
                'psi': psi_results[feature],
                'overall_drifted': ks_results[feature]['is_drifted'] or psi_results[feature]['is_drifted']
            }
        
        # Calculate overall drift score
        drift_score = sum([1 for f in combined_results.values() if f['overall_drifted']]) / len(combined_results)
        
        return {
            'features': combined_results,
            'overall_drift_score': drift_score,
            'drifted_features': [f for f, r in combined_results.items() if r['overall_drifted']]
        }


def generate_drifted_data(reference_data: pd.DataFrame, n_samples: int, drift_features: List[str], 
                         drift_magnitude: float = 0.3) -> pd.DataFrame:
    """
    Generate drifted data for testing purposes.
    
    Args:
        reference_data: Original reference data
        n_samples: Number of samples to generate
        drift_features: Features to introduce drift in
        drift_magnitude: Magnitude of drift (0-1)
        
    Returns:
        Drifted data
    """
    drifted_data = reference_data.sample(n=n_samples, replace=True, random_state=np.random.randint(1000))
    
    for feature in drift_features:
        if feature in drifted_data.columns:
            mean = drifted_data[feature].mean()
            std = drifted_data[feature].std()
            # Shift the distribution
            drifted_data[feature] = drifted_data[feature] + (mean * drift_magnitude)
    
    return drifted_data
