"""
Streamlit Dashboard for Model Reliability & Observability.
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# API endpoint
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Trust-Metrix: MLOps Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        margin: 10px 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 10px;
        margin: 10px 0;
    }
    .status-healthy {
        color: #4caf50;
    }
    .status-warning {
        color: #ff9800;
    }
    .status-critical {
        color: #f44336;
    }
    </style>
    """, unsafe_allow_html=True)


def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200 and response.json()
    except:
        return None


def load_sample_data(n_samples=100, drift=False):
    """Load sample data for testing."""
    np.random.seed(42 if not drift else 123)
    
    if drift:
        # Generate drifted data
        age = np.random.randint(40, 81, n_samples)  # Shifted age distribution
        income = np.random.uniform(30000, 250000, n_samples)  # Shifted income
        credit_score = np.random.randint(250, 800, n_samples)  # Shifted credit score
    else:
        # Generate normal data
        age = np.random.randint(18, 81, n_samples)
        income = np.random.uniform(20000, 200000, n_samples)
        credit_score = np.random.randint(300, 851, n_samples)
    
    loan_amount = np.random.uniform(1000, 100000, n_samples)
    employment_length = np.random.randint(0, 41, n_samples)
    gender = np.random.randint(0, 2, n_samples)
    
    # Generate labels based on simple logic
    default_prob = (
        0.1 +
        0.3 * (credit_score < 600) +
        0.2 * (income < 40000)
    )
    default_prob = np.clip(default_prob, 0, 1)
    default = (np.random.random(n_samples) < default_prob).astype(int)
    
    data = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Credit_Score': credit_score,
        'Loan_Amount': loan_amount,
        'Employment_Length': employment_length,
        'Gender': gender,
        'Default': default
    })
    
    return data


def main():
    """Main dashboard function."""
    
    # Title and header
    st.title("🔍 Trust-Metrix: Model Reliability & Observability Dashboard")
    st.markdown("**MLOps Watchdog for Machine Learning Models**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Overview", "📊 Data Drift", "📈 Performance Monitoring", 
         "⚖️ Fairness Audit", "🔬 Explainability", "🧪 Testing"]
    )
    
    # Check API health
    health_status = check_api_health()
    
    if health_status is None:
        st.sidebar.error("⚠️ API is not running")
        st.sidebar.info("Start the API with: `python api/main.py`")
        return
    else:
        if health_status.get('status') == 'healthy':
            st.sidebar.success("✅ API is healthy")
        else:
            st.sidebar.warning("⚠️ API is running but unhealthy")
    
    # Page routing
    if page == "🏠 Overview":
        show_overview(health_status)
    elif page == "📊 Data Drift":
        show_drift_monitoring()
    elif page == "📈 Performance Monitoring":
        show_performance_monitoring()
    elif page == "⚖️ Fairness Audit":
        show_fairness_audit()
    elif page == "🔬 Explainability":
        show_explainability()
    elif page == "🧪 Testing":
        show_testing()


def show_overview(health_status):
    """Show overview page."""
    st.header("System Overview")
    
    # System status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="API Status",
            value="Healthy" if health_status.get('status') == 'healthy' else "Unhealthy",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Model Status",
            value="Loaded" if health_status.get('model_loaded') else "Not Loaded",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Monitors",
            value="Active" if health_status.get('monitors_initialized') else "Inactive",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Last Check",
            value=datetime.now().strftime("%H:%M:%S"),
            delta=None
        )
    
    # Get monitoring history
    try:
        response = requests.get(f"{API_URL}/monitor/history")
        if response.status_code == 200:
            history = response.json()
            
            st.subheader("📊 Monitoring Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", history.get('total_predictions', 0))
            with col2:
                st.metric("Drift Checks", history.get('total_drift_checks', 0))
            with col3:
                st.metric("Performance Checks", history.get('total_performance_checks', 0))
            with col4:
                st.metric("Fairness Checks", history.get('total_fairness_checks', 0))
    except Exception as e:
        st.error(f"Error fetching monitoring history: {str(e)}")
    
    # Feature descriptions
    st.subheader("🎯 Core Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Data Drift Detection**
        - KS-test for statistical divergence
        - PSI (Population Stability Index)
        - Real-time drift alerts
        
        **📈 Concept Drift Monitoring**
        - Performance metrics tracking
        - Accuracy, Precision, Recall monitoring
        - Threshold-based alerts
        """)
    
    with col2:
        st.markdown("""
        **⚖️ Bias & Fairness Auditing**
        - Disparate impact analysis
        - Demographic parity checks
        - Equal opportunity metrics
        
        **🔬 Model Explainability**
        - SHAP-based explanations
        - Feature importance analysis
        - Prediction explanations
        """)


def show_drift_monitoring():
    """Show data drift monitoring page."""
    st.header("📊 Data Drift Detection")
    st.markdown("Monitor statistical divergence between training and production data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.slider("Number of samples", 50, 500, 100)
    with col2:
        introduce_drift = st.checkbox("Introduce artificial drift", value=False)
    
    if st.button("Check for Data Drift", type="primary"):
        with st.spinner("Analyzing data drift..."):
            # Generate sample data
            data = load_sample_data(n_samples, drift=introduce_drift)
            
            # Call API
            try:
                response = requests.post(
                    f"{API_URL}/monitor/drift",
                    json={"data": data.drop('Default', axis=1).to_dict('records')}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    drift_results = result['drift_results']
                    alert = result.get('alert')
                    
                    # Show alert if present
                    if alert:
                        st.markdown(f"""
                        <div class="alert-{alert['severity'].lower()}">
                            <strong>⚠️ {alert['alert_type']}</strong><br>
                            {alert['message']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("✅ No significant drift detected")
                    
                    # Display overall drift score
                    st.metric(
                        "Overall Drift Score",
                        f"{drift_results['overall_drift_score']:.2%}",
                        delta=None
                    )
                    
                    # Feature-level drift results
                    st.subheader("Feature-level Drift Analysis")
                    
                    features_data = []
                    for feature, results in drift_results['features'].items():
                        features_data.append({
                            'Feature': feature,
                            'KS Statistic': results['ks_test']['ks_statistic'],
                            'P-value': results['ks_test']['p_value'],
                            'PSI': results['psi']['psi'],
                            'Status': '🔴 Drifted' if results['overall_drifted'] else '🟢 Stable'
                        })
                    
                    df = pd.DataFrame(features_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Visualization
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df['Feature'],
                        y=df['PSI'],
                        name='PSI',
                        marker_color=['red' if x > 0.2 else 'orange' if x > 0.1 else 'green' 
                                     for x in df['PSI']]
                    ))
                    fig.update_layout(
                        title="PSI Values by Feature",
                        xaxis_title="Feature",
                        yaxis_title="PSI Value",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")


def show_performance_monitoring():
    """Show performance monitoring page."""
    st.header("📈 Concept Drift Monitoring")
    st.markdown("Track model performance metrics and detect degradation")
    
    n_samples = st.slider("Number of samples", 50, 500, 100)
    
    if st.button("Check Model Performance", type="primary"):
        with st.spinner("Evaluating model performance..."):
            # Generate sample data with labels
            data = load_sample_data(n_samples)
            
            try:
                response = requests.post(
                    f"{API_URL}/monitor/performance",
                    json={
                        "data": data.drop('Default', axis=1).to_dict('records'),
                        "labels": data['Default'].tolist()
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    monitoring_result = result['monitoring_result']
                    alert = result.get('alert')
                    
                    # Show alert if present
                    if alert:
                        st.markdown(f"""
                        <div class="alert-{alert['severity'].lower()}">
                            <strong>⚠️ {alert['alert_type']}</strong><br>
                            {alert['message']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("✅ Performance within acceptable range")
                    
                    # Current metrics
                    st.subheader("Current Performance Metrics")
                    current = monitoring_result['current_metrics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{current['accuracy']:.4f}")
                    with col2:
                        st.metric("Precision", f"{current['precision']:.4f}")
                    with col3:
                        st.metric("Recall", f"{current['recall']:.4f}")
                    with col4:
                        st.metric("F1-Score", f"{current['f1']:.4f}")
                    
                    # Drift detection details
                    if 'drift_detection' in monitoring_result:
                        st.subheader("Performance Drift Analysis")
                        
                        drift_data = []
                        for metric, info in monitoring_result['drift_detection']['metrics'].items():
                            drift_data.append({
                                'Metric': metric.capitalize(),
                                'Baseline': f"{info['baseline']:.4f}",
                                'Current': f"{info['current']:.4f}",
                                'Drop': f"{info['drop']:.4f}",
                                'Drop %': f"{info['drop_percentage']:.2f}%",
                                'Status': '🔴 Degraded' if info['is_drifted'] else '🟢 Normal'
                            })
                        
                        df = pd.DataFrame(drift_data)
                        st.dataframe(df, use_container_width=True)
                    
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")


def show_fairness_audit():
    """Show fairness audit page."""
    st.header("⚖️ Bias & Fairness Auditing")
    st.markdown("Check for disparate impact across demographic groups")
    
    n_samples = st.slider("Number of samples", 50, 500, 100)
    
    if st.button("Run Fairness Audit", type="primary"):
        with st.spinner("Auditing model fairness..."):
            # Generate sample data
            data = load_sample_data(n_samples)
            
            try:
                response = requests.post(
                    f"{API_URL}/monitor/fairness",
                    json={
                        "data": data.drop('Default', axis=1).to_dict('records'),
                        "labels": data['Default'].tolist()
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    audit_results = result['audit_results']
                    alert = result.get('alert')
                    
                    # Show alert if present
                    if alert:
                        st.markdown(f"""
                        <div class="alert-{alert['severity'].lower()}">
                            <strong>⚠️ {alert['alert_type']}</strong><br>
                            {alert['message']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("✅ No fairness violations detected")
                    
                    # Overall fairness status
                    st.metric(
                        "Overall Fairness Status",
                        "FAIR" if audit_results['overall_fair'] else "UNFAIR"
                    )
                    
                    # Feature-level fairness analysis
                    for feature, results in audit_results['features'].items():
                        if 'error' in results:
                            continue
                        
                        st.subheader(f"Analysis for {feature}")
                        
                        # Group metrics
                        group_data = []
                        for group, metrics in results['group_metrics'].items():
                            group_data.append({
                                'Group': group,
                                'Sample Size': metrics['sample_size'],
                                'Selection Rate': f"{metrics['selection_rate']:.4f}",
                                'TPR': f"{metrics['tpr']:.4f}",
                                'FPR': f"{metrics['fpr']:.4f}",
                                'Accuracy': f"{metrics['accuracy']:.4f}"
                            })
                        
                        df = pd.DataFrame(group_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Disparate impact
                        di = results['disparate_impact']
                        if 'reference_group' in di:
                            st.markdown(f"**Reference Group:** {di['reference_group']}")
                            
                            di_data = []
                            for group, info in di['groups'].items():
                                if not info.get('is_reference', False):
                                    di_data.append({
                                        'Group': group,
                                        'Disparate Impact Ratio': f"{info['ratio']:.4f}",
                                        '80% Rule': '✅ Pass' if info['passes_80_rule'] else '❌ Fail'
                                    })
                            
                            if di_data:
                                df_di = pd.DataFrame(di_data)
                                st.dataframe(df_di, use_container_width=True)
                        
                        st.divider()
                    
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")


def show_explainability():
    """Show explainability page."""
    st.header("🔬 Model Explainability")
    st.markdown("Understand model predictions using SHAP")
    
    tab1, tab2 = st.tabs(["Single Prediction", "Global Importance"])
    
    with tab1:
        st.subheader("Explain a Single Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", 18, 80, 35)
            income = st.number_input("Income", 20000, 200000, 60000)
            credit_score = st.number_input("Credit Score", 300, 850, 650)
        
        with col2:
            loan_amount = st.number_input("Loan Amount", 1000, 100000, 25000)
            employment_length = st.number_input("Employment Length (years)", 0, 40, 5)
            gender = st.selectbox("Gender", [0, 1])
        
        if st.button("Explain Prediction", type="primary"):
            with st.spinner("Generating explanation..."):
                try:
                    response = requests.post(
                        f"{API_URL}/explain",
                        json={
                            "Age": float(age),
                            "Income": float(income),
                            "Credit_Score": float(credit_score),
                            "Loan_Amount": float(loan_amount),
                            "Employment_Length": float(employment_length),
                            "Gender": int(gender)
                        }
                    )
                    
                    if response.status_code == 200:
                        explanation = response.json()
                        
                        # Prediction
                        st.subheader("Prediction Result")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Prediction", 
                                     "Default" if explanation['prediction'] == 1 else "No Default")
                        with col2:
                            st.metric("Probability (Default)", 
                                     f"{explanation['prediction_probability']['class_1']:.4f}")
                        with col3:
                            st.metric("Probability (No Default)", 
                                     f"{explanation['prediction_probability']['class_0']:.4f}")
                        
                        # Top contributors
                        st.subheader("Top Contributing Features")
                        
                        top_features = explanation['top_contributors']
                        
                        feature_names = [f['feature'] for f in top_features]
                        shap_values = [f['shap_value'] for f in top_features]
                        
                        fig = go.Figure(go.Bar(
                            x=shap_values,
                            y=feature_names,
                            orientation='h',
                            marker_color=['red' if v > 0 else 'blue' for v in shap_values]
                        ))
                        fig.update_layout(
                            title="SHAP Values for Top Features",
                            xaxis_title="SHAP Value (Impact on Prediction)",
                            yaxis_title="Feature",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature details
                        st.subheader("Feature Details")
                        details_data = []
                        for f in top_features:
                            details_data.append({
                                'Feature': f['feature'],
                                'Value': f['value'],
                                'SHAP Value': f'{f["shap_value"]:.4f}',
                                'Impact': '🔴 Positive' if f['impact'] == 'positive' else '🔵 Negative'
                            })
                        
                        df = pd.DataFrame(details_data)
                        st.dataframe(df, use_container_width=True)
                        
                    else:
                        st.error(f"Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("Global Feature Importance")
        
        if st.button("Calculate Global Importance", type="primary"):
            with st.spinner("Calculating feature importance..."):
                try:
                    response = requests.get(f"{API_URL}/explain/global")
                    
                    if response.status_code == 200:
                        importance = response.json()
                        
                        # Create DataFrame
                        imp_data = []
                        for feature, value in importance['feature_importance'].items():
                            imp_data.append({
                                'Feature': feature,
                                'Importance': value
                            })
                        
                        df = pd.DataFrame(imp_data).sort_values('Importance', ascending=False)
                        
                        # Visualization
                        fig = go.Figure(go.Bar(
                            x=df['Importance'],
                            y=df['Feature'],
                            orientation='h',
                            marker_color='lightblue'
                        ))
                        fig.update_layout(
                            title="Global Feature Importance (Mean Absolute SHAP)",
                            xaxis_title="Mean |SHAP Value|",
                            yaxis_title="Feature",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(df, use_container_width=True)
                        
                    else:
                        st.error(f"Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def show_testing():
    """Show testing page."""
    st.header("🧪 Testing & Demo")
    st.markdown("Test the complete monitoring pipeline")
    
    if st.button("Run Complete Test", type="primary"):
        with st.spinner("Running comprehensive tests..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Test 1: Make predictions
            status_text.text("Step 1/4: Making predictions...")
            progress_bar.progress(25)
            
            data = load_sample_data(50)
            predictions = []
            for idx, row in data.head(10).iterrows():
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json=row.drop('Default').to_dict()
                    )
                    if response.status_code == 200:
                        predictions.append(response.json())
                except:
                    pass
            
            st.success(f"✅ Made {len(predictions)} predictions")
            
            # Test 2: Check drift
            status_text.text("Step 2/4: Checking data drift...")
            progress_bar.progress(50)
            
            try:
                response = requests.post(
                    f"{API_URL}/monitor/drift",
                    json={"data": data.drop('Default', axis=1).to_dict('records')}
                )
                if response.status_code == 200:
                    st.success("✅ Drift detection completed")
                    result = response.json()
                    if result.get('alert'):
                        st.warning(f"⚠️ {result['alert']['message']}")
            except Exception as e:
                st.error(f"Drift check failed: {str(e)}")
            
            # Test 3: Check performance
            status_text.text("Step 3/4: Monitoring performance...")
            progress_bar.progress(75)
            
            try:
                response = requests.post(
                    f"{API_URL}/monitor/performance",
                    json={
                        "data": data.drop('Default', axis=1).to_dict('records'),
                        "labels": data['Default'].tolist()
                    }
                )
                if response.status_code == 200:
                    st.success("✅ Performance monitoring completed")
            except Exception as e:
                st.error(f"Performance check failed: {str(e)}")
            
            # Test 4: Check fairness
            status_text.text("Step 4/4: Auditing fairness...")
            progress_bar.progress(100)
            
            try:
                response = requests.post(
                    f"{API_URL}/monitor/fairness",
                    json={
                        "data": data.drop('Default', axis=1).to_dict('records'),
                        "labels": data['Default'].tolist()
                    }
                )
                if response.status_code == 200:
                    st.success("✅ Fairness audit completed")
                    result = response.json()
                    if result.get('alert'):
                        st.warning(f"⚠️ {result['alert']['message']}")
            except Exception as e:
                st.error(f"Fairness check failed: {str(e)}")
            
            status_text.text("All tests completed!")
            st.balloons()


if __name__ == "__main__":
    main()
