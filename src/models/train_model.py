"""
Train a sample credit risk model for demonstration purposes.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os


def generate_credit_data(n_samples=10000, random_state=42):
    """
    Generate synthetic credit risk data with demographic features.
    
    Features:
    - Age: 18-80
    - Income: 20k-200k
    - Credit_Score: 300-850
    - Loan_Amount: 1k-100k
    - Employment_Length: 0-40 years
    - Gender: Binary (0/1)
    - Default: Target variable (0/1)
    """
    np.random.seed(random_state)
    
    age = np.random.randint(18, 81, n_samples)
    income = np.random.uniform(20000, 200000, n_samples)
    credit_score = np.random.randint(300, 851, n_samples)
    loan_amount = np.random.uniform(1000, 100000, n_samples)
    employment_length = np.random.randint(0, 41, n_samples)
    gender = np.random.randint(0, 2, n_samples)
    
    # Create target with some logical relationships
    default_prob = (
        0.1 +  # Base probability
        0.3 * (credit_score < 600) +  # Low credit score increases risk
        0.2 * (income < 40000) +  # Low income increases risk
        0.1 * (loan_amount / income > 3) +  # High debt-to-income ratio
        0.05 * (employment_length < 2)  # Short employment
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


def train_model(save_dir='../../models'):
    """Train and save the credit risk model."""
    print("Generating training data...")
    data = generate_credit_data(n_samples=10000, random_state=42)
    
    # Save training data for reference
    os.makedirs('../../data', exist_ok=True)
    data.to_csv('../../data/training_data.csv', index=False)
    print(f"Training data saved: {len(data)} samples")
    
    # Split features and target
    X = data.drop('Default', axis=1)
    y = data['Default']
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nModel Performance on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'credit_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")
    
    # Save feature names
    feature_names_path = os.path.join(save_dir, 'feature_names.pkl')
    with open(feature_names_path, 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    # Save baseline statistics for monitoring
    baseline_stats = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'feature_means': X_train.mean().to_dict(),
        'feature_stds': X_train.std().to_dict()
    }
    
    baseline_path = os.path.join(save_dir, 'baseline_stats.pkl')
    with open(baseline_path, 'wb') as f:
        pickle.dump(baseline_stats, f)
    print(f"Baseline statistics saved to: {baseline_path}")
    
    return model, baseline_stats


if __name__ == '__main__':
    train_model()
