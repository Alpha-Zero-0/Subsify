import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Generate client data
np.random.seed(42)
n_clients = 1000

df = pd.DataFrame({
    'Client_ID': [f'C{i:04}' for i in range(n_clients)],
    'Service_Usage_Count': np.random.randint(1, 50, size=n_clients),  # Times used the main service
    'Avg_Interval': np.random.randint(1, 120, size=n_clients),
    'Total_Spend': np.round(np.random.uniform(50, 8000, size=n_clients), 2),
    'Account_Age_Days': np.random.randint(30, 1095, size=n_clients),  # 1 month to 3 years
    'Help_Desk_Tickets': np.random.randint(0, 15, size=n_clients),  # Times contacted support
    'Last_Login_Days_Ago': np.random.randint(0, 90, size=n_clients),
    'Feature_Usage_Count': np.random.randint(1, 20, size=n_clients),  # Different features used
    'Peak_Usage_Hour': np.random.randint(0, 24, size=n_clients),  # Time of day preference
    'Mobile_Usage_Ratio': np.round(np.random.uniform(0, 1, size=n_clients), 2),  # Mobile vs desktop
})

# 2. Create subscription logic
subscription_score = (
    (df['Service_Usage_Count'] / 50) * 0.25 +
    (1 - df['Avg_Interval'] / 120) * 0.2 +
    (df['Total_Spend'] / 8000) * 0.2 +
    (df['Account_Age_Days'] / 1095) * 0.1 +
    (1 - df['Last_Login_Days_Ago'] / 90) * 0.15 +
    (df['Feature_Usage_Count'] / 20) * 0.1
)

# Add some randomness to make it more realistic
noise = np.random.normal(0, 0.1, size=n_clients)
subscription_score += noise

df['Chose_Subscription'] = (subscription_score > 0.6).astype(int)

print(f"Subscription rate: {df['Chose_Subscription'].mean():.2%}")

# 3. Prepare features
feature_columns = ['Service_Usage_Count', 'Avg_Interval', 'Total_Spend', 'Account_Age_Days', 
                  'Help_Desk_Tickets', 'Last_Login_Days_Ago', 'Feature_Usage_Count', 
                  'Peak_Usage_Hour', 'Mobile_Usage_Ratio']

X = df[feature_columns]
y = df['Chose_Subscription']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scale features (important for Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Neural Network model
nn_model = MLPClassifier(
    hidden_layer_sizes=(600, 300),  # Optimized architecture
    activation='relu',
    solver='adam',
    alpha=0.0005,  # Regularization to prevent overfitting
    learning_rate_init=0.001,
    max_iter=10000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=200,
    tol=1e-6,
    verbose=False
)

nn_model.fit(X_train_scaled, y_train)

# 7. Evaluate performance
y_pred = nn_model.predict(X_test_scaled)
y_pred_proba = nn_model.predict_proba(X_test_scaled)

print("\n=== NEURAL NETWORK MODEL RESULTS ===")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.1%}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba[:, 1]):.3f}")

# Custom readable classification report
from sklearn.metrics import confusion_matrix

# Calculate metrics manually for better explanation
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
total_customers = len(y_test)
actual_subscribers = sum(y_test)
actual_non_subscribers = total_customers - actual_subscribers

print(f"\n📊 DETAILED BREAKDOWN:")
print(f"Total test customers: {total_customers}")
print(f"Actual subscribers: {actual_subscribers}")
print(f"Actual non-subscribers: {actual_non_subscribers}")

print(f"\n📋 CONFUSION MATRIX:")
print("                    PREDICTED")
print("                 Subscribe | Don't Subscribe")
print("ACTUAL Subscribe      {:3d}   |      {:3d}".format(tp, fn))
print("   Don't Subscribe    {:3d}   |      {:3d}".format(fp, tn))

# 8. Feature importance using permutation
def calculate_feature_importance(model, X_test, y_test):
    baseline_score = model.score(X_test, y_test)
    importances = []
    
    for i in range(X_test.shape[1]):
        X_permuted = X_test.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        permuted_score = model.score(X_permuted, y_test)
        importance = baseline_score - permuted_score
        importances.append(importance)
    
    return np.array(importances)

feature_importance = calculate_feature_importance(nn_model, X_test_scaled, y_test)
feature_rankings = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nFeature Importance Rankings:")
print(feature_rankings.to_string(index=False))

# 9. Save model and scaler
joblib.dump(nn_model, "neural_network_subscription_model.joblib")
joblib.dump(scaler, "neural_network_scaler.joblib")
print(f"\nModel saved as 'neural_network_subscription_model.joblib'")
print(f"Scaler saved as 'neural_network_scaler.joblib'")

# 10. Example prediction function
def predict_subscription_nn(service_usage_count, avg_interval, total_spend, account_age_days, 
                           help_desk_tickets, last_login_days_ago, feature_usage_count, 
                           peak_usage_hour, mobile_usage_ratio):
    """Predict subscription probability using neural network"""
    new_data = np.array([[service_usage_count, avg_interval, total_spend, account_age_days,
                         help_desk_tickets, last_login_days_ago, feature_usage_count,
                         peak_usage_hour, mobile_usage_ratio]])
    new_data_scaled = scaler.transform(new_data)
    prediction = nn_model.predict(new_data_scaled)[0]
    probability = nn_model.predict_proba(new_data_scaled)[0]
    
    return {
        'subscription_probability': f"{probability[1]:.2%}"
    }

# Example usage
example_result = predict_subscription_nn(25, 15, 2500, 365, 3, 5, 12, 14, 0.7)
print(f"\nExample prediction: {example_result}")
