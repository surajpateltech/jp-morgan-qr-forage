# ===============================
# Loan Default Prediction & Expected Loss Estimation
# ===============================

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
df = pd.read_csv(r"C:\Users\patel\Downloads\jp-morgan-qr-forage\Task 3 and 4_Loan_Data.csv")

# Quick look
print("Data preview:\n", df.head())
print("\nColumns:\n", df.columns)

# 2. Handle missing values if any
df = df.dropna()

# 3. Identify features (X) and target (y)
# Change 'Default' to your actual column name that indicates previous default (0 or 1)
target_col = 'Default'  # modify if different
X = df.drop(columns=[target_col])
y = df[target_col]

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Standardize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train two models
# Logistic Regression (simple, explainable)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

# Random Forest (more powerful)
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, random_state=42
)
rf_model.fit(X_train, y_train)

# 7. Evaluate models
log_pred_prob = log_model.predict_proba(X_test_scaled)[:, 1]
rf_pred_prob = rf_model.predict_proba(X_test)[:, 1]

print("\nModel Performance:")
print("Logistic Regression AUC:", roc_auc_score(y_test, log_pred_prob))
print("Random Forest AUC:", roc_auc_score(y_test, rf_pred_prob))
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_model.predict(X_test_scaled)))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_model.predict(X_test)))

# 8. Define function to predict expected loss
def predict_expected_loss(model, scaler, borrower_data, recovery_rate=0.1):
    """
    borrower_data: dict or DataFrame row with borrower details (same columns as training data)
    recovery_rate: proportion recovered after default (0.1 = 10%)
    """
    borrower_df = pd.DataFrame([borrower_data])
    borrower_df = pd.get_dummies(borrower_df)
    
    # Align columns with training data
    borrower_df = borrower_df.reindex(columns=X.columns, fill_value=0)
    
    # Scale numeric columns (for logistic model)
    borrower_scaled = scaler.transform(borrower_df)
    
    # Predict Probability of Default (PD)
    pd_value = model.predict_proba(borrower_scaled)[:, 1][0]
    
    # Expected Loss (EL) = PD × (1 - Recovery Rate)
    expected_loss = pd_value * (1 - recovery_rate)
    
    return {
        "Probability_of_Default": round(pd_value, 4),
        "Expected_Loss": round(expected_loss, 4)
    }

# Example borrower
example_borrower = {
    "Income": 50000,
    "Total_Loans": 20000,
    "Loan_Amount": 15000,
    "Age": 30,
    "Credit_Score": 700,
    "Default": 0  # you can exclude if not part of input
}

# 9. Predict for example borrower using Logistic Regression
result = predict_expected_loss(log_model, scaler, example_borrower)
print("\nPredicted for new borrower:")
print(result)
