# ============================================
# Task 4: Bucket FICO Scores (Quantization)
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Check columns
print("Columns:", df.columns)

# Assume there is a column like 'FICO_Score' and 'Default'
# Modify column names below if different
fico_col = 'FICO_Score'        # <-- change if your dataset uses a different name
target_col = 'Default'

if fico_col not in df.columns:
    raise ValueError(f"Column '{fico_col}' not found in dataset. Please check the actual column name.")

# Drop missing values
df = df.dropna(subset=[fico_col, target_col])

# --------------------------------------------------
# 2. Train a simple model to get PD (probability of default)
# --------------------------------------------------
X = df[[fico_col]]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

df['Predicted_PD'] = model.predict_proba(X[[fico_col]])[:, 1]

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"Logistic Regression AUC based on FICO Score: {auc:.3f}")

# --------------------------------------------------
# 3. Bucketization Approaches
# --------------------------------------------------

# ---- A. Quantile-based Bucketing (Equal Frequency Buckets)
num_buckets = 5
df['FICO_Bucket_Quantile'] = pd.qcut(df[fico_col], q=num_buckets, labels=False, duplicates='drop')

# ---- B. Mean Squared Error (Equal Width Buckets as a proxy)
# This minimizes within-bucket variance roughly similar to MSE approach
df['FICO_Bucket_MSE'] = pd.cut(df[fico_col], bins=num_buckets, labels=False, duplicates='drop')

# --------------------------------------------------
# 4. Map Buckets to Ratings (Lower Rating = Better Credit)
# --------------------------------------------------

def bucket_to_rating(bucket_series):
    """Convert bucket index to rating where 1 = best, num_buckets = worst"""
    return num_buckets - bucket_series

df['Rating_Quantile'] = bucket_to_rating(df['FICO_Bucket_Quantile'])
df['Rating_MSE'] = bucket_to_rating(df['FICO_Bucket_MSE'])

# --------------------------------------------------
# 5. Compute average PD per bucket (for interpretability)
# --------------------------------------------------
rating_summary = df.groupby('Rating_Quantile').agg(
    avg_fico=(fico_col, 'mean'),
    avg_pd=('Predicted_PD', 'mean'),
    count=('Predicted_PD', 'count')
).reset_index()

print("\n--- Rating Summary (Quantile Buckets) ---")
print(rating_summary)

# --------------------------------------------------
# 6. Visualization
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(rating_summary['avg_fico'], rating_summary['avg_pd'], marker='o')
plt.title("Probability of Default vs Average FICO (Quantile Buckets)")
plt.xlabel("Average FICO Score")
plt.ylabel("Average Probability of Default")
plt.grid(True)
plt.show()

# --------------------------------------------------
# 7. Function for future data mapping
# --------------------------------------------------
def map_fico_to_rating(fico_score, fico_series, num_buckets=5):
    """
    Maps a single FICO score to a rating (1 = best, num_buckets = worst)
    based on quantile-based bucketing from training data.
    """
    # Fit quantile bins from the training data
    bins = pd.qcut(fico_series, q=num_buckets, retbins=True, duplicates='drop')[1]
    bucket = np.digitize(fico_score, bins) - 1
    rating = num_buckets - bucket
    return int(np.clip(rating, 1, num_buckets))

# Example usage:
example_fico = 720
predicted_rating = map_fico_to_rating(example_fico, df[fico_col])
print(f"\nExample FICO Score: {example_fico} → Assigned Rating: {predicted_rating}")
