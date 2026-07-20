import joblib
import numpy as np
import pandas as pd

# Load saved model and scaler
model = joblib.load('models/kmeans_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# New customer data
new_customer = {
    'Age': 35,
    'Annual_Income': 85000,
    'Spending_Score': 72,
    'Purchase_Frequency': 30,
    'Average_Order_Value': 250,
    'Days_Since_Last_Purchase': 15,
    'Online_Visits': 45,
    'Discount_Usage': 5,
    'Returns_Count': 2,
    'Tenure_Years': 4,
    'Total_Spending': 15000,
    'Satisfaction_Score': 8,
    'Avg_Spending_Per_Visit': 333.33,
    'Engagement_Score': 37.5
}

# Convert to DataFrame
new_df = pd.DataFrame([new_customer])

# Scale the data
new_scaled = scaler.transform(new_df)

# Predict cluster
cluster = model.predict(new_scaled)

# Cluster labels
labels = {
    0: 'High Value Active Customers',
    1: 'Window Shoppers',
    2: 'Premium Rare Buyers',
    3: 'Casual Low Spenders'
}

print(f"Cluster Number: {cluster[0]}")
print(f"Segment: {labels[cluster[0]]}")