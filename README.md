# Customer Segmentation using K-Means Clustering

A machine learning web application that segments customers into actionable groups using K-Means Clustering, served through a FastAPI backend with a custom frontend.

## Live Demo

- Frontend: https://customer-segmentation-farooqi8.netlify.app
- API: https://customer-segmentation-api.onrender.com
- API Docs: https://customer-segmentation-api.onrender.com/docs

## Business Problem

Companies waste marketing budgets by treating all customers the same. This tool groups customers into distinct segments based on their behavior, enabling targeted marketing strategies that increase ROI and reduce churn.

## How It Works

1. Input: Enter customer details (age, income, spending score, purchase frequency, etc.)
2. Processing: The API scales the input using StandardScaler and feeds it to the trained K-Means model
3. Output: Returns the customer segment and recommended marketing action

## Discovered Segments

| Cluster | Segment | Count | Strategy |
|---------|---------|-------|----------|
| 0 | High Value Active | 1,328 | Loyalty rewards and VIP access |
| 1 | Window Shoppers | 1,773 | Targeted conversion offers |
| 2 | Premium Rare Buyers | 63 | Personal concierge service |
| 3 | Casual Low Spenders | 1,836 | Re-engagement campaigns |

## Model Details

| Metric | Value |
|--------|-------|
| Algorithm | K-Means Clustering |
| Optimal K | 4 (Elbow + Silhouette) |
| Silhouette Score | 0.0983 |
| Feature Scaling | StandardScaler |
| Training Data | 5,000 customers, 14 features |

## Tech Stack

- Language: Python
- ML: Scikit-learn, Pandas, NumPy
- API: FastAPI, Pydantic, Uvicorn
- Frontend: HTML, CSS, JavaScript
- Deployment: Render (API), Netlify (Frontend)
- Model Serialization: Joblib

## Feature Engineering

Three derived features were created to improve clustering:

- Avg_Spending_Per_Visit = Total_Spending / Online_Visits
- Spending_Per_Year = Total_Spending / (Tenure_Years + 1)
- Engagement_Score = (Online_Visits + Purchase_Frequency) / 2

## Why K-Means

- Customer segmentation is an unsupervised problem with no predefined labels
- K-Means is efficient, interpretable, and scales well to large datasets
- Business teams can easily understand and act on the resulting segments
- Optimal K was determined using both Elbow Method and Silhouette Score

## Author

Farooqi - GitHub: https://github.com/farooqi8