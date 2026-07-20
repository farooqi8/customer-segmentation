from pydantic import BaseModel

class CustomerInput(BaseModel):
    Age: int
    Annual_Income: int
    Spending_Score: int
    Purchase_Frequency: int
    Average_Order_Value: float
    Days_Since_Last_Purchase: int
    Online_Visits: int
    Discount_Usage: int
    Returns_Count: int
    Tenure_Years: int
    Total_Spending: float
    Satisfaction_Score: int
    Avg_Spending_Per_Visit: float
    Engagement_Score: float

class PredictionOutput(BaseModel):
    cluster: int
    segment: str