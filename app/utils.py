from sklearn.preprocessing import StandardScaler
import pandas as pd

# You’d save your fitted scaler from training; here we dummy it
scaler = StandardScaler()

def preprocess_expense(df: pd.DataFrame):
    """Convert categorical + numerical features into model input."""
    # Minimal placeholder, adapt from your notebook preprocessing
    features = ["Expense_Amount"]
    return scaler.fit_transform(df[features])
