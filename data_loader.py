import pandas as pd
import numpy as np

def load_data():
    # Creating 3 years of synthetic retail data
    dates = pd.date_range(start='2023-01-01', periods=1095, freq='D')
    df = pd.DataFrame({'date': dates})
    
    # Feature engineering: weekend and holiday flags
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_holiday'] = 0 # Placeholder for your holiday logic
    
    # Synthetic target: sales (with noise and weekly seasonality)
    df['sales'] = 100 + (df['day_of_week'] * 10) + (np.random.normal(0, 5, 1095))
    
    # Split features and target
    X = df[['is_weekend', 'is_holiday', 'day_of_week']]
    y = df['sales']
    
    # Split into train/test (80/20)
    split = int(len(df) * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]