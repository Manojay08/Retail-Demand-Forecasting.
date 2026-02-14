import pandas as pd
import numpy as np
import os

def generate_dummy_data(filepath):
    """Generates a realistic-looking retail dataset for testing."""
    print(f"Generating dummy data at {filepath}...")
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    stores = ['Store_A', 'Store_B']
    items = ['Item_1', 'Item_2', 'Item_3']
    
    data = []
    for date in dates:
        for store in stores:
            for item in items:
                # Simulate seasonality and random noise
                base_sales = np.random.randint(50, 150)
                seasonality = 10 * np.sin(date.dayofyear / 365 * 2 * np.pi)
                is_holiday = 1 if date.month == 12 and date.day == 25 else 0
                promo = np.random.choice([0, 1], p=[0.9, 0.1])
                
                sales = max(0, int(base_sales + seasonality + (promo * 50) + np.random.normal(0, 10)))
                
                data.append([date, store, item, sales, is_holiday, promo])
                
    df = pd.DataFrame(data, columns=['date', 'store_id', 'item_id', 'sales', 'is_holiday', 'is_promo'])
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    return df

def load_and_preprocess(data_path):
    """
    Loads data from CSV. If file doesn't exist, generates dummy data.
    """
    file_path = os.path.join(data_path, 'sales_data.csv')
    
    if not os.path.exists(file_path):
        print("Data file not found. Generating sample data...")
        df = generate_dummy_data(file_path)
    else:
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])

    # Basic Preprocessing
    # fillna, type conversion, etc.
    df['sales'] = df['sales'].fillna(0)
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    
    print(f"Data Loaded: {df.shape[0]} rows.")
    return df
