import argparse
import os
import pandas as pd
from util.preprocess import load_and_preprocess
from models.tft import TFTForecaster

def main():
    parser = argparse.ArgumentParser(description='Retail Sales Forecasting Pipeline')
    parser.add_argument('--data', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--method', type=str, default='avg', choices=['avg', 'tft'], help='Forecasting method')
    parser.add_argument('--output', type=str, default='./submissions/forecast.csv', help='Output path')
    
    args = parser.parse_args()
    
    print(f"--- Starting Pipeline with method: {args.method.upper()} ---")

    # 1. Load Data
    df = load_and_preprocess(args.data)

    # 2. Select Method
    if args.method == 'tft':
        model = TFTForecaster(max_prediction_length=7, max_encoder_length=30)
        model.train(df)
        predictions = model.predict(df)
        
        # Simple formatting for output (just saving raw values for demo)
        output_df = pd.DataFrame(predictions.numpy())
        output_df.columns = [f"Day_{i+1}" for i in range(output_df.shape[1])]
        
    elif args.method == 'avg':
        print("Running Average Baseline...")
        # Simple Logic: Group by item and take mean
        avg_sales = df.groupby('item_id')['sales'].mean().reset_index()
        output_df = avg_sales
        
    # 3. Save Results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"--- Pipeline Finished. Results saved to {args.output} ---")

if __name__ == "__main__":
    main()
