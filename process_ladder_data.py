import pandas as pd
import numpy as np

def calculate_historical_metrics(series):
    # We will use the entire dataset for z-score and percentile
    z_score = (series - series.mean()) / series.std()
    percentile = series.rank(pct=True)
    return z_score, percentile

def main():
    print("Loading data...")
    df = pd.read_excel('ladder_data.xlsx')
    
    # 1. Extract daily spot values and ATM IV
    # Get one row per date for Underlying
    spot_df = df.groupby('Date')['Underlying'].first().reset_index()
    spot_df = spot_df.sort_values('Date').reset_index(drop=True)
    
    # ATM IV: Filter for Label 1 == 'ATM'
    atm_df = df[df['Label 1'] == 'ATM'][['Date', 'IV 1']].rename(columns={'IV 1': 'ATM_IV'})
    
    # Merge them
    daily_data = pd.merge(spot_df, atm_df, on='Date', how='left')
    
    # Ensure Date is sorted
    daily_data['Date'] = pd.to_datetime(daily_data['Date'])
    daily_data = daily_data.sort_values('Date').reset_index(drop=True)
    
    # 2. Calculate Returns
    daily_data['1D_Log_Ret'] = np.log(daily_data['Underlying'] / daily_data['Underlying'].shift(1))
    daily_data['5D_Log_Ret'] = np.log(daily_data['Underlying'] / daily_data['Underlying'].shift(5))
    daily_data['20D_Log_Ret'] = np.log(daily_data['Underlying'] / daily_data['Underlying'].shift(20))
    
    # 3. Calculate 20D Realized Annual Volatility (Close-to-Close)
    # std of 1D log returns over 20 days, annualized
    daily_data['20D_RV'] = daily_data['1D_Log_Ret'].rolling(window=20).std() * np.sqrt(252)
    
    # 4. ATM IV - 20D RV
    # IV in the dataset seems to be in decimal form (e.g., 0.133 for 13.3%)
    daily_data['IV_Minus_RV'] = daily_data['ATM_IV'] - daily_data['20D_RV']
    
    # Function to create sheet dataframe
    def create_sheet_df(date_series, value_series, col_name):
        df_sheet = pd.DataFrame({'Date': date_series, col_name: value_series})
        z_col = 'Z-Score'
        pct_col = 'Percentile'
        df_sheet[z_col], df_sheet[pct_col] = calculate_historical_metrics(df_sheet[col_name])
        return df_sheet

    # Create sheets
    sheets = {}
    sheets['1D_Returns'] = create_sheet_df(daily_data['Date'], daily_data['1D_Log_Ret'], 'SPX 1D Log Return')
    sheets['5D_Returns'] = create_sheet_df(daily_data['Date'], daily_data['5D_Log_Ret'], 'SPX 5D Log Return')
    sheets['20D_Returns'] = create_sheet_df(daily_data['Date'], daily_data['20D_Log_Ret'], 'SPX 20D Log Return')
    sheets['ATM_IV'] = create_sheet_df(daily_data['Date'], daily_data['ATM_IV'], 'ATM IV')
    sheets['20D_RV'] = create_sheet_df(daily_data['Date'], daily_data['20D_RV'], '20D Realized Volatility')
    sheets['IV_Minus_RV'] = create_sheet_df(daily_data['Date'], daily_data['IV_Minus_RV'], 'ATM IV - 20D RV')
    
    # Write to Excel
    out_file = 'processed_ladder_data.xlsx'
    print(f"Writing to {out_file}...")
    with pd.ExcelWriter(out_file) as writer:
        sheets['1D_Returns'].to_excel(writer, sheet_name='Sheet 1', index=False)
        sheets['5D_Returns'].to_excel(writer, sheet_name='Sheet 2', index=False)
        sheets['20D_Returns'].to_excel(writer, sheet_name='Sheet 3', index=False)
        sheets['ATM_IV'].to_excel(writer, sheet_name='Sheet 4', index=False)
        sheets['20D_RV'].to_excel(writer, sheet_name='Sheet 5', index=False)
        sheets['IV_Minus_RV'].to_excel(writer, sheet_name='Sheet 6', index=False)
        
    print("Done!")

if __name__ == "__main__":
    main()
