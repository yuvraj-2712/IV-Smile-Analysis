import pandas as pd
import numpy as np
import yfinance as yf

def calculate_historical_metrics(series):
    # Drop NAs for calculation
    valid_series = series.dropna()
    if len(valid_series) == 0:
        return pd.Series(index=series.index, dtype=float), pd.Series(index=series.index, dtype=float)
    
    mean = valid_series.mean()
    std = valid_series.std()
    
    # Calculate z-score
    if std == 0 or pd.isna(std):
        z_score = pd.Series(0, index=series.index)
    else:
        z_score = (series - mean) / std
        
    # Calculate percentile
    percentile = series.rank(pct=True)
    
    return z_score, percentile

def main():
    print("Downloading data from yfinance...")
    tickers = {
        'SPX': '^GSPC',
        'VIX': '^VIX',
        'VVIX': '^VVIX',
        'VIX9D': '^VIX9D',
        'VIX3M': '^VIX3M',
        'SKEW': '^SKEW',
        'US 2Y': '^IRX', # Using IRX as fallback/proxy for short-term rate if US2Y fails, but will try to label US 2Y
        'US 10Y': '^TNX',
        'MOVE': '^MOVE',
        'DXY': 'DX-Y.NYB',
        'WTI': 'CL=F'
    }
    
    # Try downloading ^US2Y separately to avoid locking issues, if fails use IRX
    try:
        us2y_data = yf.download('^US2Y', period='5y', threads=False)
        if not us2y_data.empty:
            tickers['US 2Y'] = '^US2Y'
    except:
        pass
        
    yf_data = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, period='5y', threads=False)
            if not df.empty and 'Close' in df.columns:
                # Handle MultiIndex columns from yfinance >= 0.2.0
                if isinstance(df.columns, pd.MultiIndex):
                    yf_data[name] = df['Close'][ticker].copy()
                else:
                    yf_data[name] = df['Close'].copy()
            else:
                print(f"Warning: No data for {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    # Combine into a single dataframe
    df_yf = pd.DataFrame(yf_data)
    df_yf.index = pd.to_datetime(df_yf.index).normalize()
    
    # Calculate SPX returns
    if 'SPX' in df_yf.columns:
        df_yf['SPX 1D'] = np.log(df_yf['SPX'] / df_yf['SPX'].shift(1))
        df_yf['SPX 5D'] = np.log(df_yf['SPX'] / df_yf['SPX'].shift(5))
        df_yf['SPX 20D'] = np.log(df_yf['SPX'] / df_yf['SPX'].shift(20))
        df_yf['20D RV'] = df_yf['SPX 1D'].rolling(window=20).std() * np.sqrt(252)
        
    # Calculate Spot-Vol (Correlation between SPX 1D and VIX 1D)
    if 'SPX' in df_yf.columns and 'VIX' in df_yf.columns:
        vix_1d = np.log(df_yf['VIX'] / df_yf['VIX'].shift(1))
        df_yf['Spot-Vol'] = df_yf['SPX 1D'].rolling(window=20).corr(vix_1d)

    # Load ladder data for ATM IV
    print("Loading ladder data...")
    try:
        df_ladder = pd.read_excel('ladder_data.xlsx')
        atm_df = df_ladder[df_ladder['Label 1'] == 'ATM'][['Date', 'IV 1']].rename(columns={'IV 1': 'ATM IV'})
        atm_df['Date'] = pd.to_datetime(atm_df['Date']).dt.normalize()
        # Take first if duplicates exist
        atm_df = atm_df.groupby('Date').first().reset_index()
        atm_df.set_index('Date', inplace=True)
        
        # Merge ATM IV into df_yf
        df_yf = df_yf.join(atm_df, how='left')
    except Exception as e:
        print(f"Error loading ladder data: {e}")
        df_yf['ATM IV'] = np.nan

    # Calculate IV - RV
    if 'ATM IV' in df_yf.columns and '20D RV' in df_yf.columns:
        df_yf['IV - RV'] = df_yf['ATM IV'] - df_yf['20D RV']

    # Define indicators to include
    indicators = [
        'SPX 1D', 'SPX 5D', 'SPX 20D', 'ATM IV', '20D RV', 'IV - RV',
        'VIX', 'Spot-Vol', 'VVIX', 'VIX9D', 'VIX3M', 'SKEW', 
        'US 2Y', 'US 10Y', 'MOVE', 'DXY', 'WTI'
    ]
    
    # Filter to only indicators that exist in df_yf
    existing_indicators = [i for i in indicators if i in df_yf.columns]
    
    print("Calculating metrics and building sheets...")
    sheets = {}
    summary_data = []
    
    # Process each indicator
    for ind in existing_indicators:
        series = df_yf[ind]
        
        # Ensure series is numeric and drop complete NaNs to calculate metrics properly
        series = pd.to_numeric(series, errors='coerce')
        
        z_score, percentile = calculate_historical_metrics(series)
        
        sheet_df = pd.DataFrame({
            'Date': series.index,
            'Value': series.values,
            'Z-score': z_score.values,
            'Percentile': percentile.values
        })
        # Drop rows where Value is NaN for clean sheets
        sheet_df = sheet_df.dropna(subset=['Value'])
        sheets[ind] = sheet_df
        
        if not sheet_df.empty:
            # Get latest available valid value
            latest_row = sheet_df.iloc[-1]
            summary_data.append({
                'Indicator': ind,
                'Current Value': latest_row['Value'],
                'Z-score': latest_row['Z-score'],
                'Percentile': latest_row['Percentile']
            })
        else:
            summary_data.append({
                'Indicator': ind,
                'Current Value': np.nan,
                'Z-score': np.nan,
                'Percentile': np.nan
            })
            
    summary_df = pd.DataFrame(summary_data)
    
    out_file = 'market_dashboard_full.xlsx'
    print(f"Writing to {out_file}...")
    with pd.ExcelWriter(out_file) as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        for ind in existing_indicators:
            sheets[ind].to_excel(writer, sheet_name=ind[:31], index=False) # Sheet names max 31 chars
            
    print("Done!")

if __name__ == "__main__":
    main()
