import pandas as pd
import numpy as np

def calculate_percentile_252(series, current_date):
    """Calculates the percentile using only data up to current_date, trailing 252 days."""
    hist = series.loc[:current_date].dropna()
    if len(hist) < 2:
        return np.nan
    # Use trailing 252 days max
    hist = hist.iloc[-252:]
    current_val = hist.iloc[-1]
    # Percentile
    pct = (hist < current_val).mean() * 100
    return pct

def interpret_percentile(pct):
    if pd.isna(pct): return "N/A"
    if pct <= 20: return "Low"
    elif pct <= 50: return "Normal"
    elif pct <= 75: return "Elevated"
    elif pct <= 90: return "High"
    else: return "Extreme"

def get_manual_regime(date1, market_sheets, df_ladder):
    current_date = pd.to_datetime(date1)
    
    # 1. 60D ATM IV
    atm_series = df_ladder.pivot_table(index='Date', columns='Label 1', values='IV 1')['ATM'].dropna()
    
    # Build a unified DataFrame for the indicators
    df_list = []
    df_list.append(atm_series.rename('ATM IV'))
    
    for m in ['20D RV', 'VIX9D', 'VIX', 'VIX3M', 'VVIX', 'SPX 1D', 'SPX 5D']:
        if m in market_sheets:
            temp = market_sheets[m][['Date', 'Value']].set_index('Date')['Value'].rename(m)
            df_list.append(temp)
            
    df_master = pd.concat(df_list, axis=1).ffill()
    
    # Check if date exists
    if current_date not in df_master.index:
        # Use last available before current_date
        past = df_master.loc[:current_date]
        if past.empty:
            return None
        current_date = past.index[-1]
        
    # Calculate ratios
    df_master['VIX9D/VIX'] = df_master['VIX9D'] / df_master['VIX']
    df_master['VIX/VIX3M'] = df_master['VIX'] / df_master['VIX3M']
    
    indicators = ['ATM IV', '20D RV', 'VIX9D/VIX', 'VIX/VIX3M', 'VVIX']
    
    results = []
    pcts = []
    for ind in indicators:
        val = df_master.loc[current_date, ind] if ind in df_master.columns else np.nan
        pct = calculate_percentile_252(df_master[ind] if ind in df_master.columns else pd.Series(dtype=float), current_date)
        state = interpret_percentile(pct)
        results.append({
            'Indicator': ind,
            'Current Value': val,
            'Percentile': pct,
            'State': state
        })
        if pd.notna(pct): pcts.append(pct)
        
    top_table = pd.DataFrame(results).set_index('Indicator')
    
    stress_score = np.median(pcts) if pcts else 0
    stress_count = sum(1 for p in pcts if p >= 75)
    extreme_count = sum(1 for p in pcts if p >= 90)
    
    # Regime Logic
    atm_extreme = results[0]['Percentile'] >= 90 if pd.notna(results[0]['Percentile']) else False
    rv_extreme = results[1]['Percentile'] >= 90 if pd.notna(results[1]['Percentile']) else False
    vix9d_extreme = results[2]['Percentile'] >= 90 if pd.notna(results[2]['Percentile']) else False
    vix3m_extreme = results[3]['Percentile'] >= 90 if pd.notna(results[3]['Percentile']) else False
    
    regime = "NORMAL / BALANCED"
    
    if extreme_count >= 3 and (atm_extreme or rv_extreme or vix9d_extreme or vix3m_extreme):
        regime = "SHOCK / CRISIS"
    elif stress_score >= 70 and (stress_count >= 3 or extreme_count >= 2):
        regime = "DEFENSIVE STRESS"
    elif (stress_score >= 50 and stress_score < 70) or stress_count >= 2:
        regime = "VOL TENSION / TRANSITION"
    elif stress_score < 30 and extreme_count == 0 and stress_count <= 1:
        regime = "LOW-VOL CARRY"
        
    # Get SPX modifiers
    spx1d = df_master.loc[current_date, 'SPX 1D'] if 'SPX 1D' in df_master.columns else np.nan
    spx5d = df_master.loc[current_date, 'SPX 5D'] if 'SPX 5D' in df_master.columns else np.nan
    
    text = f"{stress_count}/5 indicators indicate elevated stress or higher. Current regime is deterministically classified as {regime}."
    
    out = {
        'Table': top_table,
        'Stress Score': stress_score,
        'Stress Count': stress_count,
        'Extreme Count': extreme_count,
        'Regime': regime,
        'SPX 1D': spx1d,
        'SPX 5D': spx5d,
        'Text': text
    }
    return out

def get_historical_regimes(date1, market_sheets, df_ladder):
    # This generates the regime timeline [-10, +10] dynamically avoiding look-ahead bias
    # because get_manual_regime limits data to `current_date` natively!
    
    atm_series = df_ladder.pivot_table(index='Date', columns='Label 1', values='IV 1')['ATM'].dropna()
    dates = atm_series.index
    
    d1 = pd.to_datetime(date1)
    if d1 not in dates:
        # fallback
        past = dates[dates <= d1]
        if len(past) == 0: return pd.DataFrame()
        d1 = past[-1]
        
    idx = list(dates).index(d1)
    
    start_idx = max(0, idx - 10)
    end_idx = min(len(dates) - 1, idx + 10)
    
    target_dates = dates[start_idx:end_idx+1]
    
    history = []
    # Pre-build df_master to pass into a faster loop
    df_list = [atm_series.rename('ATM IV')]
    for m in ['20D RV', 'VIX9D', 'VIX', 'VIX3M', 'VVIX']:
        if m in market_sheets:
            temp = market_sheets[m][['Date', 'Value']].set_index('Date')['Value'].rename(m)
            df_list.append(temp)
    df_master = pd.concat(df_list, axis=1).ffill()
    df_master['VIX9D/VIX'] = df_master['VIX9D'] / df_master['VIX']
    df_master['VIX/VIX3M'] = df_master['VIX'] / df_master['VIX3M']
    indicators = ['ATM IV', '20D RV', 'VIX9D/VIX', 'VIX/VIX3M', 'VVIX']
    
    for d in target_dates:
        # Fast regime calc inline
        pcts = []
        atm_ext = rv_ext = v9_ext = v3_ext = False
        
        for i, ind in enumerate(indicators):
            if ind in df_master.columns:
                hist = df_master[ind].loc[:d].dropna()
                if len(hist) >= 2:
                    hist = hist.iloc[-252:]
                    cur = hist.iloc[-1]
                    pct = (hist < cur).mean() * 100
                    pcts.append(pct)
                    if i == 0 and pct >= 90: atm_ext = True
                    if i == 1 and pct >= 90: rv_ext = True
                    if i == 2 and pct >= 90: v9_ext = True
                    if i == 3 and pct >= 90: v3_ext = True
                    
        stress_score = np.median(pcts) if pcts else 0
        stress_count = sum(1 for p in pcts if p >= 75)
        extreme_count = sum(1 for p in pcts if p >= 90)
        
        regime = "NORMAL / BALANCED"
        r_val = 2
        
        if extreme_count >= 3 and (atm_ext or rv_ext or v9_ext or v3_ext):
            regime = "SHOCK / CRISIS"
            r_val = 5
        elif stress_score >= 70 and (stress_count >= 3 or extreme_count >= 2):
            regime = "DEFENSIVE STRESS"
            r_val = 4
        elif (stress_score >= 50 and stress_score < 70) or stress_count >= 2:
            regime = "VOL TENSION / TRANSITION"
            r_val = 3
        elif stress_score < 30 and extreme_count == 0 and stress_count <= 1:
            regime = "LOW-VOL CARRY"
            r_val = 1
            
        history.append({'Date': d, 'Regime': regime, 'R_Val': r_val})
        
    return pd.DataFrame(history)

def get_all_historical_regimes(market_sheets, df_ladder):
    atm_series = df_ladder.pivot_table(index='Date', columns='Label 1', values='IV 1')['ATM'].dropna()
    dates = atm_series.index
    
    df_list = [atm_series.rename('ATM IV')]
    for m in ['20D RV', 'VIX9D', 'VIX', 'VIX3M', 'VVIX']:
        if m in market_sheets:
            temp = market_sheets[m][['Date', 'Value']].set_index('Date')['Value'].rename(m)
            df_list.append(temp)
    df_master = pd.concat(df_list, axis=1).ffill()
    df_master['VIX9D/VIX'] = df_master['VIX9D'] / df_master['VIX']
    df_master['VIX/VIX3M'] = df_master['VIX'] / df_master['VIX3M']
    indicators = ['ATM IV', '20D RV', 'VIX9D/VIX', 'VIX/VIX3M', 'VVIX']
    
    history = []
    
    for d in dates:
        pcts = []
        atm_ext = rv_ext = v9_ext = v3_ext = False
        
        for i, ind in enumerate(indicators):
            if ind in df_master.columns:
                hist = df_master[ind].loc[:d].dropna()
                if len(hist) >= 2:
                    hist = hist.iloc[-252:]
                    cur = hist.iloc[-1]
                    pct = (hist < cur).mean() * 100
                    pcts.append(pct)
                    if i == 0 and pct >= 90: atm_ext = True
                    if i == 1 and pct >= 90: rv_ext = True
                    if i == 2 and pct >= 90: v9_ext = True
                    if i == 3 and pct >= 90: v3_ext = True
                    
        stress_score = np.median(pcts) if pcts else 0
        stress_count = sum(1 for p in pcts if p >= 75)
        extreme_count = sum(1 for p in pcts if p >= 90)
        
        regime = "NORMAL / BALANCED"
        r_val = 2
        
        if extreme_count >= 3 and (atm_ext or rv_ext or v9_ext or v3_ext):
            regime = "SHOCK / CRISIS"
            r_val = 5
        elif stress_score >= 70 and (stress_count >= 3 or extreme_count >= 2):
            regime = "DEFENSIVE STRESS"
            r_val = 4
        elif (stress_score >= 50 and stress_score < 70) or stress_count >= 2:
            regime = "VOL TENSION / TRANSITION"
            r_val = 3
        elif stress_score < 30 and extreme_count == 0 and stress_count <= 1:
            regime = "LOW-VOL CARRY"
            r_val = 1
            
        history.append({'Date': d, 'Regime_Label': regime, 'Regime_ID': r_val})
        
    return pd.DataFrame(history).set_index('Date')

