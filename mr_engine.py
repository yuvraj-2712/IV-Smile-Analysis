import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

def calc_mr_stats(series):
    s = series.dropna()
    if len(s) < 30:
        return None
    
    stats = {}
    
    # 1. Stationarity
    try:
        adf = adfuller(s, autolag='AIC')
        stats['ADF_p'] = adf[1]
    except:
        stats['ADF_p'] = 1.0
        
    try:
        kpss_stat = kpss(s, regression='c', nlags='auto')
        stats['KPSS_p'] = kpss_stat[1]
    except:
        stats['KPSS_p'] = 0.0
        
    # 2. AR(1) and Half-life
    try:
        y = s.values[1:]
        X = sm.add_constant(s.values[:-1])
        model = sm.OLS(y, X).fit()
        phi = model.params[1]
        stats['AR1_phi'] = phi
        if 0 < phi < 1:
            stats['Half_Life'] = -np.log(2) / np.log(phi)
        else:
            stats['Half_Life'] = np.nan
    except:
        stats['AR1_phi'] = np.nan
        stats['Half_Life'] = np.nan
        
    # 3. Z-Score Deviation Test (Leakage Safe)
    rolling_mean = s.shift(1).expanding(min_periods=10).mean()
    rolling_std = s.shift(1).expanding(min_periods=10).std()
    z = (s - rolling_mean) / rolling_std
    
    # Forward Reversals
    for h in [1, 3, 5, 10]:
        change = s.shift(-h) - s
        if len(z.dropna()) > 0 and len(change.dropna()) > 0:
            df = pd.DataFrame({'Z': z, 'd': change}).dropna()
            if len(df) > 10:
                stats[f'MR_{h}D_corr'] = df['Z'].corr(-df['d'])
            else:
                stats[f'MR_{h}D_corr'] = np.nan
                
    # MR Score out of 100
    score = 0
    if stats.get('ADF_p', 1) < 0.05: score += 20
    if stats.get('KPSS_p', 0) > 0.05: score += 10
    
    phi = stats.get('AR1_phi', 1)
    if not np.isnan(phi) and phi < 0.95 and phi > 0: score += 20
    elif not np.isnan(phi) and phi < 0.99 and phi > 0: score += 10
        
    corr_5d = stats.get('MR_5D_corr', 0)
    if not np.isnan(corr_5d) and corr_5d > 0.1: score += 20
    elif not np.isnan(corr_5d) and corr_5d > 0: score += 10
        
    corr_10d = stats.get('MR_10D_corr', 0)
    if not np.isnan(corr_10d) and corr_10d > 0.1: score += 30
    elif not np.isnan(corr_10d) and corr_10d > 0: score += 15
        
    stats['MR_Score'] = min(100, score)
    
    if score >= 70:
        stats['Classification'] = 'STRONG'
    elif score >= 40:
        stats['Classification'] = 'WEAK'
    else:
        stats['Classification'] = 'NON-MEAN-REVERTING'
        
    return stats

def build_smile_timeseries(df_ladder):
    # Pivot ladder to get IVs by date and delta label
    # Assuming df_ladder has 'Date', 'Label 1', 'IV 1'
    df = df_ladder.pivot_table(index='Date', columns='Label 1', values='IV 1')
    
    ts = pd.DataFrame(index=df.index)
    
    atm = df.get('ATM', np.nan)
    p25 = df.get('25 delta put', np.nan)
    c25 = df.get('25 delta call', np.nan)
    p10 = df.get('10 delta put', np.nan)
    c10 = df.get('10 delta call', np.nan)
    
    ts['ATM IV'] = atm
    ts['Put Skew (25P - ATM)'] = p25 - atm
    ts['10D Put Skew'] = p10 - atm
    ts['Call Skew (ATM - 25C)'] = atm - c25
    ts['10D Call Skew'] = atm - c10
    ts['25D RR (25P - 25C)'] = p25 - c25
    ts['10D RR'] = p10 - c10
    ts['25D BF'] = (p25 + c25) / 2 - atm
    ts['10D BF'] = (p10 + c10) / 2 - atm
    ts['Put Wing (10P - 25P)'] = p10 - p25
    ts['Call Wing (10C - 25C)'] = c10 - c25
    ts['Put Curvature'] = ts['Put Wing (10P - 25P)'] - ts['Put Skew (25P - ATM)']
    ts['Call Curvature'] = ts['Call Wing (10C - 25C)'] - ts['Call Skew (ATM - 25C)']
    
    return ts.ffill().dropna()

def analyze_all(df_ladder, regimes_df):
    results = []
    
    # Build Smile timeseries
    master_df = build_smile_timeseries(df_ladder)
    candidates = master_df.columns.tolist()
    
    # Join regimes
    if regimes_df is not None:
        # Align indexes since regimes_df index is datetime
        master_df = master_df.join(regimes_df[['Regime_Label']])
    
    # Mode 1: Single
    for f in candidates:
        stats = calc_mr_stats(master_df[f])
        if stats:
            stats['Series'] = f
            stats['Mode'] = 'Single'
            
            # Regime Conditionals
            if 'Regime_Label' in master_df.columns:
                for reg in master_df['Regime_Label'].dropna().unique():
                    reg_s = master_df[master_df['Regime_Label'] == reg][f]
                    r_stats = calc_mr_stats(reg_s)
                    if r_stats:
                        stats[f'Score_{reg}'] = r_stats['MR_Score']
                        
            results.append(stats)
                
    # Mode 2: Pairwise
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            f1, f2 = candidates[i], candidates[j]
            spread = master_df[f1] - master_df[f2]
            stats = calc_mr_stats(spread)
            if stats:
                stats['Series'] = f"{f1} - {f2}"
                stats['Mode'] = 'Spread'
                results.append(stats)
                    
    return pd.DataFrame(results)
