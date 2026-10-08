import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from scipy import stats

def prepare_data(df_ladder, regimes_df, market_sheets=None):
    df_iv = df_ladder.pivot_table(index='Date', columns='Label 1', values='IV 1')
    spot_df = df_ladder.groupby('Date')['Underlying'].first()
    master = pd.DataFrame(index=df_iv.index)
    
    delta_cols = [
        '5 delta put', '10 delta put', '15 delta put', '20 delta put', '25 delta put',
        '30 delta put', '35 delta put', '40 delta put', '45 delta put', 'ATM',
        '45 delta call', '40 delta call', '35 delta call', '30 delta call', '25 delta call',
        '20 delta call', '15 delta call', '10 delta call', '5 delta call'
    ]
    
    for col in delta_cols:
        master[f'IV_{col}'] = df_iv[col]
        master[f'Target_dIV_{col}'] = df_iv[col] - df_iv[col].shift(1)
        
    master['Spot'] = spot_df
    master['r1'] = np.log(master['Spot'] / master['Spot'].shift(1))
    master['abs_r1'] = master['r1'].abs()
    master['r3'] = np.log(master['Spot'] / master['Spot'].shift(3))
    master['r5'] = np.log(master['Spot'] / master['Spot'].shift(5))
    master['RV_20'] = master['r1'].rolling(20).std() * np.sqrt(252)
    master['r1_z'] = (master['r1'] - master['r1'].rolling(20).mean()) / master['r1'].rolling(20).std()
    
    for col in delta_cols:
        master[f'Lag_IV_{col}'] = master[f'IV_{col}'].shift(1)
        master[f'Lag_dIV_{col}'] = master[f'Target_dIV_{col}'].shift(1)
        
    master['Lag_25D_Put_Skew'] = master['Lag_IV_25 delta put'] - master['Lag_IV_ATM']
    master['Lag_10D_Put_Skew'] = master['Lag_IV_10 delta put'] - master['Lag_IV_ATM']
    master['Lag_25D_Call_Skew'] = master['Lag_IV_ATM'] - master['Lag_IV_25 delta call']
    master['Lag_25D_BF'] = (master['Lag_IV_25 delta put'] + master['Lag_IV_25 delta call']) / 2 - master['Lag_IV_ATM']
    
    if regimes_df is not None and 'Regime_ID' in regimes_df.columns:
        master['Regime'] = regimes_df['Regime_ID'].reindex(master.index).ffill().fillna(0)
    else:
        master['Regime'] = 0
        
    if market_sheets is not None:
        for ind, df in market_sheets.items():
            if 'Value' in df.columns:
                temp = df.set_index('Date')['Value']
                # Join exact dates
                master[f'Macro_{ind}'] = temp.reindex(master.index).ffill()
                master[f'Lag_Macro_{ind}'] = master[f'Macro_{ind}'].shift(1)
                master[f'Lag_dMacro_{ind}'] = master[f'Macro_{ind}'].shift(1) - master[f'Macro_{ind}'].shift(2)
        
    return master.dropna()

def get_hit_rate(signals, actual_moves):
    hits = (signals * actual_moves) > 0
    trades = signals != 0
    t = trades.sum()
    h = hits.sum()
    hr = h / t if t > 0 else 0
    # Binomial 95% CI
    ci = 1.96 * np.sqrt((hr * (1 - hr)) / t) if t > 0 else 0
    return hr, h, t, ci

def run_backtest(df_ladder, regimes_df, market_sheets=None, permute=False):
    master = prepare_data(df_ladder, regimes_df, market_sheets)
    
    split_idx = int(len(master) * 0.8)
    train = master.iloc[:split_idx].copy()
    test = master.iloc[split_idx:].copy()
    
    delta_cols = [c.replace('IV_', '') for c in master.columns if c.startswith('IV_') and not c.startswith('Lag_')]
    feature_cols = ['r1', 'abs_r1', 'r3', 'r5', 'RV_20', 'r1_z', 'Lag_25D_Put_Skew', 'Lag_10D_Put_Skew', 'Lag_25D_BF']
    for c in delta_cols:
        feature_cols.extend([f'Lag_IV_{c}', f'Lag_dIV_{c}'])
        
    X_train = train[feature_cols].values
    X_test = test[feature_cols].values
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    models = {
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_leaf=5, random_state=42),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_depth=5, random_state=42)
    }
    
    preds = {m: pd.DataFrame(index=test.index, columns=delta_cols) for m in models.keys()}
    preds['RegimeLocal'] = pd.DataFrame(index=test.index, columns=delta_cols)
    
    # Baseline A: Unconditional MR (Predicted dIV = 0)
    preds['Baseline A (Unconditional MR)'] = pd.DataFrame(0, index=test.index, columns=delta_cols)
    
    for d_col in delta_cols:
        y_train = train[f'Target_dIV_{d_col}'].values
        if permute:
            np.random.shuffle(y_train)
            
        for m_name, model in models.items():
            model.fit(X_train_s, y_train)
            preds[m_name][d_col] = model.predict(X_test_s)
            
        # Regime Local
        from sklearn.linear_model import Ridge
        reg_preds = []
        for i in range(len(test)):
            r = test['Regime'].iloc[i]
            train_r = train[train['Regime'] == r]
            if len(train_r) > 10:
                X_tr = scaler.transform(train_r[feature_cols].values)
                y_tr = train_r[f'Target_dIV_{d_col}'].values
                rmodel = Ridge(alpha=1.0)
                rmodel.fit(X_tr, y_tr)
                reg_preds.append(rmodel.predict(X_test_s[i:i+1])[0])
            else:
                reg_preds.append(models['ElasticNet'].predict(X_test_s[i:i+1])[0])
        preds['RegimeLocal'][d_col] = reg_preds
        
    backtest_stats = []
    
    for m_name in preds.keys():
        pred_df = preds[m_name]
        
        fair_iv = test[[f'Lag_IV_{c}' for c in delta_cols]].rename(columns=lambda x: x.replace('Lag_IV_', '')) + pred_df
        actual_iv = test[[f'IV_{c}' for c in delta_cols]].rename(columns=lambda x: x.replace('IV_', ''))
        
        residual = actual_iv - fair_iv
        rel_residual = residual.sub(residual.median(axis=1), axis=0)
        
        if m_name == 'Baseline A (Unconditional MR)':
            sigma_d = 1.4826 * rel_residual.abs().median()
        else:
            train_preds = pd.DataFrame(index=train.index, columns=delta_cols)
            for d_col in delta_cols:
                if m_name == 'RegimeLocal':
                    train_preds[d_col] = models['ElasticNet'].predict(X_train_s)
                else:
                    train_preds[d_col] = models[m_name].predict(X_train_s)
                    
            train_fair = train[[f'Lag_IV_{c}' for c in delta_cols]].rename(columns=lambda x: x.replace('Lag_IV_', '')) + train_preds
            train_actual = train[[f'IV_{c}' for c in delta_cols]].rename(columns=lambda x: x.replace('IV_', ''))
            train_res = train_actual - train_fair
            train_rel_res = train_res.sub(train_res.median(axis=1), axis=0)
            sigma_d = 1.4826 * train_rel_res.abs().median()
            
        sigma_d = sigma_d.replace(0, 0.0001)
        Z = rel_residual / sigma_d
        
        next_rel_res = rel_residual.shift(-1)
        delta_rel_res = next_rel_res - rel_residual
        
        Z_valid = Z.iloc[:-1]
        delta_valid = delta_rel_res.iloc[:-1]
        signal = -np.sign(Z_valid)
        
        for thresh in [0, 1.0, 1.5, 2.0]:
            active_signals = signal.where(Z_valid.abs() >= thresh, 0)
            hr, h, t, ci = get_hit_rate(active_signals.values.flatten(), delta_valid.values.flatten())
            backtest_stats.append({
                'Model': m_name,
                'Threshold': f'|Z| >= {thresh}' if thresh > 0 else 'All Trades (No Threshold)',
                'Trades': t,
                'Hits': h,
                'Hit Rate': f"{hr*100:.1f}%",
                '95% CI': f"±{ci*100:.1f}%"
            })
            
    return pd.DataFrame(backtest_stats), master
