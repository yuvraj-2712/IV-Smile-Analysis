import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import mr_engine
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide", page_title="Options Market Dashboard")

# Inject Custom CSS for a Premium Look
st.markdown("""
<style>
/* Main Background and Typography */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* Style the Tabs to look like modern pills */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #1a1c24;
    padding: 10px;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border-radius: 8px !important;
    padding-top: 10px;
    padding-bottom: 10px;
    padding-left: 20px;
    padding-right: 20px;
    color: #a0aabf;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background-color: #2e3b5e !important;
    color: #ffffff !important;
    font-weight: 600;
    box-shadow: 0 2px 4px rgba(0,0,0,0.5);
}

/* Style Metric Cards (Spot Level, Regime, etc) */
div[data-testid="metric-container"] {
    background-color: #1a1c24;
    border-radius: 10px;
    padding: 15px 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
    border: 1px solid #2d3342;
    transition: transform 0.2s ease-in-out;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    border-color: #4b5563;
}

/* DataFrame Styling tweaks */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}

/* Button Styling */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.6);
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_ladder_data():
    df = pd.read_excel('ladder_data.xlsx')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def load_market_data_v2():
    file_path = 'market_dashboard_full.xlsx'
    sheets = pd.read_excel(file_path, sheet_name=None)
    if 'Summary' in sheets:
        del sheets['Summary']
    if 'SR1=F' in sheets:
        del sheets['SR1=F']
    
    for k, v in sheets.items():
        v['Date'] = pd.to_datetime(v['Date'])
    return sheets

import torch
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal

@st.cache_data
def detect_regimes(market_sheets):
    """
    Regime Detection using Hidden Markov Model (HMM).
    Constructs a leakage-safe feature vector and fits a Gaussian HMM.
    """
    try:
        df_list = []
        features = ['VIX', 'ATM IV', 'SPX 1D', 'IV - RV', 'SKEW', 'MOVE']
        for f in features:
            if f in market_sheets:
                temp = market_sheets[f][['Date', 'Value']].rename(columns={'Value': f})
                df_list.append(temp.set_index('Date'))
        
        master = pd.concat(df_list, axis=1).ffill().dropna()
        if master.empty:
            return None, None, None
        
        # STEP 2: Leakage-safe rolling z-score (using expanding window strictly before t)
        Z_df = pd.DataFrame(index=master.index)
        for col in master.columns:
            shifted = master[col].shift(1)
            expanding_mean = shifted.expanding(min_periods=5).mean()
            expanding_std = shifted.expanding(min_periods=5).std()
            Z_df[col] = (master[col] - expanding_mean) / expanding_std
            
        Z_df = Z_df.dropna()
        if Z_df.empty:
            return None, None, None
            
        # STEP 4: Hidden Markov Model Estimation
        X = torch.tensor(Z_df.values, dtype=torch.float32).unsqueeze(0)
        n_states = 4
        
        # Initialize Normal distributions for each state
        dists = [Normal() for _ in range(n_states)]
        model = DenseHMM(distributions=dists)
        model.fit(X)
        
        # Get posterior probabilities and state predictions
        probs = model.predict_proba(X)[0].numpy()
        labels = np.argmax(probs, axis=1)
        
        master = master.loc[Z_df.index].copy()
        master['Regime_ID'] = labels
        master['Confidence'] = np.max(probs, axis=1) * 100
        
        for i in range(n_states):
            master[f'Prob_{i}'] = probs[:, i]
            
        # STEP 5: Economic Labeling based on VIX
        regime_vix = master.groupby('Regime_ID')['VIX'].median().sort_values()
        state_names = ['CALM', 'NORMAL', 'STRESS', 'POST-SHOCK']
        mapping = {regime_vix.index[i]: state_names[i] for i in range(n_states)}
        master['Regime_Label'] = master['Regime_ID'].map(mapping)
        
        # Duration calculation
        master = master.sort_index()
        master['Regime_Changed'] = master['Regime_ID'] != master['Regime_ID'].shift(1)
        master['Duration'] = master.groupby(master['Regime_Changed'].cumsum()).cumcount() + 1
        
        # Empirical Transition Matrix
        transitions = pd.crosstab(master['Regime_Label'].shift(1), master['Regime_Label'], normalize='index')
        
        return master, transitions, mapping
    except Exception as e:
        print(f"HMM Regime error: {e}")
        return None, None, None

# Helper function to get stats for a specific date
def get_vol_stats(df, date):
    day_df = df[df['Date'] == date]
    if day_df.empty:
        return None
    
    def get_iv(label):
        val = day_df[day_df['Label 1'] == label]['IV 1']
        return val.iloc[0] if not val.empty else np.nan

    atm = get_iv('ATM')
    p25 = get_iv('25 delta put')
    c25 = get_iv('25 delta call')
    p10 = get_iv('10 delta put')
    c10 = get_iv('10 delta call')

    stats = {
        'ATM IV': atm,
        '25D RR (25P - 25C)': p25 - c25 if pd.notna(p25) and pd.notna(c25) else np.nan,
        'Put Skew (25P - ATM)': p25 - atm if pd.notna(p25) and pd.notna(atm) else np.nan,
        'Call Skew (ATM - 25C)': atm - c25 if pd.notna(c25) and pd.notna(atm) else np.nan,
        'Put Wing (10P - 25P)': p10 - p25 if pd.notna(p10) and pd.notna(p25) else np.nan,
        'Call Wing (10C - 25C)': c10 - c25 if pd.notna(c10) and pd.notna(c25) else np.nan,
    }
    
    stats['Put Curvature'] = stats['Put Wing (10P - 25P)'] - stats['Put Skew (25P - ATM)'] if pd.notna(stats['Put Wing (10P - 25P)']) and pd.notna(stats['Put Skew (25P - ATM)']) else np.nan
    stats['Call Curvature'] = stats['Call Wing (10C - 25C)'] - stats['Call Skew (ATM - 25C)'] if pd.notna(stats['Call Wing (10C - 25C)']) and pd.notna(stats['Call Skew (ATM - 25C)']) else np.nan

    return stats

@st.cache_data
def load_mr_stats_v2(df_ladder, regimes_df):
    return mr_engine.analyze_all(df_ladder, regimes_df)

try:
    df_ladder = load_ladder_data()
    market_sheets = load_market_data_v2()
    import manual_regime
    import importlib
    importlib.reload(manual_regime)
    regimes_df = manual_regime.get_all_historical_regimes(market_sheets, df_ladder)
    # Ensure Regime_ID is available as string for grouping downstream
    regimes_df['Regime_ID'] = regimes_df['Regime_ID'].astype(str)
    # Dummy variables to avoid breaking existing signatures
    transitions, mapping = None, None
    available_dates = sorted(df_ladder['Date'].dt.date.unique())
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Initialize session state for date selection
if 'd1_sel' not in st.session_state:
    st.session_state.d1_sel = available_dates[-1] if available_dates else None

def shift_d1(delta):
    if available_dates:
        current_idx = available_dates.index(st.session_state.d1_sel)
        new_idx = max(1, min(len(available_dates) - 1, current_idx + delta))
        st.session_state.d1_sel = available_dates[new_idx]

# UI Global Controls
with st.sidebar:
    st.header("⚙️ Global Settings")
    st.divider()
    
    st.markdown("**Date 1 (Today)**")
    sc1, sc2, sc3 = st.columns([1, 3, 1])
    sc1.button("⬅️", key="d1_prev", on_click=shift_d1, args=(-1,))
    date1 = sc2.selectbox("Date 1", available_dates, key='d1_sel', label_visibility="collapsed")
    sc3.button("➡️", key="d1_next", on_click=shift_d1, args=(1,))
    
    st.markdown("**Date 2 (Yesterday)**")
    if available_dates:
        current_idx = available_dates.index(date1)
        date2 = available_dates[current_idx - 1] if current_idx > 0 else available_dates[0]
    else:
        date2 = None
    st.info(f"{date2}")
    
    st.divider()
    st.markdown("**Historical Window**")
    start_date = st.date_input("Window Start", min(available_dates) if available_dates else None)
    end_date = st.date_input("Window End", max(available_dates) if available_dates else None)

    st.divider()
    st.markdown("**3-Factor Master Engine**")
    f1_thresh = st.number_input("F1 Threshold (Time-Series)", value=1.5, step=0.1, min_value=0.5, max_value=4.0)
    f2_thresh = st.number_input("F2 Threshold (Spatial Arb)", value=1.5, step=0.1, min_value=0.5, max_value=4.0)
    f3_thresh = st.number_input("F3 Threshold (Macro Arb)", value=1.5, step=0.1, min_value=0.5, max_value=4.0)

st.title("Options Volatility & Arbitrage Engine")

# Tabs
tab_vol, tab_regime, tab_mr, tab_trade = st.tabs(["📈 Volatility Surface", "🧠 Regime Analysis", "📊 MR Classification", "🤖 Trade Backtester"])

with tab_vol:

    # --- SECTION 2: MACRO ENVIRONMENT & SMILE ---
    # 1. MACRO ENVIRONMENT HEADER
    d1_df = df_ladder[df_ladder['Date'].dt.date == date1]
    d2_df = df_ladder[df_ladder['Date'].dt.date == date2] if date2 else pd.DataFrame()
    
    if not d1_df.empty and not d2_df.empty and 'Underlying' in d1_df.columns:
        spot1 = d1_df['Underlying'].iloc[0]
        spot2 = d2_df['Underlying'].iloc[0]
        abs_chg = spot1 - spot2
        pct_chg = (abs_chg / spot2) * 100 if spot2 != 0 else 0
        
        import manual_regime
        r_out = manual_regime.get_manual_regime(date1, market_sheets, df_ladder)
        regime_label = r_out['Regime'] if r_out else "N/A"
        stress_score = r_out['Stress Score'] if r_out else 0
        
        atm_row_1 = d1_df[d1_df['Label 1'] == 'ATM']
        atm_val = atm_row_1['IV 1'].iloc[0] if not atm_row_1.empty else 0
        
        atm_row_2 = d2_df[d2_df['Label 1'] == 'ATM']
        atm_val_y = atm_row_2['IV 1'].iloc[0] if not atm_row_2.empty else 0
        
        st.markdown(f"### 🌐 Macro Environment: {date1}")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric(label=f"📈 SPX Spot Level", value=f"{spot1:,.2f}", delta=f"{abs_chg:+.2f} ({pct_chg:+.2f}%)")
        sc2.metric(label=f"🧠 Volatility Regime", value=regime_label, delta=f"Stress Score: {stress_score:.1f}", delta_color="off")
        sc3.metric(label=f"⚡ ATM Implied Vol", value=f"{atm_val*100:.2f}%", delta=f"{(atm_val-atm_val_y)*100:+.2f}%")
        st.divider()

    col_3factor, col_smile = st.columns([1, 1])

    with col_3factor:
        # --- SECTION 1: 3-FACTOR MASTER ENGINE ---
        st.subheader(f"Current 5D Spreads and Flies ({date1}) - 3 Factor Engine")
        st.markdown("Evaluates **F1 (Time-Series Z-Score)**, **F2 (Macro Fair Value Arb)**, and **F3 (Conditional Daily Change)**. If all three flag as Expensive or Cheap, it's a massive arbitrage.")
    
        iv_pure = df_ladder.pivot_table(index='Date', columns='Label 1', values='IV 1').dropna(axis=1, how='all')
        iv_pure.index = pd.to_datetime(iv_pure.index).date
    
        # Build Macro DF for F2 and F3
        spot_series = df_ladder.groupby(df_ladder['Date'].dt.date)['Underlying'].first()
        if 'ATM' in iv_pure.columns:
            macro_df = pd.DataFrame({
                'ATM_IV': iv_pure['ATM'],
                'd_ATM': iv_pure['ATM'].diff(),
                'Spot_Ret': spot_series.pct_change()
            }).dropna()
        else: macro_df = pd.DataFrame()
    
        ordered_cols = [
            '5 delta put', '10 delta put', '15 delta put', '20 delta put', '25 delta put',
            '30 delta put', '35 delta put', '40 delta put', '45 delta put', 'ATM',
            '45 delta call', '40 delta call', '35 delta call', '30 delta call', '25 delta call',
            '20 delta call', '15 delta call', '10 delta call', '5 delta call'
        ]
        cols = [c for c in ordered_cols if c in iv_pure.columns]
    
        import statsmodels.api as sm
        def run_3factor(series, m_df, spatial_df, t1, t2, t3):
            if len(series) < 50 or m_df.empty: return "N/A", "N/A", "N/A", "Neutral"
            c_idx = series.index.intersection(m_df.index)
            if not spatial_df.empty: c_idx = c_idx.intersection(spatial_df.index)
            if len(c_idx) < 50: return "N/A", "N/A", "N/A", "Neutral"
        
            s = series.loc[c_idx]
            m = m_df.loc[c_idx]
            sp = spatial_df.loc[c_idx] if not spatial_df.empty else pd.DataFrame()
        
            # F1: Time-Series Z-Score
            z1 = (s.iloc[-1] - s.mean()) / s.std() if s.std() != 0 else 0
            f1 = "Expensive" if z1 > t1 else ("Cheap" if z1 < -t1 else "-")
        
            # F2: Spatial Arb (Neighbors)
            if not sp.empty:
                try:
                    X2 = sm.add_constant(sp)
                    model2 = sm.OLS(s, X2).fit()
                    resid2 = s - model2.fittedvalues
                    z2 = resid2.iloc[-1] / resid2.std() if resid2.std() != 0 else 0
                    f2 = "Expensive" if z2 > t2 else ("Cheap" if z2 < -t2 else "-")
                except: f2 = "N/A"
            else:
                f2 = "N/A"
        
            # F3: Macro Fair Value Arb (ATM Level, dATM, Spot Ret)
            try:
                X3 = sm.add_constant(m[['ATM_IV', 'd_ATM', 'Spot_Ret']])
                model3 = sm.OLS(s, X3).fit()
                resid3 = s - model3.fittedvalues
                z3 = resid3.iloc[-1] / resid3.std() if resid3.std() != 0 else 0
                f3 = "Expensive" if z3 > t3 else ("Cheap" if z3 < -t3 else "-")
            except: f3 = "N/A"
        
            if f1 == "Expensive" and f2 == "Expensive" and f3 == "Expensive": cons = "STRONG SELL"
            elif f1 == "Cheap" and f2 == "Cheap" and f3 == "Cheap": cons = "STRONG BUY"
            else: cons = "-"
        
            return f1, f2, f3, cons
    
        hist_iv = iv_pure[iv_pure.index <= date1]
    
        if not hist_iv.empty:
            struct_data = []
            # Spreads
            for i in range(len(cols) - 1):
                leg1, leg2 = cols[i], cols[i+1]
                name = f"{leg1} vs {leg2}"
                series = (hist_iv[leg1] - hist_iv[leg2]).dropna()
            
                sp_dict = {}
                for j in range(max(0, i-5), i):
                    sp_dict[f'L_{i-j}'] = hist_iv[cols[j]] - hist_iv[cols[j+1]]
                for j in range(i+1, min(len(cols)-1, i+6)):
                    sp_dict[f'R_{j-i}'] = hist_iv[cols[j]] - hist_iv[cols[j+1]]
                spatial_df = pd.DataFrame(sp_dict).dropna()
            
                if not series.empty:
                    val = series.iloc[-1]
                    val_yday = series.loc[date2] if date2 in series.index else (series.iloc[-2] if len(series) > 1 else np.nan)
                    chg = val - val_yday if pd.notna(val_yday) else np.nan
                    f1, f2, f3, cons = run_3factor(series, macro_df, spatial_df, f1_thresh, f2_thresh, f3_thresh)
                    struct_data.append({'Type': 'Spread', 'Structure': name, 'Current': val, 'Chg': chg, 'F1': f1, 'F2': f2, 'F3': f3, 'Signal': cons})
        
            # Flies
            for i in range(len(cols) - 2):
                leg1, leg2, leg3 = cols[i], cols[i+1], cols[i+2]
                name = f"{leg1} / {leg2} / {leg3} Fly"
                series = (hist_iv[leg1] - 2*hist_iv[leg2] + hist_iv[leg3]).dropna()
            
                sp_dict = {}
                for j in range(max(0, i-5), i):
                    sp_dict[f'L_{i-j}'] = hist_iv[cols[j]] - 2*hist_iv[cols[j+1]] + hist_iv[cols[j+2]]
                for j in range(i+1, min(len(cols)-2, i+6)):
                    sp_dict[f'R_{j-i}'] = hist_iv[cols[j]] - 2*hist_iv[cols[j+1]] + hist_iv[cols[j+2]]
                spatial_df = pd.DataFrame(sp_dict).dropna()
            
                if not series.empty:
                    val = series.iloc[-1]
                    val_yday = series.loc[date2] if date2 in series.index else (series.iloc[-2] if len(series) > 1 else np.nan)
                    chg = val - val_yday if pd.notna(val_yday) else np.nan
                    f1, f2, f3, cons = run_3factor(series, macro_df, spatial_df, f1_thresh, f2_thresh, f3_thresh)
                    struct_data.append({'Type': 'Fly', 'Structure': name, 'Current': val, 'Chg': chg, 'F1': f1, 'F2': f2, 'F3': f3, 'Signal': cons})
                
            if struct_data:
                struct_df = pd.DataFrame(struct_data)
            
                def color_sig(val):
                    if val == 'STRONG SELL': return 'color: #ff3333; font-weight: bold;'
                    if val == 'STRONG BUY': return 'color: #00ff00; font-weight: bold;'
                    if val == 'Expensive': return 'color: #ff9999;'
                    if val == 'Cheap': return 'color: #99ff99;'
                    return ''
                
                def color_chg(val):
                    try:
                        v = float(val)
                        if v > 0: return 'color: #3399ff;'
                        if v < 0: return 'color: #ff9933;'
                        return ''
                    except: return ''
                
                st.markdown(f"**5D Spreads (vs {date2})**")
                sp_df = struct_df[struct_df['Type'] == 'Spread'].drop(columns=['Type'])
                styled_sp = sp_df.style.format({'Current': '{:.4f}', 'Chg': '{:+.4f}'}).map(color_sig, subset=['F1', 'F2', 'F3', 'Signal']).map(color_chg, subset=['Chg'])
                st.dataframe(styled_sp, use_container_width=True, height=400)
            
                st.markdown(f"**5D Butterflies (vs {date2})**")
                fly_df = struct_df[struct_df['Type'] == 'Fly'].drop(columns=['Type'])
                styled_fly = fly_df.style.format({'Current': '{:.4f}', 'Chg': '{:+.4f}'}).map(color_sig, subset=['F1', 'F2', 'F3', 'Signal']).map(color_chg, subset=['Chg'])
                st.dataframe(styled_fly, use_container_width=True, height=400)


    with col_smile:
        col_x, col_n = st.columns(2)
        with col_x:
            x_axis = st.selectbox("Smile X-Axis", ["Label 1 (Delta)", "Strike 1"])
            x_col = 'Label 1' if x_axis == "Label 1 (Delta)" else x_axis
        with col_n:
            num_days = st.selectbox("Days to Compare", list(range(2, 11)), index=0)
        
        st.subheader("IV Smile Comparison")
        fig = go.Figure()
    
        # Color palette for up to 10 days
        smile_colors = [
            '#00ff00', '#ff3333', '#3399ff', '#ff9933', '#cc33ff',
            '#33ffff', '#ff33cc', '#ffff33', '#99ff33', '#ffffff'
        ]
    
        date_idx = available_dates.index(date1) if date1 in available_dates else -1
    
        if date_idx != -1:
            plot_dates = []
            y_mins = []
            y_maxs = []
        
            for i in range(num_days):
                if date_idx - i >= 0:
                    plot_dates.append(available_dates[date_idx - i])
                
            for i, p_date in enumerate(plot_dates):
                d_df = df_ladder[df_ladder['Date'].dt.date == p_date]
                if not d_df.empty:
                    d_df = d_df.sort_values('Strike 1')
                    y_mins.append(d_df['IV 1'].min())
                    y_maxs.append(d_df['IV 1'].max())
                
                    name_label = f"Day {i} ({p_date})" if i > 0 else f"Current ({p_date})"
                    fig.add_trace(go.Scatter(
                        x=d_df[x_col], y=d_df['IV 1'], 
                        mode='lines+markers', 
                        name=name_label,
                        line=dict(color=smile_colors[i])
                    ))
                
                    if i == 0 and x_col == 'Label 1':
                        fig.update_xaxes(categoryorder='array', categoryarray=d_df[x_col].tolist())
        
            if y_mins and y_maxs:
                min_iv = min(y_mins)
                max_iv = max(y_maxs)
                padding = (max_iv - min_iv) * 0.05
                fig.update_yaxes(range=[min_iv - padding, max_iv + padding], tickformat='.2%')
            
            fig.update_layout(
                height=650, 
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
        
        st.plotly_chart(fig, use_container_width=True)


    st.divider()
    # --- MACRO & SMILE DATA TABLES ---
    st.header("Macro Data")
    st.write(f"Data for **{date1}**")
    macro_data = []
    for ind, df in market_sheets.items():
        day_row = df[df['Date'].dt.date == date1]
        if not day_row.empty:
            row = day_row.iloc[0]
            macro_data.append({'Indicator': ind, 'Value': row.get('Value', np.nan), 'Z-score': row.get('Z-score', np.nan), 'Percentile': row.get('Percentile', np.nan)})
        else:
            past_df = df[df['Date'].dt.date <= date1]
            if not past_df.empty:
                row = past_df.iloc[-1]
                macro_data.append({'Indicator': f"{ind} (Last: {row['Date'].date()})", 'Value': row.get('Value', np.nan), 'Z-score': row.get('Z-score', np.nan), 'Percentile': row.get('Percentile', np.nan)})
            else:
                macro_data.append({'Indicator': ind, 'Value': np.nan, 'Z-score': np.nan, 'Percentile': np.nan})

    if macro_data:
        macro_df = pd.DataFrame(macro_data).set_index('Indicator')
        st.dataframe(macro_df.style.format({'Value': "{:.4f}", 'Z-score': "{:.2f}", 'Percentile': "{:.2%}"}), use_container_width=True, height=400)

    st.header("Smile Data")
    st.write(f"IV context for **{date1}**")

    smile_df = df_ladder.pivot_table(index='Date', columns='Label 1', values='IV 1')
    delta_order = [
        '5 delta put', '10 delta put', '15 delta put', '20 delta put', '25 delta put',
        '30 delta put', '35 delta put', '40 delta put', '45 delta put', 'ATM',
        '45 delta call', '40 delta call', '35 delta call', '30 delta call', '25 delta call',
        '20 delta call', '15 delta call', '10 delta call', '5 delta call'
    ]

    d1_dt = pd.to_datetime(date1)
    smile_data_list = []

    if d1_dt in smile_df.index:
        for d_label in delta_order:
            if d_label in smile_df.columns:
                val = smile_df.loc[d1_dt, d_label]
                hist = smile_df[d_label].dropna()

                if len(hist) > 2 and pd.notna(val):
                    mean = hist.mean()
                    std = hist.std()
                    z = (val - mean) / std if std > 0 else 0
                    pctile = (hist < val).mean()
                    smile_data_list.append({'Delta': d_label, 'IV': val, 'Z-score': z, 'Percentile': pctile})
                else:
                    smile_data_list.append({'Delta': d_label, 'IV': val, 'Z-score': np.nan, 'Percentile': np.nan})

        if smile_data_list:
            s_table = pd.DataFrame(smile_data_list).set_index('Delta')
            st.dataframe(s_table.style.format({'IV': "{:.2%}", 'Z-score': "{:.2f}", 'Percentile': "{:.2%}"}), use_container_width=True, height=400)

    st.divider()
    # --- SECTION 4: TIME SERIES ---
    st.subheader("Delta Time Series Comparison")
    tsc1, tsc2 = st.columns([1, 3])
    with tsc1:
        num_ts = st.radio("Number of Overlapped Deltas", [1, 2, 3], horizontal=True)
        ts_options = [
            '5 delta put', '10 delta put', '15 delta put', '20 delta put', '25 delta put',
            '30 delta put', '35 delta put', '40 delta put', '45 delta put', 'ATM',
            '45 delta call', '40 delta call', '35 delta call', '30 delta call', '25 delta call',
            '20 delta call', '15 delta call', '10 delta call', '5 delta call'
        ]
        sel_ts_deltas = []
        for i in range(num_ts):
            default_idx = i * 4 if i * 4 < len(ts_options) else 0
            sel_ts_deltas.append(st.selectbox(f"Delta {i+1}", ts_options, index=default_idx, key=f"ts_sel_{i}"))

    with tsc2:
        ts_fig = go.Figure()
        full_smile_df = df_ladder.pivot_table(index='Date', columns='Label 1', values='IV 1')

        # Filter by window explicitly
        ts_window_df = full_smile_df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]

        colors_ts = ['#3399ff', '#ff9933', '#00ff00']
        for i, delta_lbl in enumerate(sel_ts_deltas):
            if delta_lbl in ts_window_df.columns:
                ts_fig.add_trace(go.Scatter(
                    x=ts_window_df.index, y=ts_window_df[delta_lbl],
                    mode='lines', name=delta_lbl, line=dict(color=colors_ts[i], width=2)
                ))

        ts_fig.update_layout(
            height=350, 
            margin=dict(l=0, r=0, t=10, b=0), 
            hovermode='x unified', 
            yaxis_tickformat='.2%',
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(ts_fig, use_container_width=True)

    st.divider()
    # --- SECTION 3: VOL STATS ---
    st.subheader("Day Comparison & Changes")
    stats1 = get_vol_stats(df_ladder, pd.to_datetime(date1))
    stats2 = get_vol_stats(df_ladder, pd.to_datetime(date2))
    
    if stats1 and stats2:
        comp_df = pd.DataFrame([stats1, stats2], index=[str(date1), str(date2)]).T
        comp_df['Abs Change'] = comp_df[str(date1)] - comp_df[str(date2)]
        comp_df['% Change'] = (comp_df['Abs Change'] / comp_df[str(date2)].replace(0, np.nan)) * 100
        
        styled_comp = comp_df.style.format({
            str(date1): "{:.4f}",
            str(date2): "{:.4f}",
            'Abs Change': "{:.4f}",
            '% Change': "{:.2f}%"
        })
        st.dataframe(styled_comp, use_container_width=True)
        
    st.subheader("Historical Metrics in Window (Consolidated %)")
    window_dates = [d for d in available_dates if start_date <= d <= end_date]
    window_data = []
    for d in window_dates:
        stats = get_vol_stats(df_ladder, pd.to_datetime(d))
        if stats:
            stats['Date'] = d
            window_data.append(stats)
            
    if window_data:
        win_df = pd.DataFrame(window_data).set_index('Date')
        
        # Convert index to datetime for robust joining
        win_df.index = pd.to_datetime(win_df.index)
        
        # Requested macro indicators
        macro_cols = [
            'SPX 1D', 'SPX 5D', 'SPX 20D', '20D RV', 'IV - RV',
            'VIX', 'Spot-Vol', 'VVIX', 'VIX9D', 'VIX3M', 'SKEW',
            'US 2Y', 'US 10Y', 'MOVE', 'DXY', 'WTI'
        ]
        
        # Build an aligned, forward-filled macro master to prevent NaNs on holidays
        macro_master = pd.DataFrame()
        for m in macro_cols:
            if m in market_sheets:
                temp = market_sheets[m][['Date', 'Value']].rename(columns={'Value': m}).set_index('Date')
                macro_master = pd.concat([macro_master, temp], axis=1)
        
        macro_master = macro_master.ffill()
        win_df = win_df.join(macro_master)
        
        # Full historical master for z-score and %ile
        import mr_engine
        import importlib
        importlib.reload(mr_engine)
        
        full_iv_ts = mr_engine.build_smile_timeseries(df_ladder)
        
        # Failsafe: force rename just in case the old mr_engine is still stuck in memory
        failsafe_rename = {
            '25D Put Skew': 'Put Skew (25P - ATM)',
            '25D Call Skew': 'Call Skew (ATM - 25C)',
            '25D RR': '25D RR (25P - 25C)',
            'Put Wing Slope': 'Put Wing (10P - 25P)',
            'Call Wing Slope': 'Call Wing (10C - 25C)'
        }
        full_iv_ts = full_iv_ts.rename(columns=failsafe_rename)
        
        full_master = full_iv_ts.join(macro_master)
        
        if regimes_df is not None and 'Regime_Label' in regimes_df.columns:
            win_df = win_df.join(regimes_df[['Regime_Label']])
        
        # Columns to multiply by 100 and append %
        mult_100_pct = [
            'ATM IV', '25D RR (25P - 25C)', 'Put Skew (25P - ATM)', 'Call Skew (ATM - 25C)', 
            'Put Wing (10P - 25P)', 'Call Wing (10C - 25C)', 'Put Curvature', 'Call Curvature',
            'SPX 1D', 'SPX 5D', 'SPX 20D', '20D RV', 'IV - RV'
        ]
        
        # Columns to just append %
        direct_pct = ['US 2Y', 'US 10Y']
        
        def format_with_context(row, col_name):
            val = row[col_name]
            if pd.isna(val):
                return ""
            
            date = row.name
            if col_name in full_master.columns:
                hist = full_master[col_name].dropna()
                if len(hist) > 2:
                    mean = hist.mean()
                    std = hist.std()
                    z = (val - mean) / std if std > 0 else 0
                    pctile = (hist < val).mean() * 100
                    z_str = f"{z:+.1f}"
                    p_str = f"{pctile:.0f}th"
                else:
                    z_str = "N/A"
                    p_str = "N/A"
            else:
                z_str = "N/A"
                p_str = "N/A"
            
            # Format base value
            if col_name in mult_100_pct:
                base_str = f"{val * 100:.2f}%"
            elif col_name in direct_pct:
                base_str = f"{val:.2f}%"
            else:
                base_str = f"{val:.2f}"
                
            return f"{base_str} (Z: {z_str}, {p_str})"
            
        formatted_win_df = pd.DataFrame(index=win_df.index)
        for c in win_df.columns:
            if c == 'Regime_Label':
                formatted_win_df[c] = win_df[c]
            else:
                formatted_win_df[c] = win_df.apply(lambda r: format_with_context(r, c), axis=1)
        
        st.dataframe(formatted_win_df, use_container_width=True)
        
        st.subheader("Filter by Volatility Regime")
        st.write("Select a regime to view all historical dates matching that specific environment.")
        
        # Use regimes_df to get all dates for a given regime
        unique_regimes = ["LOW-VOL CARRY", "NORMAL / BALANCED", "VOL TENSION / TRANSITION", "DEFENSIVE STRESS", "SHOCK / CRISIS"]
        selected_regime = st.selectbox("Select Regime to Filter", unique_regimes, index=1)
        
        if regimes_df is not None and not regimes_df.empty:
            regime_dates = regimes_df[regimes_df['Regime_Label'] == selected_regime].index.tolist()
            
            # Fetch full historical metrics
            # To be fast, we compute only for matching dates
            if regime_dates:
                r_data = []
                for d in regime_dates:
                    # Limit to available dates in ladder
                    if pd.to_datetime(d).date() in available_dates:
                        stats = get_vol_stats(df_ladder, pd.to_datetime(d))
                        if stats:
                            stats['Date'] = d
                            r_data.append(stats)
                            
                if r_data:
                    r_df = pd.DataFrame(r_data).set_index('Date')
                    r_df.index = pd.to_datetime(r_df.index)
                    
                    # Join macro data
                    r_df = r_df.join(macro_master)
                    r_df = r_df.join(regimes_df[['Regime_Label']])
                    
                    formatted_r_df = pd.DataFrame(index=r_df.index)
                    for c in r_df.columns:
                        if c == 'Regime_Label':
                            formatted_r_df[c] = r_df[c]
                        else:
                            formatted_r_df[c] = r_df.apply(lambda r: format_with_context(r, c), axis=1)
                            
                    st.dataframe(formatted_r_df, use_container_width=True)
                    
                    # Add Download Button
                    csv_data = r_df.to_csv(index=True)
                    st.download_button(
                        label=f"Download {selected_regime} Data as CSV",
                        data=csv_data,
                        file_name=f"spx_{selected_regime.replace('/', '_').replace(' ', '_').lower()}_data.csv",
                        mime='text/csv'
                    )
                else:
                    st.info("No options ladder data available for dates in this regime.")
            else:
                st.info(f"No dates found matching the {selected_regime} regime.")
        else:
            st.warning("Regime data not generated.")

with tab_regime:
    import manual_regime
    import importlib
    importlib.reload(manual_regime)
    
    st.header(f"SPX Volatility Regime (Manual Analysis): {date1}")
    st.write("Deterministic, rule-based regime engine using trailing 252-day percentiles. Strictly ZERO look-ahead bias.")
    
    reg_out = manual_regime.get_manual_regime(date1, market_sheets, df_ladder)
    
    if reg_out is not None:
        c1, c2 = st.columns([3, 2])
        
        with c1:
            st.subheader("5-Factor Regime Indicators")
            st.dataframe(reg_out['Table'].style.format({'Current Value': "{:.4f}", 'Percentile': "{:.1f}th"}), use_container_width=True)
            
            st.subheader("Regime State")
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Today's Regime", reg_out['Regime'])
            rc2.metric("Stress Score", f"{reg_out['Stress Score']:.1f}")
            rc3.metric("Stress Count (>=75th)", reg_out['Stress Count'])
            rc4.metric("Extreme Count (>=90th)", reg_out['Extreme Count'])
            
            mc1, mc2 = st.columns(2)
            mc1.metric("SPX 1D (Modifier)", f"{reg_out['SPX 1D']*100:.2f}%" if pd.notna(reg_out['SPX 1D']) else "N/A")
            mc2.metric("SPX 5D (Modifier)", f"{reg_out['SPX 5D']*100:.2f}%" if pd.notna(reg_out['SPX 5D']) else "N/A")
            
            st.info(reg_out['Text'])
            
        with c2:
            st.subheader("Regime Timeline ([-10, +10] Days)")
            hist_df = manual_regime.get_historical_regimes(date1, market_sheets, df_ladder)
            if not hist_df.empty:
                import plotly.graph_objects as go
                fig_r = go.Figure()
                
                # Colors based on regime
                color_map = {
                    1: '#00ff00', 
                    2: '#3399ff',
                    3: '#ff9933',
                    4: '#ff3333',
                    5: '#cc0000'
                }
                
                fig_r.add_trace(go.Scatter(
                    x=hist_df['Date'], 
                    y=hist_df['R_Val'], 
                    mode='lines+markers', 
                    line_shape='hv',
                    line=dict(color='#ff9933', width=2),
                    marker=dict(
                        size=[12 if d == pd.to_datetime(date1) else 6 for d in hist_df['Date']],
                        color=[color_map.get(v, '#ffffff') for v in hist_df['R_Val']]
                    )
                ))
                
                tick_text = ['Low-Vol Carry', 'Normal / Balanced', 'Vol Tension', 'Defensive Stress', 'Shock / Crisis']
                fig_r.update_layout(
                    height=300, 
                    margin=dict(l=0, r=0, t=30, b=0), 
                    yaxis=dict(tickvals=[1,2,3,4,5], ticktext=tick_text, range=[0.5, 5.5])
                )
                
                # Highlight current date
                fig_r.add_vline(x=pd.to_datetime(date1), line_dash="dash", line_color="white")
                
                st.plotly_chart(fig_r, use_container_width=True)
                if len(hist_df) < 21:
                    st.caption("Note: Showing max available dates. Future or past dates missing from dataset are natively excluded.")
    
    st.divider()
    st.subheader("Interpretation for 60DTE Smile")
    c_reg = reg_out['Regime'] if reg_out else 'N/A'
    
    if c_reg == "LOW-VOL CARRY":
        st.success("**LOW-VOL CARRY:** Small spot moves. Relatively stable ATM IV. Slower skew movement. Local smile dislocations can persist longer.")
    elif c_reg == "NORMAL / BALANCED":
        st.info("**NORMAL / BALANCED:** Normal spot-vol relationship. Moderate smile adjustments. Standard skew/curvature behavior.")
    elif c_reg == "VOL TENSION / TRANSITION":
        st.warning("**VOL TENSION / TRANSITION:** Stronger spot-vol sensitivity. ATM IV becomes more responsive. Skew begins repricing faster. Historical low-vol relationships become less reliable.")
    elif c_reg == "DEFENSIVE STRESS":
        st.error("**DEFENSIVE STRESS:** Strong negative spot-vol relationship. Put wing becomes more reactive. Skew and curvature can reprice aggressively. Vanna effects become more important.")
    elif c_reg == "SHOCK / CRISIS":
        st.error("**SHOCK / CRISIS:** Nonlinear smile behavior. Large ATM and wing repricing. Skew can gap. Normal historical relationships can temporarily break.")




with tab_mr:
    st.header("Mean Reversion Classification (3-Test Suite)")
    st.markdown("Evaluates whether structures are statistically mean-reverting using a robust 3-test suite (ADF, KPSS, Half-Life).")
    
    struct_type = st.radio("Select Structure Type", ["Spreads (5D Gap)", "Butterflies (5D Gap)"], horizontal=True)
    
    if st.button("Run Classification", type="primary"):
        with st.spinner("Running statistical tests..."):
            iv_pure = df_ladder.pivot_table(index='Date', columns='Label 1', values='IV 1').dropna(axis=1, how='all')
            iv_pure.index = pd.to_datetime(iv_pure.index).date
            
            ordered_cols = [
                '5 delta put', '10 delta put', '15 delta put', '20 delta put', '25 delta put',
                '30 delta put', '35 delta put', '40 delta put', '45 delta put', 'ATM',
                '45 delta call', '40 delta call', '35 delta call', '30 delta call', '25 delta call',
                '20 delta call', '15 delta call', '10 delta call', '5 delta call'
            ]
            cols = [c for c in ordered_cols if c in iv_pure.columns]
            
            from statsmodels.tsa.stattools import adfuller, kpss
            import numpy as np
            import warnings
            
            results = []
            
            is_spread = (struct_type == "Spreads (5D Gap)")
            loop_len = len(cols) - 1 if is_spread else len(cols) - 2
            
            for i in range(loop_len):
                if is_spread:
                    name = f"{cols[i]} vs {cols[i+1]}"
                    series = (iv_pure[cols[i]] - iv_pure[cols[i+1]]).dropna()
                else:
                    name = f"{cols[i]} / {cols[i+1]} / {cols[i+2]} Fly"
                    series = (iv_pure[cols[i]] - 2*iv_pure[cols[i+1]] + iv_pure[cols[i+2]]).dropna()
                
                if len(series) > 50:
                    adf_result = adfuller(series, autolag='AIC')
                    adf_pass = adf_result[1] < 0.05
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        kpss_result = kpss(series, regression='c', nlags="auto")
                    kpss_pass = kpss_result[1] > 0.05
                    
                    y = series.values
                    dy = y[1:] - y[:-1]
                    slope, _ = np.polyfit(y[:-1], dy, 1)
                    
                    if slope >= 0:
                        hl = np.inf
                        hl_pass = False
                    else:
                        hl = -np.log(2) / slope
                        hl_pass = (0 < hl < 30)
                        
                    passes = sum([adf_pass, kpss_pass, hl_pass])
                    verdict = "Strongly MR" if passes == 3 else ("Moderately MR" if passes == 2 else "Not MR")
                        
                    results.append({
                        'Structure': name,
                        'ADF Pass': 'Yes' if adf_pass else 'No',
                        'KPSS Pass': 'Yes' if kpss_pass else 'No',
                        'Half-Life (Days)': f"{hl:.1f}" if hl != np.inf else "Inf",
                        'Final Verdict': verdict
                    })
                    
            res_df = pd.DataFrame(results)
            def color_mr(val):
                if val == 'Strongly MR': return 'color: #00ff00; font-weight: bold;'
                if val == 'Moderately MR': return 'color: #ffa500; font-weight: bold;'
                return 'color: #ff0000;'
            st.dataframe(res_df.style.map(color_mr, subset=['Final Verdict']), use_container_width=True)

with tab_trade:
    st.header("Trade Backtester (3-Factor Master Engine)")
    st.markdown("Takes trades ONLY when all 3 factors (**F1, F2, F3**) strongly agree the spread is mispriced. It calculates rolling 252-day out-of-sample regressions. When consensus exceeds the threshold, it enters and holds until F1 normalizes.")
    
    struct_type_trade = st.radio("Select Structure Type for Backtest", ["Spreads (5D Gap)", "Butterflies (5D Gap)"], horizontal=True, key='tt_radio')
    c1, c2 = st.columns(2)
    exit_z = c1.number_input("Exit Z-Score Threshold", value=0.0, step=0.1, min_value=0.0, max_value=2.0, help="Target F1 Z-score to take profit.")
    max_hold_days = c2.number_input("Time Stop (Max Hold Days)", value=10, step=1, min_value=1, max_value=60, help="Forces the trade to close after this many days.")
    
    if st.button("Run Backtest on Qualified Structures", type="primary"):
        with st.spinner("Classifying structures and running backtest..."):
            iv_pure = df_ladder.pivot_table(index='Date', columns='Label 1', values='IV 1').dropna(axis=1, how='all')
            iv_pure.index = pd.to_datetime(iv_pure.index).date
            
            ordered_cols = [
                '5 delta put', '10 delta put', '15 delta put', '20 delta put', '25 delta put',
                '30 delta put', '35 delta put', '40 delta put', '45 delta put', 'ATM',
                '45 delta call', '40 delta call', '35 delta call', '30 delta call', '25 delta call',
                '20 delta call', '15 delta call', '10 delta call', '5 delta call'
            ]
            cols = [c for c in ordered_cols if c in iv_pure.columns]
            
            from statsmodels.tsa.stattools import adfuller, kpss
            import numpy as np
            import warnings
            
            spot_series = df_ladder.groupby(df_ladder['Date'].dt.date)['Underlying'].first()
            if 'ATM' in iv_pure.columns:
                macro_df = pd.DataFrame({
                    'ATM_IV': iv_pure['ATM'],
                    'd_ATM': iv_pure['ATM'].diff(),
                    'Spot_Ret': spot_series.pct_change()
                }).dropna()
            else: macro_df = pd.DataFrame()
            
            import statsmodels.api as sm
            from statsmodels.regression.rolling import RollingOLS
            
            is_spread = (struct_type_trade == "Spreads (5D Gap)")
            loop_len = len(cols) - 1 if is_spread else len(cols) - 2
            
            trade_results = []
            
            for i in range(loop_len):
                if is_spread:
                    name = f"{cols[i]} vs {cols[i+1]}"
                    series = (iv_pure[cols[i]] - iv_pure[cols[i+1]]).dropna()
                    sp_dict = {}
                    for j in range(max(0, i-5), i):
                        sp_dict[f'L_{i-j}'] = iv_pure[cols[j]] - iv_pure[cols[j+1]]
                    for j in range(i+1, min(len(cols)-1, i+6)):
                        sp_dict[f'R_{j-i}'] = iv_pure[cols[j]] - iv_pure[cols[j+1]]
                    spatial_df = pd.DataFrame(sp_dict).dropna()
                else:
                    name = f"{cols[i]} / {cols[i+1]} / {cols[i+2]} Fly"
                    series = (iv_pure[cols[i]] - 2*iv_pure[cols[i+1]] + iv_pure[cols[i+2]]).dropna()
                    sp_dict = {}
                    for j in range(max(0, i-5), i):
                        sp_dict[f'L_{i-j}'] = iv_pure[cols[j]] - 2*iv_pure[cols[j+1]] + iv_pure[cols[j+2]]
                    for j in range(i+1, min(len(cols)-2, i+6)):
                        sp_dict[f'R_{j-i}'] = iv_pure[cols[j]] - 2*iv_pure[cols[j+1]] + iv_pure[cols[j+2]]
                    spatial_df = pd.DataFrame(sp_dict).dropna()
                
                if len(series) > 252 and not macro_df.empty:
                    c_idx = series.index.intersection(macro_df.index)
                    if not spatial_df.empty: c_idx = c_idx.intersection(spatial_df.index)
                    s = series.loc[c_idx]
                    m = macro_df.loc[c_idx]
                    sp = spatial_df.loc[c_idx] if not spatial_df.empty else pd.DataFrame()
                    
                    # F1: Time-Series Z-Score
                    r_mean = s.rolling(252, min_periods=50).mean()
                    r_std = s.rolling(252, min_periods=50).std()
                    z1 = (s - r_mean) / r_std
                    
                    # F2: Spatial Arb (Neighbors)
                    if not sp.empty:
                        try:
                            X2 = sm.add_constant(sp)
                            model2 = RollingOLS(s, X2, window=252, min_nobs=50).fit()
                            pred2 = (model2.params * X2).sum(axis=1)
                            resid2 = s - pred2
                            z2 = resid2 / resid2.rolling(252, min_periods=50).std()
                        except:
                            z2 = pd.Series(0, index=s.index)
                    else:
                        z2 = pd.Series(0, index=s.index)
                        
                    # F3: Macro Fair Value Arb
                    X3 = sm.add_constant(m[['ATM_IV', 'd_ATM', 'Spot_Ret']])
                    try:
                        model3 = RollingOLS(s, X3, window=252, min_nobs=50).fit()
                        pred3 = (model3.params * X3).sum(axis=1)
                        resid3 = s - pred3
                        z3 = resid3 / resid3.rolling(252, min_periods=50).std()
                    except:
                        z3 = pd.Series(0, index=s.index)
                        
                    in_trade = 0
                    entry_val = 0
                    trades = 0
                    hits = 0
                    hold_days = []
                    current_hold = 0
                    
                    for j in range(len(s)):
                        curr_v = s.iloc[j]
                        c_z1, c_z2, c_z3 = z1.iloc[j], z2.iloc[j], z3.iloc[j]
                        
                        if pd.isna(c_z1) or pd.isna(c_z2) or pd.isna(c_z3): continue
                        
                        # 3-FACTOR MASTER SIGNAL: Enter only if ALL THREE strongly agree
                        is_short_signal = (c_z1 > f1_thresh) and (c_z2 > f2_thresh) and (c_z3 > f3_thresh)
                        is_long_signal = (c_z1 < -f1_thresh) and (c_z2 < -f2_thresh) and (c_z3 < -f3_thresh)
                        
                        if in_trade == 0:
                            if is_short_signal:
                                in_trade = -1
                                entry_val = curr_v
                                current_hold = 0
                            elif is_long_signal:
                                in_trade = 1
                                entry_val = curr_v
                                current_hold = 0
                        else:
                            current_hold += 1
                            # Exit logic purely based on F1 reverting (the primary mean)
                            if (in_trade == 1 and c_z1 >= -exit_z) or (in_trade == -1 and c_z1 <= exit_z) or current_hold >= max_hold_days:
                                trades += 1
                                hold_days.append(current_hold)
                                pnl = (curr_v - entry_val) if in_trade == 1 else (entry_val - curr_v)
                                if pnl > 0: hits += 1
                                in_trade = 0
                                
                    hit_rate = hits / trades if trades > 0 else 0
                    avg_hold = np.mean(hold_days) if hold_days else 0
                    
                    trade_results.append({
                        'Structure': name,
                        'Total Trades (3-Factor)': trades,
                        'Hits': hits,
                        'Hit Rate': f"{hit_rate*100:.1f}%",
                        'Avg Hold (Days)': f"{avg_hold:.1f}"
                    })
                        
            if trade_results:
                st.success("Backtest Complete on Qualified Structures!")
                res_df = pd.DataFrame(trade_results)
                
                # Visualizing Backtest
                res_df['Hit %'] = res_df['Hit Rate'].str.rstrip('%').astype(float)
                
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Bar(
                    x=res_df['Structure'], 
                    y=res_df['Hit %'],
                    marker=dict(
                        color=res_df['Hit %'],
                        colorscale='RdYlGn',
                        showscale=True,
                        line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
                    ),
                    text=res_df['Hit Rate'],
                    textposition='auto',
                    name='Hit Rate'
                ))
                fig_bt.update_layout(
                    title="Hit Rate by Structure (3-Factor Consensus)", 
                    template="plotly_dark", 
                    plot_bgcolor="rgba(0,0,0,0)", 
                    paper_bgcolor="rgba(0,0,0,0)", 
                    yaxis=dict(title="Hit Rate (%)", range=[0, 100]),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_bt, use_container_width=True)
                
                st.dataframe(res_df.drop(columns=['Hit %']), use_container_width=True)
            else:
                st.warning("No structures qualified as MR.")

