import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
import warnings
import pickle
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="EA Pairs Trading - Candidate Finder",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1e2130; padding: 15px; border-radius: 10px;}
    h1, h2, h3 {color: #ffffff;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SISTEMA DE CACHE PERSISTENTE
# ============================================================================

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

CACHE_FILE = CACHE_DIR / "asset_data.pkl"
METADATA_FILE = CACHE_DIR / "metadata.pkl"

def save_data_to_cache(data_dict, metadata):
    """Guarda datos y metadata en disco"""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(data_dict, f)
        with open(METADATA_FILE, 'wb') as f:
            pickle.dump(metadata, f)
        return True
    except Exception as e:
        st.error(f"Error guardando cache: {str(e)}")
        return False

def load_data_from_cache():
    """Carga datos y metadata desde disco"""
    try:
        if CACHE_FILE.exists() and METADATA_FILE.exists():
            with open(CACHE_FILE, 'rb') as f:
                data_dict = pickle.load(f)
            with open(METADATA_FILE, 'rb') as f:
                metadata = pickle.load(f)
            return data_dict, metadata
        return None, None
    except Exception as e:
        st.error(f"Error cargando cache: {str(e)}")
        return None, None

def get_cache_info():
    """Obtiene información del cache"""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, 'rb') as f:
                metadata = pickle.load(f)
            return metadata
        except:
            return None
    return None

# ============================================================================
# CONFIGURACIÓN DE ACTIVOS
# ============================================================================
ASSETS = {
    # ========== ÍNDICES GLOBALES ==========
    'us500': {'label': 'US SPX 500 (S&P 500)', 'symbol': '^GSPC', 'category': 'Indices'},
    'us30': {'label': 'US Wall Street 30 (Dow Jones)', 'symbol': '^DJI', 'category': 'Indices'},
    'ustec': {'label': 'US Tech 100 (NASDAQ)', 'symbol': '^IXIC', 'category': 'Indices'},
    'russell': {'label': 'Russell 2000', 'symbol': '^RUT', 'category': 'Indices'},
    'uk100': {'label': 'UK 100 (FTSE)', 'symbol': '^FTSE', 'category': 'Indices'},
    'de30': {'label': 'Germany 30 (DAX)', 'symbol': '^GDAXI', 'category': 'Indices'},
    'fr40': {'label': 'France 40 (CAC 40)', 'symbol': '^FCHI', 'category': 'Indices'},
    'stoxx50': {'label': 'EU Stocks 50 (Euro Stoxx)', 'symbol': '^STOXX50E', 'category': 'Indices'},
    'jp225': {'label': 'Japan 225 (Nikkei)', 'symbol': '^N225', 'category': 'Indices'},
    'hk50': {'label': 'Hong Kong 50 (Hang Seng)', 'symbol': '^HSI', 'category': 'Indices'},
    'aus200': {'label': 'Australia 200 (ASX)', 'symbol': '^AXJO', 'category': 'Indices'},
    'in50': {'label': 'India 50 (Nifty)', 'symbol': '^NSEI', 'category': 'Indices'},
    
    # ========== DIVISAS ==========
    'dxy': {'label': 'US Dollar Index', 'symbol': 'DX-Y.NYB', 'category': 'Forex'},
    'eurusd': {'label': 'EUR/USD', 'symbol': 'EURUSD=X', 'category': 'Forex'},
    'gbpusd': {'label': 'GBP/USD', 'symbol': 'GBPUSD=X', 'category': 'Forex'},
    'usdjpy': {'label': 'USD/JPY', 'symbol': 'JPYUSD=X', 'category': 'Forex'},
    'audusd': {'label': 'AUD/USD', 'symbol': 'AUDUSD=X', 'category': 'Forex'},
    'usdcad': {'label': 'USD/CAD', 'symbol': 'CADUSD=X', 'category': 'Forex'},
    'usdchf': {'label': 'USD/CHF', 'symbol': 'CHFUSD=X', 'category': 'Forex'},
    'nzdusd': {'label': 'NZD/USD', 'symbol': 'NZDUSD=X', 'category': 'Forex'},
    
    # ========== METALES Y COMMODITIES ==========
    'gold': {'label': 'Gold (GC)', 'symbol': 'GC=F', 'category': 'Commodities'},
    'silver': {'label': 'Silver (SI)', 'symbol': 'SI=F', 'category': 'Commodities'},
    'copper': {'label': 'Copper (HG)', 'symbol': 'HG=F', 'category': 'Commodities'},
    'platinum': {'label': 'Platinum (PL)', 'symbol': 'PL=F', 'category': 'Commodities'},
    'oil': {'label': 'Crude Oil WTI', 'symbol': 'CL=F', 'category': 'Commodities'},
    'brent': {'label': 'Brent Crude Oil', 'symbol': 'BZ=F', 'category': 'Commodities'},
    'natgas': {'label': 'Natural Gas', 'symbol': 'NG=F', 'category': 'Commodities'},
    'corn': {'label': 'Corn', 'symbol': 'ZC=F', 'category': 'Commodities'},
    'wheat': {'label': 'Wheat', 'symbol': 'ZW=F', 'category': 'Commodities'},
    'soybeans': {'label': 'Soybeans', 'symbol': 'ZS=F', 'category': 'Commodities'},
    'sugar': {'label': 'Sugar', 'symbol': 'SB=F', 'category': 'Commodities'},
    'coffee': {'label': 'Coffee', 'symbol': 'KC=F', 'category': 'Commodities'},
    'cotton': {'label': 'Cotton', 'symbol': 'CT=F', 'category': 'Commodities'},
    
    # ========== CRIPTOMONEDAS ==========
    'btc': {'label': 'Bitcoin', 'symbol': 'BTC-USD', 'category': 'Crypto'},
    'eth': {'label': 'Ethereum', 'symbol': 'ETH-USD', 'category': 'Crypto'},
    'bnb': {'label': 'Binance Coin', 'symbol': 'BNB-USD', 'category': 'Crypto'},
    'xrp': {'label': 'Ripple', 'symbol': 'XRP-USD', 'category': 'Crypto'},
    'sol': {'label': 'Solana', 'symbol': 'SOL-USD', 'category': 'Crypto'},
    
    # ========== VOLATILIDAD ==========
    'vix': {'label': 'VIX (S&P 500 Volatility)', 'symbol': '^VIX', 'category': 'Volatility'},
}

# ============================================================================
# FUNCIONES DE DESCARGA
# ============================================================================

def fetch_asset_data(symbol, start_date='2020-01-01', end_date=None):
    """Descarga datos históricos de un activo"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        return prices.dropna()
    except Exception as e:
        return None

def download_all_assets(delay=3, start_date='2020-01-01'):
    """Descarga TODOS los activos con delay"""
    all_data = {}
    failed = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(ASSETS)
    
    for idx, (key, asset_info) in enumerate(ASSETS.items()):
        symbol = asset_info['symbol']
        
        status_text.text(f"⏳ Descargando {asset_info['label']} ({idx+1}/{total})...")
        
        data = fetch_asset_data(symbol, start_date)
        
        if data is not None and len(data) > 0:
            all_data[key] = data
            status_text.text(f"✅ {asset_info['label']} - {len(data)} días")
        else:
            failed.append(key)
            status_text.text(f"❌ {asset_info['label']} - Sin datos")
        
        progress_bar.progress((idx + 1) / total)
        
        if idx < total - 1:
            time.sleep(delay)
    
    progress_bar.empty()
    status_text.empty()
    
    metadata = {
        'last_update': datetime.now(),
        'total_assets': len(all_data),
        'failed_assets': failed,
        'date_range': {
            'start': start_date,
            'end': datetime.now().strftime('%Y-%m-%d')
        }
    }
    
    return all_data, metadata

def update_existing_data(existing_data, existing_metadata, delay=2):
    """Actualiza datos existentes con información nueva"""
    updated_data = existing_data.copy()
    
    last_update = existing_metadata['last_update']
    start_date = (last_update + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date >= end_date:
        st.info("✅ Los datos ya están actualizados")
        return existing_data, existing_metadata
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    updated_count = 0
    failed = []
    
    total = len(existing_data)
    
    for idx, (key, old_data) in enumerate(existing_data.items()):
        asset_info = ASSETS.get(key)
        if not asset_info:
            continue
            
        symbol = asset_info['symbol']
        
        status_text.text(f"🔄 Actualizando {asset_info['label']} ({idx+1}/{total})...")
        
        new_data = fetch_asset_data(symbol, start_date, end_date)
        
        if new_data is not None and len(new_data) > 0:
            combined = pd.concat([old_data, new_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
            
            updated_data[key] = combined
            updated_count += 1
            status_text.text(f"✅ {asset_info['label']} - +{len(new_data)} días")
        else:
            failed.append(key)
            status_text.text(f"⚠️ {asset_info['label']} - Sin actualización")
        
        progress_bar.progress((idx + 1) / total)
        
        if idx < total - 1:
            time.sleep(delay)
    
    progress_bar.empty()
    status_text.empty()
    
    new_metadata = {
        'last_update': datetime.now(),
        'total_assets': len(updated_data),
        'failed_assets': failed,
        'date_range': {
            'start': existing_metadata['date_range']['start'],
            'end': end_date
        },
        'updated_count': updated_count
    }
    
    return updated_data, new_metadata

def merge_asset_data(data_dict):
    """Combina todos los datos en un único DataFrame"""
    if not data_dict:
        return pd.DataFrame()
    
    dfs = []
    for key, data in data_dict.items():
        if isinstance(data, pd.Series):
            df_temp = data.to_frame(name=key)
        elif isinstance(data, pd.DataFrame):
            df_temp = data.copy()
            if len(df_temp.columns) == 1:
                df_temp.columns = [key]
            else:
                df_temp = df_temp.iloc[:, 0].to_frame(name=key)
        else:
            continue
        dfs.append(df_temp)
    
    if not dfs:
        return pd.DataFrame()
    
    df = dfs[0]
    for df_temp in dfs[1:]:
        df = df.join(df_temp, how='inner')
    
    return df

# ============================================================================
# FUNCIONES ESPECÍFICAS PARA EL EA
# ============================================================================

def calculate_log_ratio_spread(prices1, prices2):
    """Calcula spread usando log-ratio (como el EA)"""
    spread = np.log(prices1) - np.log(prices2)
    return spread.dropna()

def calculate_zscore(series, window=100):
    """Calcula Z-Score rolling"""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    zscore = (series - mean) / std
    return zscore.dropna()

def calculate_correlation(prices1, prices2, window=100):
    """Calcula correlación rolling"""
    returns1 = np.log(prices1 / prices1.shift(1))
    returns2 = np.log(prices2 / prices2.shift(1))
    corr = returns1.rolling(window).corr(returns2)
    return corr.dropna()

def calculate_rolling_correlation(df, asset1, asset2, window=30, step=5):
    """Calcula la correlación móvil entre dos activos"""
    correlations = []
    dates = []
    
    prices1 = df[asset1]
    prices2 = df[asset2]
    
    for i in range(window, len(df), step):
        window_data = df.iloc[i-window:i]
        corr = window_data[asset1].corr(window_data[asset2])
        correlations.append(corr)
        dates.append(df.index[i])
    
    return pd.DataFrame({'date': dates, 'correlation': correlations})

def simulate_ea_signals(prices1, prices2, zscore_threshold=3.3, lookback=100, correlation_threshold=0.5):
    """Simula las señales que generaría el EA"""
    spread = calculate_log_ratio_spread(prices1, prices2)
    zscore = calculate_zscore(spread, window=lookback)
    corr = calculate_correlation(prices1, prices2, window=lookback)
    
    common_idx = zscore.index.intersection(corr.index)
    zscore = zscore.loc[common_idx]
    corr = corr.loc[common_idx]
    
    signals = pd.DataFrame(index=common_idx)
    
    valid_corr = corr.abs() >= correlation_threshold
    
    signals['long_signal'] = (zscore > zscore_threshold) & valid_corr
    signals['short_signal'] = (zscore < -zscore_threshold) & valid_corr
    signals['any_signal'] = signals['long_signal'] | signals['short_signal']
    
    total_signals = signals['any_signal'].sum()
    long_signals = signals['long_signal'].sum()
    short_signals = signals['short_signal'].sum()
    
    signal_groups = (signals['any_signal'] != signals['any_signal'].shift()).cumsum()
    signal_durations = signals[signals['any_signal']].groupby(signal_groups).size()
    avg_duration = signal_durations.mean() if len(signal_durations) > 0 else 0
    
    win_count = 0
    total_trades = 0
    
    for idx in signals[signals['any_signal']].index:
        idx_pos = common_idx.get_loc(idx)
        
        if idx_pos + 30 < len(common_idx):
            future_zscore = zscore.iloc[idx_pos:idx_pos+30]
            current_zscore = zscore.loc[idx]
            
            if signals.loc[idx, 'long_signal']:
                if future_zscore.min() < current_zscore * 0.5:
                    win_count += 1
                total_trades += 1
            
            elif signals.loc[idx, 'short_signal']:
                if future_zscore.max() > current_zscore * 0.5:
                    win_count += 1
                total_trades += 1
    
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    days_data = (common_idx[-1] - common_idx[0]).days
    years = days_data / 365.25
    signals_per_year = total_signals / years if years > 0 else 0
    
    stats = {
        'total_signals': int(total_signals),
        'long_signals': int(long_signals),
        'short_signals': int(short_signals),
        'signals_per_year': signals_per_year,
        'avg_duration_days': avg_duration,
        'win_rate': win_rate,
        'long_short_ratio': long_signals / short_signals if short_signals > 0 else np.inf
    }
    
    return signals, stats, zscore, corr, spread

def calculate_hurst_exponent(series, max_lag=100):
    """Calcula el Hurst Exponent"""
    lags = range(2, min(max_lag, len(series)//2))
    tau = [np.std(np.subtract(series[lag:].values, series[:-lag].values)) for lag in lags]
    
    try:
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return np.nan

def adf_test(series):
    """Test de estacionariedad"""
    try:
        result = adfuller(series.dropna())
        return {'adf_stat': result[0], 'pvalue': result[1], 'stationary': result[1] < 0.05}
    except:
        return {'adf_stat': np.nan, 'pvalue': np.nan, 'stationary': False}

def calculate_half_life(spread):
    """Calcula half-life del mean reversion"""
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    spread_lag = spread_lag.dropna()
    spread_diff = spread_diff.dropna()
    
    common_idx = spread_lag.index.intersection(spread_diff.index)
    spread_lag = spread_lag.loc[common_idx]
    spread_diff = spread_diff.loc[common_idx]
    
    if len(spread_lag) < 2:
        return np.nan
    
    model = np.polyfit(spread_lag, spread_diff, 1)
    half_life = -np.log(2) / model[0] if model[0] < 0 else np.nan
    return half_life

def test_cointegration(prices1, prices2):
    """Test de cointegración"""
    try:
        score, pvalue, _ = coint(prices1, prices2)
        return {'score': score, 'pvalue': pvalue, 'cointegrated': pvalue < 0.05}
    except:
        return {'score': np.nan, 'pvalue': np.nan, 'cointegrated': False}

def calculate_correlation_stability(corr_series, window=60):
    """Mide estabilidad de correlación"""
    rolling_std = corr_series.rolling(window).std()
    rolling_mean = corr_series.rolling(window).mean()
    cv = (rolling_std / rolling_mean.abs()).replace([np.inf, -np.inf], np.nan)
    
    return {
        'mean_cv': cv.mean(),
        'current_cv': cv.iloc[-1] if len(cv) > 0 else np.nan,
        'std_corr': corr_series.std()
    }

def detect_regime_changes(corr_series, threshold=0.3):
    """Detecta cambios de régimen"""
    corr_diff = corr_series.diff().abs()
    breakpoints = corr_diff[corr_diff > threshold]
    return breakpoints

def calculate_conditional_correlation(returns1, returns2):
    """Correlación en diferentes condiciones de mercado"""
    mask_positive = (returns1 > 0) & (returns2 > 0)
    corr_positive = returns1[mask_positive].corr(returns2[mask_positive])
    
    mask_negative = (returns1 < 0) & (returns2 < 0)
    corr_negative = returns1[mask_negative].corr(returns2[mask_negative])
    
    vol_threshold = returns1.std() * 2
    mask_crisis = (returns1.abs() > vol_threshold) | (returns2.abs() > vol_threshold)
    corr_crisis = returns1[mask_crisis].corr(returns2[mask_crisis])
    
    return {
        'positive_markets': corr_positive,
        'negative_markets': corr_negative,
        'high_volatility': corr_crisis,
        'normal': returns1.corr(returns2)
    }

def find_best_pairs_for_ea(df, correlation_type='positive', min_correlation=0.5, 
                           min_signals_per_year=10, max_cv=0.4, zscore_threshold=3.3, lookback=100):
    """
    Encuentra los mejores pares para el EA
    correlation_type: 'positive' o 'negative'
    """
    assets = df.columns
    candidates = []
    
    total_pairs = len(assets) * (len(assets) - 1) // 2
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    pair_idx = 0
    
    for i, asset1 in enumerate(assets):
        for asset2 in assets[i+1:]:
            pair_idx += 1
            progress_bar.progress(pair_idx / total_pairs)
            status_text.text(f"Analizando par {pair_idx}/{total_pairs}: {ASSETS[asset1]['label']} vs {ASSETS[asset2]['label']}")
            
            prices1 = df[asset1].dropna()
            prices2 = df[asset2].dropna()
            
            common_idx = prices1.index.intersection(prices2.index)
            if len(common_idx) < 252:
                continue
            
            p1 = prices1.loc[common_idx]
            p2 = prices2.loc[common_idx]
            
            signals, stats, zscore, corr, spread = simulate_ea_signals(
                p1, p2, zscore_threshold, lookback, min_correlation
            )
            
            mean_corr = corr.mean()
            
            # Filtrar según tipo de correlación
            if correlation_type == 'positive':
                if mean_corr < min_correlation:
                    continue
            else:  # negative
                if mean_corr > -min_correlation:
                    continue
            
            if stats['signals_per_year'] < min_signals_per_year:
                continue
            
            adf_result = adf_test(spread)
            hurst = calculate_hurst_exponent(spread.dropna())
            half_life = calculate_half_life(spread)
            coint_result = test_cointegration(p1, p2)
            
            stability = calculate_correlation_stability(corr)
            
            if stability['mean_cv'] > max_cv:
                continue
            
            positive_corr_pct = (corr > 0).sum() / len(corr) * 100
            suggest_invert = positive_corr_pct < 50
            
            returns1 = np.log(p1 / p1.shift(1)).dropna()
            returns2 = np.log(p2 / p2.shift(1)).dropna()
            cond_corr = calculate_conditional_correlation(returns1, returns2)
            
            # SCORE COMPUESTO
            score = 0
            
            if stability['mean_cv'] < 0.2:
                score += 30
            elif stability['mean_cv'] < 0.3:
                score += 20
            else:
                score += 10
            
            if stats['signals_per_year'] > 50:
                score += 25
            elif stats['signals_per_year'] > 30:
                score += 20
            elif stats['signals_per_year'] > 20:
                score += 15
            else:
                score += 10
            
            if hurst < 0.4:
                score += 20
            elif hurst < 0.5:
                score += 15
            else:
                score += 5
            
            if adf_result['stationary']:
                score += 15
            
            if stats['win_rate'] > 60:
                score += 10
            elif stats['win_rate'] > 50:
                score += 7
            else:
                score += 3
            
            candidates.append({
                'asset1': asset1,
                'asset2': asset2,
                'score': score,
                'mean_correlation': mean_corr,
                'corr_stability_cv': stability['mean_cv'],
                'signals_per_year': stats['signals_per_year'],
                'win_rate': stats['win_rate'],
                'hurst': hurst,
                'half_life': half_life,
                'adf_pvalue': adf_result['pvalue'],
                'stationary': adf_result['stationary'],
                'cointegrated': coint_result['cointegrated'],
                'coint_pvalue': coint_result['pvalue'],
                'suggest_invert_trades': suggest_invert,
                'positive_corr_pct': positive_corr_pct,
                'long_signals': stats['long_signals'],
                'short_signals': stats['short_signals'],
                'long_short_ratio': stats['long_short_ratio'],
                'avg_signal_duration': stats['avg_duration_days'],
                'corr_positive_markets': cond_corr['positive_markets'],
                'corr_negative_markets': cond_corr['negative_markets'],
                'corr_high_volatility': cond_corr['high_volatility']
            })
    
    progress_bar.empty()
    status_text.empty()
    
    if len(candidates) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(candidates).sort_values('score', ascending=False)

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def plot_rolling_correlation(corr_df, asset1_name, asset2_name, asset1_color='#10b981', asset2_color='#3b82f6'):
    """Crea un gráfico interactivo de la correlación móvil"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=corr_df['date'],
        y=corr_df['correlation'],
        mode='lines',
        name=f'{asset1_name} vs {asset2_name}',
        line=dict(color='#3b82f6', width=3),
        hovertemplate='%{x}<br>Correlación: %{y:.4f}<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", 
                  annotation_text="Neutral", annotation_position="right")
    fig.add_hline(y=0.5, line_dash="dot", line_color="#10b981", opacity=0.5)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="#ef4444", opacity=0.5)
    
    fig.add_hrect(y0=0.5, y1=1, fillcolor="#10b981", opacity=0.1, line_width=0)
    fig.add_hrect(y0=-1, y1=-0.5, fillcolor="#ef4444", opacity=0.1, line_width=0)
    
    fig.update_layout(
        title=f'Rolling Correlation: {asset1_name} vs {asset2_name}',
        xaxis_title='Fecha',
        yaxis_title='Correlación',
        yaxis=dict(range=[-1, 1]),
        template='plotly_dark',
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    return fig

def plot_price_comparison(df, asset1, asset2, asset1_name, asset2_name):
    """Gráfico comparativo de precios normalizados"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    norm1 = (df[asset1] / df[asset1].iloc[0]) * 100
    norm2 = (df[asset2] / df[asset2].iloc[0]) * 100
    
    fig.add_trace(
        go.Scatter(x=df.index, y=norm1, name=asset1_name, 
                   line=dict(color='#10b981', width=2)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=norm2, name=asset2_name, 
                   line=dict(color='#3b82f6', width=2)),
        secondary_y=False
    )
    
    fig.update_layout(
        title='Comparación de Precios (Normalizado base 100)',
        template='plotly_dark',
        hovermode='x unified',
        height=400
    )
    
    fig.update_yaxes(title_text="Índice (Base 100)", secondary_y=False)
    
    return fig

def plot_conditional_correlation(cond_corr):
    """Gráfico de correlación condicional"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Normal', 'Mercados Alcistas', 'Mercados Bajistas', 'Alta Volatilidad'],
            y=[cond_corr['normal'], cond_corr['positive_markets'], 
               cond_corr['negative_markets'], cond_corr['high_volatility']],
            marker_color=['#3b82f6', '#10b981', '#ef4444', '#f59e0b']
        )
    ])
    
    fig.update_layout(
        title='Correlación en Diferentes Condiciones de Mercado',
        yaxis_title='Correlación',
        template='plotly_dark',
        height=400
    )
    
    return fig

def plot_correlation_stability(stability_df):
    """Visualiza estabilidad de la correlación"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Correlación Media Rolling', 'Estabilidad (Coefficient of Variation)'),
        vertical_spacing=0.15
    )
    
    fig.add_trace(go.Scatter(
        x=stability_df.index,
        y=stability_df['corr_mean'],
        name='Correlación Media',
        line=dict(color='#3b82f6', width=2),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.2)'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=stability_df.index,
        y=stability_df['stability_cv'],
        name='Coef. Variación',
        line=dict(color='#f59e0b', width=2)
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", row=1, col=1)
    
    fig.update_layout(
        height=600,
        template='plotly_dark',
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Correlación", row=1, col=1)
    fig.update_yaxes(title_text="CV", row=2, col=1)
    
    return fig

def plot_regime_changes(corr_series, threshold=0.3):
    """Visualiza puntos de cambio de régimen"""
    breakpoints = detect_regime_changes(corr_series, threshold)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=corr_series.index, y=corr_series,
                             mode='lines', name='Correlation',
                             line=dict(color='#3b82f6', width=2)))
    
    if len(breakpoints) > 0:
        fig.add_trace(go.Scatter(
            x=breakpoints.index,
            y=[corr_series.loc[idx] for idx in breakpoints.index],
            mode='markers',
            name='Regime Change',
            marker=dict(color='#ef4444', size=10, symbol='x')
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#666666")
    
    fig.update_layout(
        title='Detección de Cambios de Régimen',
        xaxis_title='Fecha',
        yaxis_title='Correlación',
        template='plotly_dark',
        height=400
    )
    
    return fig

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

st.title("🎯 EA Pairs Trading - Candidate Finder")
st.markdown("**Encuentra los mejores pares para tu Expert Advisor de MetaTrader 5**")

# ============================================================================
# SIDEBAR - GESTIÓN DE DATOS
# ============================================================================

st.sidebar.header("💾 Gestión de Datos")

cache_info = get_cache_info()

if cache_info:
    st.sidebar.success("✅ Datos en cache")
    st.sidebar.metric("Última actualización", cache_info['last_update'].strftime('%Y-%m-%d %H:%M'))
    st.sidebar.metric("Total activos", cache_info['total_assets'])
    
    days_old = (datetime.now() - cache_info['last_update']).days
    if days_old > 0:
        st.sidebar.warning(f"⏰ Datos de hace {days_old} días")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Actualizar", use_container_width=True):
            with st.spinner("Actualizando datos..."):
                existing_data, existing_metadata = load_data_from_cache()
                
                if existing_data and existing_metadata:
                    updated_data, updated_metadata = update_existing_data(
                        existing_data, existing_metadata, delay=2
                    )
                    
                    if save_data_to_cache(updated_data, updated_metadata):
                        st.success(f"✅ Actualizados {updated_metadata.get('updated_count', 0)} activos")
                        st.rerun()
    
    with col2:
        if st.button("🗑️ Borrar", use_container_width=True):
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
            if METADATA_FILE.exists():
                METADATA_FILE.unlink()
            st.success("Cache borrado")
            st.rerun()
    
    if 'all_asset_data' not in st.session_state:
        with st.spinner("Cargando datos desde cache..."):
            data, metadata = load_data_from_cache()
            if data and metadata:
                st.session_state.all_asset_data = data
                st.session_state.metadata = metadata
                st.success("✅ Datos cargados desde cache")
    
else:
    st.sidebar.warning("⚠️ No hay datos descargados")
    
    if st.sidebar.button("📥 Descargar Todos los Activos", type="primary", use_container_width=True):
        with st.spinner(f"Descargando {len(ASSETS)} activos..."):
            all_data, metadata = download_all_assets(delay=3, start_date='2020-01-01')
        
        if len(all_data) > 0:
            if save_data_to_cache(all_data, metadata):
                st.success(f"✅ Descargados {len(all_data)} activos")
                st.session_state.all_asset_data = all_data
                st.session_state.metadata = metadata
                
                if len(metadata['failed_assets']) > 0:
                    st.warning(f"⚠️ {len(metadata['failed_assets'])} activos fallaron")
                
                st.rerun()

if 'all_asset_data' not in st.session_state:
    st.info(f"""
    ### 👋 Bienvenido al EA Pairs Trading Candidate Finder
    
    **Activos disponibles ({len(ASSETS)}):**
    - 📊 {len([a for a in ASSETS.values() if a['category'] == 'Indices'])} Índices globales
    - 💱 {len([a for a in ASSETS.values() if a['category'] == 'Forex'])} Pares de divisas
    - 🏆 {len([a for a in ASSETS.values() if a['category'] == 'Commodities'])} Commodities
    - ₿ {len([a for a in ASSETS.values() if a['category'] == 'Crypto'])} Criptomonedas
    
    **Para comenzar:**
    1. Presiona "📥 Descargar Todos los Activos"
    2. Los datos se guardarán en disco
    3. Se usarán TODOS los activos para encontrar correlaciones
    """)
    st.stop()

# ============================================================================
# PARÁMETROS DEL EA (SIN AUTO-UPDATE)
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parámetros del EA")

lookback = st.sidebar.slider("InpLookback", 50, 200, 100, 10, key='param_lookback')
zscore_threshold = st.sidebar.slider("InpZScoreThreshold", 1.5, 5.0, 3.3, 0.1, key='param_zscore')
correlation_threshold = st.sidebar.slider("InpCorrelationThreshold", 0.3, 0.9, 0.5, 0.05, key='param_corr')

st.sidebar.subheader("Filtros de Búsqueda")
min_signals_year = st.sidebar.slider("Mín. Señales/Año", 5, 100, 10, 5, key='param_signals')
max_cv = st.sidebar.slider("Máx. CV (estabilidad)", 0.2, 0.8, 0.4, 0.05, key='param_cv')

# Crear DataFrame con TODOS los activos
df_all_prices = merge_asset_data(st.session_state.all_asset_data)

if df_all_prices.empty:
    st.error("No hay datos suficientes")
    st.stop()

st.success(f"✅ {len(df_all_prices)} días | {df_all_prices.index[0].date()} → {df_all_prices.index[-1].date()}")
st.info(f"📊 Usando {len(df_all_prices.columns)} activos para análisis de correlaciones")

# ============================================================================
# BÚSQUEDA INICIAL DE PARES (CON BOTÓN)
# ============================================================================

st.markdown("---")
st.header("🔍 Mejores Pares Encontrados (Todos los Activos)")

if st.button("🚀 Buscar Mejores Pares", type="primary", key='btn_search_pairs'):
    
    # Buscar pares con CORRELACIÓN POSITIVA
    st.markdown("### 📈 Buscando Pares con Correlación POSITIVA...")
    with st.spinner("Analizando correlaciones positivas..."):
        positive_pairs = find_best_pairs_for_ea(
            df_all_prices,
            correlation_type='positive',
            min_correlation=correlation_threshold,
            min_signals_per_year=min_signals_year,
            max_cv=max_cv,
            zscore_threshold=zscore_threshold,
            lookback=lookback
        )
    
    # Buscar pares con CORRELACIÓN NEGATIVA
    st.markdown("### 📉 Buscando Pares con Correlación NEGATIVA (Inversa)...")
    with st.spinner("Analizando correlaciones negativas..."):
        negative_pairs = find_best_pairs_for_ea(
            df_all_prices,
            correlation_type='negative',
            min_correlation=correlation_threshold,
            min_signals_per_year=min_signals_year,
            max_cv=max_cv,
            zscore_threshold=zscore_threshold,
            lookback=lookback
        )
    
    # Guardar en session state
    st.session_state.positive_pairs = positive_pairs
    st.session_state.negative_pairs = negative_pairs
    st.success("✅ Búsqueda completada!")

# Mostrar resultados si existen
if 'positive_pairs' in st.session_state and 'negative_pairs' in st.session_state:
    
    col1, col2 = st.columns(2)
    
    # ========== TABLA CORRELACIÓN POSITIVA ==========
    with col1:
        st.markdown("### 📈 Top 10 Pares - Correlación POSITIVA")
        
        if len(st.session_state.positive_pairs) > 0:
            display_pos = st.session_state.positive_pairs.head(10).copy()
            display_pos['Activo 1'] = display_pos['asset1'].apply(lambda x: ASSETS[x]['label'])
            display_pos['Activo 2'] = display_pos['asset2'].apply(lambda x: ASSETS[x]['label'])
            display_pos['InpInvertTrades'] = 'false'
            
            table_pos = display_pos[['Activo 1', 'Activo 2', 'score', 'mean_correlation', 
                                     'signals_per_year', 'win_rate', 'InpInvertTrades']].rename(columns={
                'score': 'Score',
                'mean_correlation': 'Corr',
                'signals_per_year': 'Señales/Año',
                'win_rate': 'Win Rate %'
            })
            
            st.dataframe(
                table_pos.style.format({
                    'Score': '{:.1f}',
                    'Corr': '{:.3f}',
                    'Señales/Año': '{:.1f}',
                    'Win Rate %': '{:.1f}%'
                }),
                use_container_width=True
            )
            
            st.metric("Total pares positivos", len(st.session_state.positive_pairs))
        else:
            st.warning("No se encontraron pares con correlación positiva")
    
    # ========== TABLA CORRELACIÓN NEGATIVA ==========
    with col2:
        st.markdown("### 📉 Top 10 Pares - Correlación NEGATIVA (Inversa)")
        
        if len(st.session_state.negative_pairs) > 0:
            display_neg = st.session_state.negative_pairs.head(10).copy()
            display_neg['Activo 1'] = display_neg['asset1'].apply(lambda x: ASSETS[x]['label'])
            display_neg['Activo 2'] = display_neg['asset2'].apply(lambda x: ASSETS[x]['label'])
            display_neg['InpInvertTrades'] = 'true'
            
            table_neg = display_neg[['Activo 1', 'Activo 2', 'score', 'mean_correlation', 
                                     'signals_per_year', 'win_rate', 'InpInvertTrades']].rename(columns={
                'score': 'Score',
                'mean_correlation': 'Corr',
                'signals_per_year': 'Señales/Año',
                'win_rate': 'Win Rate %'
            })
            
            st.dataframe(
                table_neg.style.format({
                    'Score': '{:.1f}',
                    'Corr': '{:.3f}',
                    'Señales/Año': '{:.1f}',
                    'Win Rate %': '{:.1f}%'
                }),
                use_container_width=True
            )
            
            st.metric("Total pares negativos", len(st.session_state.negative_pairs))
        else:
            st.warning("No se encontraron pares con correlación negativa")

# ============================================================================
# ANÁLISIS DETALLADO (CON BOTÓN)
# ============================================================================

st.markdown("---")
st.header("📊 Análisis Detallado de Par")

# Selección de activos
available_assets = list(st.session_state.all_asset_data.keys())

col1, col2 = st.columns(2)

with col1:
    asset1 = st.selectbox(
        "Activo 1",
        options=available_assets,
        format_func=lambda x: ASSETS[x]['label'],
        key='detail_asset1'
    )

with col2:
    asset2 = st.selectbox(
        "Activo 2",
        options=[a for a in available_assets if a != asset1],
        format_func=lambda x: ASSETS[x]['label'],
        key='detail_asset2'
    )

# Configuración del análisis
st.markdown("### ⚙️ Configuración del Análisis")

col1, col2, col3 = st.columns(3)

with col1:
    correlation_type_analysis = st.radio(
        "Tipo de Correlación a Buscar",
        options=['Positiva', 'Negativa'],
        help="**Positiva**: Busca pares que se mueven juntos (correlación >0.5)\n**Negativa**: Busca pares que se mueven en direcciones opuestas (correlación <-0.5)"
    )

with col2:
    invert_trades = st.checkbox(
        "InpInvertTrades",
        value=False,
        help="**InpInvertTrades**: Invierte las señales de trading del EA\n- false: Z-score alto = SHORT, Z-score bajo = LONG\n- true: Z-score alto = LONG, Z-score bajo = SHORT"
    )

with col3:
    if st.button("🔄 Actualizar Análisis", type="primary", key='btn_analyze'):
        st.session_state.run_analysis = True

st.info("""
**📝 Diferencia entre Tipo de Correlación e InpInvertTrades:**

- **Tipo de Correlación**: Determina QUÉ tipo de pares estás buscando
  - Positiva: Activos que se mueven en la MISMA dirección (ej: S&P 500 y NASDAQ)
  - Negativa: Activos que se mueven en dirección OPUESTA (ej: USD Index y EUR/USD)

- **InpInvertTrades**: Invierte las SEÑALES de trading del EA
  - false (default): Si el spread sube mucho (Z-score alto) → Espera que baje → SHORT
  - true: Si el spread sube mucho (Z-score alto) → Espera que siga subiendo → LONG
""")

# Solo ejecutar análisis si se presionó el botón
if st.session_state.get('run_analysis', False):
    
    prices1 = df_all_prices[asset1]
    prices2 = df_all_prices[asset2]
    
    # Rolling Correlation
    st.markdown("### 📈 Rolling Correlation")
    corr_df = calculate_rolling_correlation(df_all_prices, asset1, asset2, window=lookback, step=5)
    st.plotly_chart(
        plot_rolling_correlation(corr_df, ASSETS[asset1]['label'], ASSETS[asset2]['label']),
        use_container_width=True
    )
    
    # Métricas de correlación
    st.markdown("### 📊 Métricas de Correlación")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_corr = corr_df['correlation'].iloc[-1]
    mean_corr = corr_df['correlation'].mean()
    max_corr = corr_df['correlation'].max()
    min_corr = corr_df['correlation'].min()
    
    col1.metric("Correlación Actual", f"{current_corr:.4f}")
    col2.metric("Correlación Media", f"{mean_corr:.4f}")
    col3.metric("Máxima", f"{max_corr:.4f}")
    col4.metric("Mínima", f"{min_corr:.4f}")
    
    # Verificar si cumple con el tipo de correlación buscado
    if correlation_type_analysis == 'Positiva':
        if mean_corr >= correlation_threshold:
            st.success(f"✅ Este par tiene correlación POSITIVA fuerte ({mean_corr:.3f} >= {correlation_threshold})")
        else:
            st.warning(f"⚠️ Este par NO tiene correlación positiva suficiente ({mean_corr:.3f} < {correlation_threshold})")
    else:  # Negativa
        if mean_corr <= -correlation_threshold:
            st.success(f"✅ Este par tiene correlación NEGATIVA fuerte ({mean_corr:.3f} <= {-correlation_threshold})")
        else:
            st.warning(f"⚠️ Este par NO tiene correlación negativa suficiente ({mean_corr:.3f} > {-correlation_threshold})")
    
    # Mostrar configuración sugerida para el EA
    st.markdown("### 💻 Configuración Sugerida para el EA")
    
    suggest_invert_based_on_corr = mean_corr < 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.code(f"""
// Configuración Manual Seleccionada:
InpSecondSymbol = "{ASSETS[asset2]['symbol']}"
InpInvertTrades = {str(invert_trades).lower()}
InpLookback = {lookback}
InpZScoreThresholdLong = {zscore_threshold}
InpZScoreThresholdShort = {zscore_threshold}
InpCorrelationThreshold = {correlation_threshold}
        """, language="c++")
    
    with col2:
        st.code(f"""
// Configuración Sugerida (basada en correlación):
InpSecondSymbol = "{ASSETS[asset2]['symbol']}"
InpInvertTrades = {str(suggest_invert_based_on_corr).lower()}  // {"Sugerido" if suggest_invert_based_on_corr == invert_trades else "⚠️ Diferente"}
InpLookback = {lookback}
InpZScoreThresholdLong = {zscore_threshold}
InpZScoreThresholdShort = {zscore_threshold}
InpCorrelationThreshold = {correlation_threshold}
        """, language="c++")
    
    if suggest_invert_based_on_corr != invert_trades:
        if suggest_invert_based_on_corr:
            st.warning("⚠️ **Nota**: La correlación es negativa, se sugiere InpInvertTrades = true")
        else:
            st.warning("⚠️ **Nota**: La correlación es positiva, se sugiere InpInvertTrades = false")
    
    # Comparación de precios
    st.markdown("### 📉 Comparación de Precios Normalizados")
    st.plotly_chart(
        plot_price_comparison(df_all_prices, asset1, asset2, 
                             ASSETS[asset1]['label'], ASSETS[asset2]['label']),
        use_container_width=True
    )
    
    # Correlación condicional
    st.markdown("### 🔍 Correlación Condicional")
    returns1 = np.log(prices1 / prices1.shift(1)).dropna()
    returns2 = np.log(prices2 / prices2.shift(1)).dropna()
    cond_corr = calculate_conditional_correlation(returns1, returns2)
    
    st.plotly_chart(plot_conditional_correlation(cond_corr), use_container_width=True)
    
    # Distribución Temporal
    st.markdown("### 📈 Distribución Temporal")
    
    positive = (corr_df['correlation'] > 0).sum()
    negative = (corr_df['correlation'] < 0).sum()
    strong_pos = (corr_df['correlation'] > 0.5).sum()
    strong_neg = (corr_df['correlation'] < -0.5).sum()
    total = len(corr_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("% Positiva", f"{positive/total*100:.1f}%")
    col2.metric("% Negativa", f"{negative/total*100:.1f}%")
    col3.metric("% Fuerte Pos (>0.5)", f"{strong_pos/total*100:.1f}%")
    col4.metric("% Fuerte Neg (<-0.5)", f"{strong_neg/total*100:.1f}%")
    
    # Distribución Percentiles
    st.markdown("### 📊 Distribución Percentiles")
    
    percentiles = {
        'p10': np.percentile(corr_df['correlation'].dropna(), 10),
        'p25': np.percentile(corr_df['correlation'].dropna(), 25),
        'p50': np.percentile(corr_df['correlation'].dropna(), 50),
        'p75': np.percentile(corr_df['correlation'].dropna(), 75),
        'p90': np.percentile(corr_df['correlation'].dropna(), 90)
    }
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("P10", f"{percentiles['p10']:.3f}")
    col2.metric("P25", f"{percentiles['p25']:.3f}")
    col3.metric("P50 (Mediana)", f"{percentiles['p50']:.3f}")
    col4.metric("P75", f"{percentiles['p75']:.3f}")
    col5.metric("P90", f"{percentiles['p90']:.3f}")
    
    # Estabilidad de Correlación
    st.markdown("### 🎯 Estabilidad de Correlación")
    
    corr_series = pd.Series(
        corr_df['correlation'].values,
        index=corr_df['date']
    )
    
    rolling_std = corr_series.rolling(60).std()
    rolling_mean = corr_series.rolling(60).mean()
    cv = (rolling_std / rolling_mean.abs()).replace([np.inf, -np.inf], np.nan)
    
    stability_df = pd.DataFrame({
        'corr_std': rolling_std,
        'corr_mean': rolling_mean,
        'stability_cv': cv
    })
    
    st.plotly_chart(plot_correlation_stability(stability_df), use_container_width=True)
    
    stability = calculate_correlation_stability(corr_series, window=60)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("CV Medio", f"{stability['mean_cv']:.3f}")
    col2.metric("CV Actual", f"{stability['current_cv']:.3f}")
    col3.metric("Desv. Std Corr", f"{stability['std_corr']:.3f}")
    
    # Puntos de Cambio de Régimen
    st.markdown("### ⚡ Puntos de Cambio de Régimen")
    
    breakpoints = detect_regime_changes(corr_series, threshold=0.3)
    
    if len(breakpoints) > 0:
        st.warning(f"⚠️ Detectados {len(breakpoints)} cambios significativos")
        st.plotly_chart(plot_regime_changes(corr_series, threshold=0.3), use_container_width=True)
        
        bp_df = pd.DataFrame({
            'Fecha': breakpoints.index,
            'Cambio Absoluto': breakpoints.values
        }).sort_values('Cambio Absoluto', ascending=False).head(10)
        
        st.dataframe(bp_df, use_container_width=True)
    else:
        st.success("✅ Correlación relativamente estable (sin cambios abruptos)")
    
    # Resetear flag
    st.session_state.run_analysis = False

else:
    st.info("👆 Selecciona los activos, configura el tipo de correlación y presiona **'🔄 Actualizar Análisis'** para ver los resultados")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Guía Rápida")
st.sidebar.markdown("""
**Botones de Control:**
- 🚀 **Buscar Pares**: Analiza TODOS los activos
- 🔄 **Actualizar Análisis**: Actualiza análisis individual

**Configuración:**
- **Tipo de Correlación**: QUÉ pares buscar
- **InpInvertTrades**: CÓMO tradear las señales

**No se actualiza automáticamente** - debes presionar los botones manualmente
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Activos")
st.sidebar.markdown(f"""
- 📊 Indices: {len([a for a in ASSETS.values() if a['category'] == 'Indices'])}
- 💱 Forex: {len([a for a in ASSETS.values() if a['category'] == 'Forex'])}
- 🏆 Commodities: {len([a for a in ASSETS.values() if a['category'] == 'Commodities'])}
- ₿ Crypto: {len([a for a in ASSETS.values() if a['category'] == 'Crypto'])}
- 📈 Volatility: {len([a for a in ASSETS.values() if a['category'] == 'Volatility'])}
""")

st.sidebar.success("✨ Diseñado para EA MQL5")
