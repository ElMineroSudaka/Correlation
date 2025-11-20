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

# Directorio para guardar datos
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
# CONFIGURACIÓN DE ACTIVOS - SIMPLIFICADA
# ============================================================================
ASSETS = {
    # ========== ÍNDICES GLOBALES ==========
    # US
    'us500': {'label': 'US SPX 500 (S&P 500)', 'symbol': '^GSPC', 'category': 'Indices'},
    'us30': {'label': 'US Wall Street 30 (Dow Jones)', 'symbol': '^DJI', 'category': 'Indices'},
    'ustec': {'label': 'US Tech 100 (NASDAQ)', 'symbol': '^IXIC', 'category': 'Indices'},
    'russell': {'label': 'Russell 2000', 'symbol': '^RUT', 'category': 'Indices'},
    
    # Europa
    'uk100': {'label': 'UK 100 (FTSE)', 'symbol': '^FTSE', 'category': 'Indices'},
    'de30': {'label': 'Germany 30 (DAX)', 'symbol': '^GDAXI', 'category': 'Indices'},
    'fr40': {'label': 'France 40 (CAC 40)', 'symbol': '^FCHI', 'category': 'Indices'},
    'stoxx50': {'label': 'EU Stocks 50 (Euro Stoxx)', 'symbol': '^STOXX50E', 'category': 'Indices'},
    
    # Asia-Pacífico
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
    
    # ========== METALES PRECIOSOS ==========
    'gold': {'label': 'Gold (GC)', 'symbol': 'GC=F', 'category': 'Commodities'},
    'silver': {'label': 'Silver (SI)', 'symbol': 'SI=F', 'category': 'Commodities'},
    'copper': {'label': 'Copper (HG)', 'symbol': 'HG=F', 'category': 'Commodities'},
    'platinum': {'label': 'Platinum (PL)', 'symbol': 'PL=F', 'category': 'Commodities'},
    
    # ========== ENERGÍA ==========
    'oil': {'label': 'Crude Oil WTI', 'symbol': 'CL=F', 'category': 'Commodities'},
    'brent': {'label': 'Brent Crude Oil', 'symbol': 'BZ=F', 'category': 'Commodities'},
    'natgas': {'label': 'Natural Gas', 'symbol': 'NG=F', 'category': 'Commodities'},
    
    # ========== AGRICULTURA ==========
    'corn': {'label': 'Corn', 'symbol': 'ZC=F', 'category': 'Commodities'},
    'wheat': {'label': 'Wheat', 'symbol': 'ZW=F', 'category': 'Commodities'},
    'soybeans': {'label': 'Soybeans', 'symbol': 'ZS=F', 'category': 'Commodities'},
    'sugar': {'label': 'Sugar', 'symbol': 'SB=F', 'category': 'Commodities'},
    'coffee': {'label': 'Coffee', 'symbol': 'KC=F', 'category': 'Commodities'},
    'cotton': {'label': 'Cotton', 'symbol': 'CT=F', 'category': 'Commodities'},
    
    # ========== CRIPTOMONEDAS ==========
    'btc': {'label': 'Bitcoin', 'symbol': 'BTC-USD', 'category': 'Crypto'},
    'eth': {'label': 'Ethereum', 'symbol': 'ETH-USD', 'category': 'Crypto'},
    
    # ========== VOLATILIDAD ==========
    'vix': {'label': 'VIX (S&P 500 Volatility)', 'symbol': '^VIX', 'category': 'Volatility'},
}

# ============================================================================
# FUNCIONES DE DESCARGA Y ACTUALIZACIÓN
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
    
    # Metadata
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
    
    # Determinar desde qué fecha actualizar
    last_update = existing_metadata['last_update']
    start_date = (last_update + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Si ya está actualizado
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
        
        # Descargar datos nuevos
        new_data = fetch_asset_data(symbol, start_date, end_date)
        
        if new_data is not None and len(new_data) > 0:
            # Combinar datos antiguos con nuevos
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
    
    # Actualizar metadata
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
    """Calcula Z-Score rolling (como el EA usa InpLookback=100)"""
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

def simulate_ea_signals(prices1, prices2, zscore_threshold=3.3, lookback=100, correlation_threshold=0.5):
    """
    Simula las señales que generaría el EA
    Retorna: DataFrame con señales, estadísticas y métricas
    """
    # Calcular spread log-ratio
    spread = calculate_log_ratio_spread(prices1, prices2)
    
    # Calcular Z-score
    zscore = calculate_zscore(spread, window=lookback)
    
    # Calcular correlación
    corr = calculate_correlation(prices1, prices2, window=lookback)
    
    # Alinear índices
    common_idx = zscore.index.intersection(corr.index)
    zscore = zscore.loc[common_idx]
    corr = corr.loc[common_idx]
    
    # Detectar señales
    signals = pd.DataFrame(index=common_idx)
    
    # Filtro de correlación (como el EA)
    valid_corr = corr.abs() >= correlation_threshold
    
    # Señales LONG (cuando Z-score es muy alto)
    signals['long_signal'] = (zscore > zscore_threshold) & valid_corr
    
    # Señales SHORT (cuando Z-score es muy bajo)
    signals['short_signal'] = (zscore < -zscore_threshold) & valid_corr
    
    # Cualquier señal
    signals['any_signal'] = signals['long_signal'] | signals['short_signal']
    
    # Métricas
    total_signals = signals['any_signal'].sum()
    long_signals = signals['long_signal'].sum()
    short_signals = signals['short_signal'].sum()
    
    # Calcular duración media de señales
    signal_groups = (signals['any_signal'] != signals['any_signal'].shift()).cumsum()
    signal_durations = signals[signals['any_signal']].groupby(signal_groups).size()
    avg_duration = signal_durations.mean() if len(signal_durations) > 0 else 0
    
    # Win rate simulado
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
    
    # Señales por año
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
    """Calcula el Hurst Exponent para medir mean reversion"""
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
        return {
            'adf_stat': result[0],
            'pvalue': result[1],
            'stationary': result[1] < 0.05
        }
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
    """Detecta cambios significativos de régimen en correlación"""
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

def find_best_pairs_for_ea(df, min_correlation=0.5, min_signals_per_year=10, 
                           max_cv=0.4, zscore_threshold=3.3, lookback=100):
    """
    Encuentra los mejores pares para el EA usando múltiples criterios
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
            
            if stats['signals_per_year'] < min_signals_per_year:
                continue
            
            adf_result = adf_test(spread)
            hurst = calculate_hurst_exponent(spread.dropna())
            half_life = calculate_half_life(spread)
            coint_result = test_cointegration(p1, p2)
            
            stability = calculate_correlation_stability(corr)
            
            if stability['mean_cv'] > max_cv:
                continue
            
            mean_corr = corr.mean()
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

def plot_pair_analysis(prices1, prices2, asset1_name, asset2_name, lookback=100, zscore_threshold=3.3):
    """Gráfico completo de análisis de un par"""
    signals, stats, zscore, corr, spread = simulate_ea_signals(
        prices1, prices2, zscore_threshold, lookback
    )
    
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            f'Precios Normalizados: {asset1_name} vs {asset2_name}',
            f'Rolling Correlation (window={lookback})',
            'Log-Ratio Spread',
            f'Z-Score con Señales (threshold=±{zscore_threshold})'
        ),
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )
    
    norm1 = (prices1 / prices1.iloc[0]) * 100
    norm2 = (prices2 / prices2.iloc[0]) * 100
    
    fig.add_trace(go.Scatter(x=norm1.index, y=norm1, name=asset1_name,
                             line=dict(color='#10b981', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=norm2.index, y=norm2, name=asset2_name,
                             line=dict(color='#3b82f6', width=2)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=corr.index, y=corr, name='Correlation',
                             line=dict(color='#8b5cf6', width=2)), row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="#10b981", opacity=0.5, row=2, col=1)
    fig.add_hline(y=-0.5, line_dash="dash", line_color="#ef4444", opacity=0.5, row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#666666", row=2, col=1)
    
    fig.add_trace(go.Scatter(x=spread.index, y=spread, name='Spread',
                             line=dict(color='#f59e0b', width=2)), row=3, col=1)
    
    fig.add_trace(go.Scatter(x=zscore.index, y=zscore, name='Z-Score',
                             line=dict(color='#06b6d4', width=2)), row=4, col=1)
    
    long_signals = signals[signals['long_signal']]
    short_signals = signals[signals['short_signal']]
    
    if len(long_signals) > 0:
        fig.add_trace(go.Scatter(
            x=long_signals.index,
            y=[zscore.loc[idx] for idx in long_signals.index],
            mode='markers',
            name='LONG Signal',
            marker=dict(color='#ef4444', size=8, symbol='triangle-down')
        ), row=4, col=1)
    
    if len(short_signals) > 0:
        fig.add_trace(go.Scatter(
            x=short_signals.index,
            y=[zscore.loc[idx] for idx in short_signals.index],
            mode='markers',
            name='SHORT Signal',
            marker=dict(color='#10b981', size=8, symbol='triangle-up')
        ), row=4, col=1)
    
    fig.add_hline(y=zscore_threshold, line_dash="dash", line_color="#ef4444", row=4, col=1)
    fig.add_hline(y=-zscore_threshold, line_dash="dash", line_color="#10b981", row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#666666", row=4, col=1)
    
    fig.add_hrect(y0=zscore_threshold, y1=10, fillcolor="#ef4444", opacity=0.1, line_width=0, row=4, col=1)
    fig.add_hrect(y0=-10, y1=-zscore_threshold, fillcolor="#10b981", opacity=0.1, line_width=0, row=4, col=1)
    
    fig.update_layout(height=1200, template='plotly_dark', showlegend=True, hovermode='x unified')
    fig.update_yaxes(title_text="Base 100", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=2, col=1)
    fig.update_yaxes(title_text="Spread", row=3, col=1)
    fig.update_yaxes(title_text="Z-Score", row=4, col=1)
    
    return fig, stats

def plot_correlation_analysis(corr_series, asset1_name, asset2_name):
    """Análisis detallado de correlación"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Rolling Correlation', 'Distribución de Correlación', 
                       'Estabilidad (Rolling CV)', 'Régimen de Correlación'),
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    fig.add_trace(go.Scatter(x=corr_series.index, y=corr_series,
                             line=dict(color='#3b82f6', width=2),
                             name='Correlation'), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", row=1, col=1)
    
    fig.add_trace(go.Histogram(x=corr_series.dropna(), nbinsx=50,
                               marker_color='#8b5cf6', name='Distribution'), row=1, col=2)
    
    rolling_std = corr_series.rolling(60).std()
    rolling_mean = corr_series.rolling(60).mean()
    cv = (rolling_std / rolling_mean.abs()).replace([np.inf, -np.inf], np.nan)
    
    fig.add_trace(go.Scatter(x=cv.index, y=cv,
                             line=dict(color='#f59e0b', width=2),
                             name='CV'), row=2, col=1)
    
    regime = pd.Series(0, index=corr_series.index)
    regime[corr_series > 0.3] = 1
    regime[corr_series < -0.3] = -1
    
    colors = ['#ef4444' if r == -1 else '#10b981' if r == 1 else '#6b7280' for r in regime]
    
    fig.add_trace(go.Scatter(x=regime.index, y=regime,
                             mode='markers',
                             marker=dict(color=colors, size=3),
                             name='Regime'), row=2, col=2)
    
    fig.update_layout(height=800, template='plotly_dark', showlegend=False)
    
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

def plot_conditional_correlation(cond_corr):
    """Gráfico de correlación condicional"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Normal', 'Alcista', 'Bajista', 'Alta Volatilidad'],
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

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

st.title("🎯 EA Pairs Trading - Candidate Finder")
st.markdown("**Encuentra los mejores pares para tu Expert Advisor de MetaTrader 5**")

# ============================================================================
# SIDEBAR - GESTIÓN DE DATOS
# ============================================================================

st.sidebar.header("💾 Gestión de Datos")

# Verificar si existe cache
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
                    else:
                        st.error("Error guardando actualización")
    
    with col2:
        if st.button("🗑️ Borrar", use_container_width=True):
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
            if METADATA_FILE.exists():
                METADATA_FILE.unlink()
            st.success("Cache borrado")
            st.rerun()
    
    # Cargar datos desde cache
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
        with st.spinner(f"Descargando {len(ASSETS)} activos... (delay 3s por activo)"):
            all_data, metadata = download_all_assets(delay=3, start_date='2020-01-01')
        
        if len(all_data) > 0:
            if save_data_to_cache(all_data, metadata):
                st.success(f"✅ Descargados {len(all_data)} activos")
                st.session_state.all_asset_data = all_data
                st.session_state.metadata = metadata
                
                if len(metadata['failed_assets']) > 0:
                    st.warning(f"⚠️ {len(metadata['failed_assets'])} activos fallaron")
                
                st.rerun()
            else:
                st.error("Error guardando datos")
        else:
            st.error("No se pudieron descargar datos")

# Si no hay datos cargados, detener aquí
if 'all_asset_data' not in st.session_state:
    st.info(f"""
    ### 👋 Bienvenido al EA Pairs Trading Candidate Finder
    
    **Esta herramienta está diseñada específicamente para tu Expert Advisor de MQL5**
    
    **Activos disponibles ({len(ASSETS)}):**
    - 📊 {len([a for a in ASSETS.values() if a['category'] == 'Indices'])} Índices globales (US, Europa, Asia)
    - 💱 {len([a for a in ASSETS.values() if a['category'] == 'Forex'])} Pares de divisas
    - 🏆 {len([a for a in ASSETS.values() if a['category'] == 'Commodities'])} Commodities (metales, energía, agricultura)
    - ₿ {len([a for a in ASSETS.values() if a['category'] == 'Crypto'])} Criptomonedas
    
    **Para comenzar:**
    1. Presiona "📥 Descargar Todos los Activos" en el sidebar
    2. Los datos se guardarán en disco (solo 1 vez)
    3. Después solo tendrás que actualizarlos
    
    **Características:**
    - 🎯 Simula señales del EA (log-ratio + Z-score)
    - 📊 Win rate y frecuencia de señales
    - 💡 Sugiere parámetros óptimos
    - 💾 Cache persistente
    """)
    st.stop()

# ============================================================================
# SELECCIÓN DE ACTIVOS
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuración")

# Filtrar por categoría
categories = list(set([ASSETS[k]['category'] for k in ASSETS.keys()]))
categories.sort()

selected_categories = st.sidebar.multiselect(
    "Categorías",
    options=categories,
    default=[c for c in ['Indices', 'Commodities', 'Crypto'] if c in categories]
)

available_assets = [k for k in st.session_state.all_asset_data.keys() 
                   if ASSETS[k]['category'] in selected_categories]

st.sidebar.subheader("Activos Seleccionados")

if len(available_assets) > 0:
    default_selection = available_assets[:min(8, len(available_assets))]
else:
    default_selection = []

selected_assets = st.sidebar.multiselect(
    "Activos para análisis",
    options=available_assets,
    default=default_selection,
    format_func=lambda x: ASSETS[x]['label']
)

if len(selected_assets) < 2:
    st.warning("⚠️ Selecciona al menos 2 activos")
    st.stop()

st.sidebar.info(f"✅ {len(selected_assets)} activos seleccionados")

# Parámetros del EA
st.sidebar.subheader("Parámetros del EA")
lookback = st.sidebar.slider("InpLookback", 50, 200, 100, 10)
zscore_threshold = st.sidebar.slider("InpZScoreThreshold", 1.5, 5.0, 3.3, 0.1)
correlation_threshold = st.sidebar.slider("InpCorrelationThreshold", 0.3, 0.9, 0.5, 0.05)

# Filtros de búsqueda
st.sidebar.subheader("Filtros de Búsqueda")
min_signals_year = st.sidebar.slider("Mín. Señales/Año", 5, 100, 10, 5)
max_cv = st.sidebar.slider("Máx. CV (estabilidad)", 0.2, 0.8, 0.4, 0.05)

# Crear DataFrame con activos seleccionados
df_prices = merge_asset_data({k: st.session_state.all_asset_data[k] for k in selected_assets})

if df_prices.empty:
    st.error("No hay datos suficientes para los activos seleccionados")
    st.stop()

st.success(f"✅ {len(df_prices)} días | {df_prices.index[0].date()} → {df_prices.index[-1].date()}")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Búsqueda Automática",
    "📈 Análisis Individual",
    "📊 Análisis de Correlación",
    "💡 Optimización de Parámetros"
])

with tab1:
    st.subheader("🔍 Búsqueda Automática de Mejores Pares")
    st.caption("Encuentra automáticamente los mejores candidatos para tu EA")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🚀 Buscar Pares", type="primary", use_container_width=True):
            st.session_state.search_done = True
    
    with col2:
        st.info(f"Analizando {len(selected_assets)} activos = {len(selected_assets)*(len(selected_assets)-1)//2} pares posibles")
    
    if 'search_done' in st.session_state and st.session_state.search_done:
        with st.spinner("Analizando pares..."):
            best_pairs = find_best_pairs_for_ea(
                df_prices[selected_assets],
                min_correlation=correlation_threshold,
                min_signals_per_year=min_signals_year,
                max_cv=max_cv,
                zscore_threshold=zscore_threshold,
                lookback=lookback
            )
        
        if len(best_pairs) > 0:
            st.success(f"✅ Encontrados {len(best_pairs)} pares")
            
            top_pairs = best_pairs.head(15).copy()
            top_pairs['pair_label'] = top_pairs['asset1'].apply(lambda x: ASSETS[x]['label']) + ' / ' + \
                                      top_pairs['asset2'].apply(lambda x: ASSETS[x]['label'])
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=top_pairs['pair_label'],
                x=top_pairs['score'],
                orientation='h',
                marker=dict(color=top_pairs['score'], colorscale='Viridis', showscale=True),
                text=top_pairs['score'].round(1),
                textposition='auto'
            ))
            
            fig.update_layout(
                title='Top 15 Mejores Pares',
                xaxis_title='Score',
                yaxis_title='Par',
                template='plotly_dark',
                height=600,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla detallada
            display_df = best_pairs.head(20).copy()
            display_df['asset1_name'] = display_df['asset1'].apply(lambda x: ASSETS[x]['label'])
            display_df['asset2_name'] = display_df['asset2'].apply(lambda x: ASSETS[x]['label'])
            display_df['invert_trades'] = display_df['suggest_invert_trades'].apply(lambda x: '✅ YES' if x else '❌ NO')
            
            cols_to_show = {
                'asset1_name': 'Activo 1',
                'asset2_name': 'Activo 2',
                'score': 'Score',
                'signals_per_year': 'Señales/Año',
                'win_rate': 'Win Rate %',
                'mean_correlation': 'Corr Media',
                'corr_stability_cv': 'CV',
                'hurst': 'Hurst',
                'stationary': 'Estacionario',
                'invert_trades': 'InpInvertTrades'
            }
            
            display_table = display_df[list(cols_to_show.keys())].rename(columns=cols_to_show)
            
            styled = display_table.style.format({
                'Score': '{:.1f}',
                'Señales/Año': '{:.1f}',
                'Win Rate %': '{:.1f}%',
                'Corr Media': '{:.3f}',
                'CV': '{:.3f}',
                'Hurst': '{:.3f}'
            })
            
            st.dataframe(styled, use_container_width=True, height=600)
            
            # Análisis del mejor par
            st.markdown("---")
            st.markdown("### 🏆 Análisis del Mejor Par")
            
            best = best_pairs.iloc[0]
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.metric("Par", f"{ASSETS[best['asset1']]['label']} / {ASSETS[best['asset2']]['label']}")
                st.metric("Score", f"{best['score']:.1f}")
                
                if best['suggest_invert_trades']:
                    st.error("**InpInvertTrades = true**")
                    st.caption("Correlación predominantemente negativa")
                else:
                    st.success("**InpInvertTrades = false**")
                    st.caption("Correlación predominantemente positiva")
            
            with col2:
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Señales/Año", f"{best['signals_per_year']:.1f}")
                col_b.metric("Win Rate", f"{best['win_rate']:.1f}%")
                col_c.metric("Hurst", f"{best['hurst']:.3f}")
                col_d.metric("CV", f"{best['corr_stability_cv']:.3f}")
            
            # Código sugerido
            st.markdown("### 💻 Código Sugerido para el EA")
            
            st.code(f"""
InpSecondSymbol = "{ASSETS[best['asset2']]['symbol']}"
InpInvertTrades = {"true" if best['suggest_invert_trades'] else "false"}
InpLookback = {lookback}
InpCorrelationThreshold = {correlation_threshold}
InpZScoreThresholdLong = {zscore_threshold}
InpZScoreThresholdShort = {zscore_threshold}

// Resultados esperados:
// - Señales por año: {best['signals_per_year']:.0f}
// - Win rate: {best['win_rate']:.1f}%
// - LONG/SHORT ratio: {best['long_signals']}/{best['short_signals']}
            """, language="c++")
            
            # Descargar CSV
            csv = best_pairs.to_csv(index=False)
            st.download_button("📥 Descargar CSV", csv, "ea_best_pairs.csv", "text/csv")
        else:
            st.warning("⚠️ No se encontraron pares que cumplan los criterios")

with tab2:
    st.subheader("📈 Análisis Individual de Par")
    
    col1, col2 = st.columns(2)
    
    with col1:
        asset1 = st.selectbox("Activo 1", selected_assets, 
                             format_func=lambda x: ASSETS[x]['label'], key='ind_asset1')
    
    with col2:
        asset2 = st.selectbox("Activo 2", [a for a in selected_assets if a != asset1],
                             format_func=lambda x: ASSETS[x]['label'], key='ind_asset2')
    
    prices1 = df_prices[asset1]
    prices2 = df_prices[asset2]
    
    fig, stats = plot_pair_analysis(prices1, prices2, ASSETS[asset1]['label'], 
                                     ASSETS[asset2]['label'], lookback, zscore_threshold)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Señales", stats['total_signals'])
    col2.metric("LONG", stats['long_signals'])
    col3.metric("SHORT", stats['short_signals'])
    col4.metric("Señales/Año", f"{stats['signals_per_year']:.1f}")
    col5.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlación condicional
    st.markdown("### 🔍 Correlación Condicional")
    returns1 = np.log(prices1 / prices1.shift(1)).dropna()
    returns2 = np.log(prices2 / prices2.shift(1)).dropna()
    cond_corr = calculate_conditional_correlation(returns1, returns2)
    
    st.plotly_chart(plot_conditional_correlation(cond_corr), use_container_width=True)

with tab3:
    st.subheader("📊 Análisis Detallado de Correlación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        corr_asset1 = st.selectbox("Activo 1", selected_assets,
                                  format_func=lambda x: ASSETS[x]['label'], key='corr_asset1')
    
    with col2:
        corr_asset2 = st.selectbox("Activo 2", [a for a in selected_assets if a != corr_asset1],
                                  format_func=lambda x: ASSETS[x]['label'], key='corr_asset2')
    
    corr = calculate_correlation(df_prices[corr_asset1], df_prices[corr_asset2], lookback)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Media", f"{corr.mean():.3f}")
    col2.metric("Actual", f"{corr.iloc[-1]:.3f}")
    col3.metric("Máxima", f"{corr.max():.3f}")
    col4.metric("Mínima", f"{corr.min():.3f}")
    
    st.plotly_chart(plot_correlation_analysis(corr, ASSETS[corr_asset1]['label'],
                                              ASSETS[corr_asset2]['label']), use_container_width=True)
    
    # Cambios de régimen
    st.markdown("### ⚡ Cambios de Régimen")
    breakpoints = detect_regime_changes(corr, threshold=0.3)
    
    if len(breakpoints) > 0:
        st.warning(f"⚠️ Detectados {len(breakpoints)} cambios significativos")
        st.plotly_chart(plot_regime_changes(corr, threshold=0.3), use_container_width=True)
    else:
        st.success("✅ Correlación estable")

with tab4:
    st.subheader("💡 Optimización de Parámetros")
    st.info("Encuentra los mejores Z-Score y Lookback para un par específico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        opt_asset1 = st.selectbox("Activo 1", selected_assets,
                                 format_func=lambda x: ASSETS[x]['label'], key='opt_asset1')
    
    with col2:
        opt_asset2 = st.selectbox("Activo 2", [a for a in selected_assets if a != opt_asset1],
                                 format_func=lambda x: ASSETS[x]['label'], key='opt_asset2')
    
    st.caption("Presiona para probar diferentes combinaciones de parámetros")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Categorías")
st.sidebar.markdown(f"""
**Activos ({len(ASSETS)}):**
- 📊 Indices: {len([a for a in ASSETS.values() if a['category'] == 'Indices'])}
- 💱 Forex: {len([a for a in ASSETS.values() if a['category'] == 'Forex'])}
- 🏆 Commodities: {len([a for a in ASSETS.values() if a['category'] == 'Commodities'])}
- ₿ Crypto: {len([a for a in ASSETS.values() if a['category'] == 'Crypto'])}
- 📈 Volatility: {len([a for a in ASSETS.values() if a['category'] == 'Volatility'])}
""")

st.sidebar.success("✨ Diseñado para EA MQL5")
