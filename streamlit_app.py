import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import spearmanr, kendalltau
from statsmodels.tsa.stattools import adfuller, coint
import warnings
import pickle
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Pairs Trading - Beta & Correlation Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1e2130; padding: 15px; border-radius: 10px;}
    h1, h2, h3 {color: #ffffff;}
    .beta-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
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
    'dxy': {'label': 'US Dollar Index (DXY)', 'symbol': 'DX-Y.NYB', 'category': 'Forex'},
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

def fetch_asset_data(symbol, start_date='2015-01-01', end_date=None):
    """Descarga datos históricos de un activo - 10 años"""
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

def download_all_assets(delay=3, start_date='2015-01-01'):
    """Descarga TODOS los activos con delay - 10 años"""
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
    """Actualiza datos existentes"""
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
# FUNCIONES DE CÁLCULO DE BETA
# ============================================================================

def calculate_beta_ols(prices1, prices2, use_log=True):
    """
    Calcula Beta usando OLS (Ordinary Least Squares)
    Beta = Cov(ln(P1), ln(P2)) / Var(ln(P2))
    
    Spread = ln(P1) - Beta * ln(P2)
    """
    if use_log:
        y = np.log(prices1).dropna()
        x = np.log(prices2).dropna()
    else:
        y = prices1.dropna()
        x = prices2.dropna()
    
    common_idx = y.index.intersection(x.index)
    y = y.loc[common_idx]
    x = x.loc[common_idx]
    
    if len(x) < 2:
        return np.nan, np.nan
    
    # Beta = Cov(Y, X) / Var(X)
    cov = np.cov(y, x)[0, 1]
    var = np.var(x, ddof=1)
    
    if var == 0:
        return np.nan, np.nan
    
    beta = cov / var
    
    # Alpha (intercepto)
    alpha = np.mean(y) - beta * np.mean(x)
    
    return beta, alpha


def calculate_beta_theil_sen(prices1, prices2, use_log=True):
    """
    Calcula Beta usando Theil-Sen (robusto a outliers)
    Usa la mediana de las pendientes entre todos los pares de puntos
    """
    if use_log:
        y = np.log(prices1).dropna()
        x = np.log(prices2).dropna()
    else:
        y = prices1.dropna()
        x = prices2.dropna()
    
    common_idx = y.index.intersection(x.index)
    y = y.loc[common_idx].values
    x = x.loc[common_idx].values
    
    if len(x) < 2:
        return np.nan, np.nan
    
    try:
        result = stats.theilslopes(y, x)
        beta = result[0]  # slope
        alpha = result[1]  # intercept
        return beta, alpha
    except:
        return np.nan, np.nan


def calculate_rolling_beta(prices1, prices2, window=100, method='ols', use_log=True):
    """
    Calcula Beta rolling usando ventana deslizante
    
    Args:
        method: 'ols' o 'theil_sen'
    """
    if use_log:
        y = np.log(prices1).dropna()
        x = np.log(prices2).dropna()
    else:
        y = prices1.dropna()
        x = prices2.dropna()
    
    common_idx = y.index.intersection(x.index)
    y = y.loc[common_idx]
    x = x.loc[common_idx]
    
    betas = []
    alphas = []
    dates = []
    
    for i in range(window, len(y)):
        window_y = y.iloc[i-window:i]
        window_x = x.iloc[i-window:i]
        
        if method == 'ols':
            cov = np.cov(window_y, window_x)[0, 1]
            var = np.var(window_x, ddof=1)
            if var != 0:
                beta = cov / var
                alpha = np.mean(window_y) - beta * np.mean(window_x)
            else:
                beta = np.nan
                alpha = np.nan
        else:  # theil_sen
            try:
                result = stats.theilslopes(window_y.values, window_x.values)
                beta = result[0]
                alpha = result[1]
            except:
                beta = np.nan
                alpha = np.nan
        
        betas.append(beta)
        alphas.append(alpha)
        dates.append(y.index[i])
    
    return pd.DataFrame({
        'date': dates,
        'beta': betas,
        'alpha': alphas
    })


def calculate_beta_volatility_adjusted(prices1, prices2, window=100):
    """
    Beta ajustado por volatilidad
    Beta = (σ1/σ2) * correlation
    """
    returns1 = np.log(prices1 / prices1.shift(1)).dropna()
    returns2 = np.log(prices2 / prices2.shift(1)).dropna()
    
    common_idx = returns1.index.intersection(returns2.index)
    returns1 = returns1.loc[common_idx]
    returns2 = returns2.loc[common_idx]
    
    betas = []
    dates = []
    
    for i in range(window, len(returns1)):
        window_r1 = returns1.iloc[i-window:i]
        window_r2 = returns2.iloc[i-window:i]
        
        std1 = window_r1.std()
        std2 = window_r2.std()
        corr = window_r1.corr(window_r2)
        
        if std2 != 0:
            beta = (std1 / std2) * corr
        else:
            beta = np.nan
        
        betas.append(beta)
        dates.append(returns1.index[i])
    
    return pd.DataFrame({
        'date': dates,
        'beta': betas
    })


def analyze_beta_stability(beta_series):
    """Analiza la estabilidad del beta en el tiempo"""
    beta_values = beta_series['beta'].dropna()
    
    if len(beta_values) < 10:
        return None
    
    return {
        'mean': beta_values.mean(),
        'std': beta_values.std(),
        'cv': beta_values.std() / abs(beta_values.mean()) if beta_values.mean() != 0 else np.nan,
        'min': beta_values.min(),
        'max': beta_values.max(),
        'range': beta_values.max() - beta_values.min(),
        'current': beta_values.iloc[-1],
        'distance_from_1': abs(beta_values.mean() - 1),
        'pct_above_1': (beta_values > 1).sum() / len(beta_values) * 100,
        'pct_below_1': (beta_values < 1).sum() / len(beta_values) * 100,
    }


# ============================================================================
# FUNCIONES DE SPREAD CON BETA
# ============================================================================

def calculate_spread_with_beta(prices1, prices2, beta, use_log=True):
    """
    Calcula el spread usando el beta estimado
    Spread = ln(P1) - beta * ln(P2)
    """
    if use_log:
        log_p1 = np.log(prices1)
        log_p2 = np.log(prices2)
        spread = log_p1 - beta * log_p2
    else:
        spread = prices1 - beta * prices2
    
    return spread.dropna()


def calculate_zscore_spread(spread, window=100):
    """Calcula Z-Score del spread"""
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    zscore = (spread - mean) / std
    return zscore.dropna()


# ============================================================================
# FUNCIONES DE CORRELACIÓN MÚLTIPLE
# ============================================================================

def calculate_all_correlations(prices1, prices2, window=100):
    """
    Calcula múltiples tipos de correlación:
    - Pearson (lineal)
    - Spearman (monótona, rank-based)
    - Kendall (concordancia)
    """
    returns1 = np.log(prices1 / prices1.shift(1)).dropna()
    returns2 = np.log(prices2 / prices2.shift(1)).dropna()
    
    common_idx = returns1.index.intersection(returns2.index)
    returns1 = returns1.loc[common_idx]
    returns2 = returns2.loc[common_idx]
    
    pearson_list = []
    spearman_list = []
    kendall_list = []
    dates = []
    
    for i in range(window, len(returns1)):
        window_r1 = returns1.iloc[i-window:i]
        window_r2 = returns2.iloc[i-window:i]
        
        # Pearson
        pearson = window_r1.corr(window_r2)
        
        # Spearman
        spearman, _ = spearmanr(window_r1, window_r2)
        
        # Kendall
        kendall, _ = kendalltau(window_r1, window_r2)
        
        pearson_list.append(pearson)
        spearman_list.append(spearman)
        kendall_list.append(kendall)
        dates.append(returns1.index[i])
    
    return pd.DataFrame({
        'date': dates,
        'pearson': pearson_list,
        'spearman': spearman_list,
        'kendall': kendall_list
    })


def analyze_correlation_divergence(corr_df):
    """
    Analiza la divergencia entre tipos de correlación
    Si Pearson y Spearman divergen mucho, la relación no es lineal
    """
    if len(corr_df) < 10:
        return None
    
    pearson = corr_df['pearson']
    spearman = corr_df['spearman']
    kendall = corr_df['kendall']
    
    divergence_ps = abs(pearson - spearman).mean()
    divergence_pk = abs(pearson - kendall).mean()
    divergence_sk = abs(spearman - kendall).mean()
    
    return {
        'pearson_mean': pearson.mean(),
        'spearman_mean': spearman.mean(),
        'kendall_mean': kendall.mean(),
        'divergence_pearson_spearman': divergence_ps,
        'divergence_pearson_kendall': divergence_pk,
        'divergence_spearman_kendall': divergence_sk,
        'is_linear': divergence_ps < 0.1,  # Si divergencia < 0.1, relación es lineal
        'recommendation': 'Pearson' if divergence_ps < 0.1 else 'Spearman'
    }


# ============================================================================
# FUNCIONES DE ANÁLISIS EXISTENTES (MEJORADAS)
# ============================================================================

def calculate_log_ratio_spread(prices1, prices2):
    """Calcula spread usando log-ratio (beta=1)"""
    spread = np.log(prices1) - np.log(prices2)
    return spread.dropna()

def calculate_rolling_correlation(df, asset1, asset2, window=30, step=1):
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

def calculate_spread_volatility(spread):
    """Calcula volatilidad del spread"""
    returns = spread.diff()
    return returns.std()

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

# ============================================================================
# FUNCIONES DE ANÁLISIS DE ESTACIONALIDAD
# ============================================================================

def analyze_seasonality(df, asset1, asset2, lookback=100):
    """Analiza patrones estacionales en la correlación y el spread"""
    prices1 = df[asset1]
    prices2 = df[asset2]
    
    spread = calculate_log_ratio_spread(prices1, prices2)
    
    returns1 = np.log(prices1 / prices1.shift(1))
    returns2 = np.log(prices2 / prices2.shift(1))
    corr_rolling = returns1.rolling(lookback).corr(returns2)
    
    spread_df = spread.to_frame('spread')
    spread_df['month'] = spread_df.index.month
    spread_df['quarter'] = spread_df.index.quarter
    spread_df['year'] = spread_df.index.year
    spread_df['day_of_year'] = spread_df.index.dayofyear
    spread_df['volatility'] = spread_df['spread'].rolling(30).std()
    
    corr_df = corr_rolling.to_frame('correlation')
    corr_df['month'] = corr_df.index.month
    corr_df['quarter'] = corr_df.index.quarter
    corr_df['year'] = corr_df.index.year
    
    monthly_corr = corr_df.groupby('month')['correlation'].agg(['mean', 'std', 'min', 'max'])
    monthly_spread_vol = spread_df.groupby('month')['volatility'].mean()
    
    quarterly_corr = corr_df.groupby('quarter')['correlation'].agg(['mean', 'std', 'min', 'max'])
    quarterly_spread_vol = spread_df.groupby('quarter')['volatility'].mean()
    
    yearly_corr = corr_df.groupby('year')['correlation'].agg(['mean', 'std', 'min', 'max'])
    yearly_spread_vol = spread_df.groupby('year')['volatility'].mean()
    
    monthly_spread_stats = spread_df.groupby('month')['spread'].agg(['mean', 'std'])
    
    return {
        'monthly_corr': monthly_corr,
        'monthly_spread_vol': monthly_spread_vol,
        'monthly_spread_stats': monthly_spread_stats,
        'quarterly_corr': quarterly_corr,
        'quarterly_spread_vol': quarterly_spread_vol,
        'yearly_corr': yearly_corr,
        'yearly_spread_vol': yearly_spread_vol,
        'spread_df': spread_df,
        'corr_df': corr_df
    }

def calculate_historical_periods(df, asset1, asset2):
    """Calcula estadísticas por períodos históricos significativos"""
    prices1 = df[asset1]
    prices2 = df[asset2]
    
    spread = calculate_log_ratio_spread(prices1, prices2)
    
    periods = []
    
    historical_events = [
        ('2015-2016', '2015-01-01', '2016-12-31', 'Período 2015-2016'),
        ('2017-2018', '2017-01-01', '2018-12-31', 'Período 2017-2018'),
        ('2019', '2019-01-01', '2019-12-31', 'Pre-COVID 2019'),
        ('COVID-2020', '2020-01-01', '2020-12-31', 'COVID-19 2020'),
        ('2021', '2021-01-01', '2021-12-31', 'Recuperación 2021'),
        ('2022', '2022-01-01', '2022-12-31', 'Inflación 2022'),
        ('2023', '2023-01-01', '2023-12-31', 'Normalización 2023'),
        ('2024-2025', '2024-01-01', '2025-12-31', 'Período 2024-2025'),
    ]
    
    for period_id, start, end, label in historical_events:
        try:
            mask = (spread.index >= start) & (spread.index <= end)
            period_spread = spread[mask]
            period_p1 = prices1[mask]
            period_p2 = prices2[mask]
            
            if len(period_spread) > 30:
                period_corr = period_p1.corr(period_p2)
                
                # Calcular beta del período
                beta_ols, _ = calculate_beta_ols(period_p1, period_p2)
                
                periods.append({
                    'period': label,
                    'start': start,
                    'end': end,
                    'days': len(period_spread),
                    'correlation': period_corr,
                    'beta_ols': beta_ols,
                    'spread_mean': period_spread.mean(),
                    'spread_std': period_spread.std(),
                    'spread_min': period_spread.min(),
                    'spread_max': period_spread.max()
                })
        except:
            continue
    
    return pd.DataFrame(periods)


def find_best_pairs(df, correlation_type='positive', min_correlation=0.5, 
                    max_cv=0.4, lookback=100, include_beta=True):
    """Encuentra los mejores pares usando criterios estadísticos incluyendo beta"""
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
            
            mean_corr = p1.corr(p2)
            
            if correlation_type == 'positive':
                if mean_corr < min_correlation:
                    continue
            else:
                if mean_corr > -min_correlation:
                    continue
            
            # Calcular Beta
            beta_ols, alpha_ols = calculate_beta_ols(p1, p2)
            beta_theil, alpha_theil = calculate_beta_theil_sen(p1, p2)
            
            # Spread con beta=1 (simple)
            spread_simple = calculate_log_ratio_spread(p1, p2)
            
            # Spread con beta estimado
            spread_beta = calculate_spread_with_beta(p1, p2, beta_ols)
            
            # Tests estadísticos para ambos spreads
            adf_simple = adf_test(spread_simple)
            adf_beta = adf_test(spread_beta)
            
            hurst_simple = calculate_hurst_exponent(spread_simple.dropna())
            hurst_beta = calculate_hurst_exponent(spread_beta.dropna())
            
            half_life_simple = calculate_half_life(spread_simple)
            half_life_beta = calculate_half_life(spread_beta)
            
            coint_result = test_cointegration(p1, p2)
            
            # Rolling correlation stability
            corr_rolling = calculate_rolling_correlation(df, asset1, asset2, window=lookback)
            corr_series = pd.Series(corr_rolling['correlation'].values, index=corr_rolling['date'])
            stability = calculate_correlation_stability(corr_series)
            
            if stability['mean_cv'] > max_cv:
                continue
            
            spread_vol = calculate_spread_volatility(spread_beta)
            
            # SCORE mejorado
            score = 0
            
            # Estabilidad de correlación
            if stability['mean_cv'] < 0.15:
                score += 30
            elif stability['mean_cv'] < 0.25:
                score += 20
            elif stability['mean_cv'] < 0.35:
                score += 12
            else:
                score += 5
            
            # Mean Reversion (usar el mejor entre simple y beta)
            best_hurst = min(hurst_simple, hurst_beta) if not np.isnan(hurst_beta) else hurst_simple
            if best_hurst < 0.35:
                score += 25
            elif best_hurst < 0.45:
                score += 18
            elif best_hurst < 0.5:
                score += 10
            
            # Estacionariedad (bonus si spread con beta es más estacionario)
            if adf_beta['stationary']:
                if adf_beta['pvalue'] < 0.01:
                    score += 20
                elif adf_beta['pvalue'] < 0.05:
                    score += 15
            elif adf_simple['stationary']:
                if adf_simple['pvalue'] < 0.01:
                    score += 15
                elif adf_simple['pvalue'] < 0.05:
                    score += 10
            
            # Cointegración
            if coint_result['cointegrated']:
                if coint_result['pvalue'] < 0.01:
                    score += 15
                elif coint_result['pvalue'] < 0.05:
                    score += 10
            
            # Beta cercano a 1 (simplicidad)
            if 0.85 <= abs(beta_ols) <= 1.15:
                score += 5  # Bonus por simplicidad
            
            # Beta estable (OLS y Theil-Sen similares)
            beta_divergence = abs(beta_ols - beta_theil) if not np.isnan(beta_theil) else 0
            if beta_divergence < 0.1:
                score += 5  # Beta estable
            
            if spread_vol > spread_beta.std() * 2:
                score *= 0.8
            
            if not np.isnan(half_life_beta) and half_life_beta > 100:
                score *= 0.9
            
            returns1 = np.log(p1 / p1.shift(1)).dropna()
            returns2 = np.log(p2 / p2.shift(1)).dropna()
            cond_corr = calculate_conditional_correlation(returns1, returns2)
            
            years_data = (common_idx[-1] - common_idx[0]).days / 365.25
            
            candidates.append({
                'asset1': asset1,
                'asset2': asset2,
                'score': score,
                'mean_correlation': mean_corr,
                'corr_stability_cv': stability['mean_cv'],
                # Beta info
                'beta_ols': beta_ols,
                'beta_theil': beta_theil,
                'beta_divergence': beta_divergence,
                'beta_distance_from_1': abs(beta_ols - 1),
                # Hurst
                'hurst_simple': hurst_simple,
                'hurst_beta': hurst_beta,
                # Half-life
                'half_life_simple': half_life_simple,
                'half_life_beta': half_life_beta,
                # Estacionariedad
                'adf_pvalue_simple': adf_simple['pvalue'],
                'adf_pvalue_beta': adf_beta['pvalue'],
                'stationary_simple': adf_simple['stationary'],
                'stationary_beta': adf_beta['stationary'],
                # Cointegración
                'cointegrated': coint_result['cointegrated'],
                'coint_pvalue': coint_result['pvalue'],
                # Volatilidad
                'spread_volatility': spread_vol,
                # Correlación condicional
                'corr_positive_markets': cond_corr['positive_markets'],
                'corr_negative_markets': cond_corr['negative_markets'],
                'corr_high_volatility': cond_corr['high_volatility'],
                # Metadata
                'years_data': years_data,
                'total_days': len(common_idx)
            })
    
    progress_bar.empty()
    status_text.empty()
    
    if len(candidates) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(candidates).sort_values('score', ascending=False)

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def plot_rolling_correlation(corr_df, asset1_name, asset2_name):
    """Gráfico de correlación móvil"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=corr_df['date'],
        y=corr_df['correlation'],
        mode='lines',
        name=f'{asset1_name} vs {asset2_name}',
        line=dict(color='#3b82f6', width=2),
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
        height=400,
        showlegend=True
    )
    
    return fig


def plot_all_correlations(corr_df, asset1_name, asset2_name):
    """Gráfico comparativo de Pearson, Spearman, Kendall"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=corr_df['date'],
        y=corr_df['pearson'],
        mode='lines',
        name='Pearson (lineal)',
        line=dict(color='#3b82f6', width=2),
    ))
    
    fig.add_trace(go.Scatter(
        x=corr_df['date'],
        y=corr_df['spearman'],
        mode='lines',
        name='Spearman (rank)',
        line=dict(color='#10b981', width=2),
    ))
    
    fig.add_trace(go.Scatter(
        x=corr_df['date'],
        y=corr_df['kendall'],
        mode='lines',
        name='Kendall (concordancia)',
        line=dict(color='#f59e0b', width=2),
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#666666")
    
    fig.update_layout(
        title=f'Comparación de Correlaciones: {asset1_name} vs {asset2_name}',
        xaxis_title='Fecha',
        yaxis_title='Correlación',
        yaxis=dict(range=[-1, 1]),
        template='plotly_dark',
        hovermode='x unified',
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_rolling_beta(beta_df, asset1_name, asset2_name, method='OLS'):
    """Gráfico de Beta rolling"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=beta_df['date'],
        y=beta_df['beta'],
        mode='lines',
        name=f'Beta ({method})',
        line=dict(color='#8b5cf6', width=2),
        hovertemplate='%{x}<br>Beta: %{y:.4f}<extra></extra>'
    ))
    
    # Línea de referencia en beta = 1
    fig.add_hline(y=1, line_dash="dash", line_color="#ef4444", 
                  annotation_text="β = 1", annotation_position="right")
    
    # Banda de confianza alrededor de 1
    fig.add_hrect(y0=0.85, y1=1.15, fillcolor="#10b981", opacity=0.1, line_width=0,
                  annotation_text="Zona β ≈ 1", annotation_position="top left")
    
    fig.update_layout(
        title=f'Rolling Beta ({method}): {asset1_name} vs {asset2_name}',
        xaxis_title='Fecha',
        yaxis_title='Beta (Hedge Ratio)',
        template='plotly_dark',
        hovermode='x unified',
        height=400,
    )
    
    return fig


def plot_beta_comparison(beta_ols_df, beta_theil_df, asset1_name, asset2_name):
    """Comparación de Beta OLS vs Theil-Sen"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=beta_ols_df['date'],
        y=beta_ols_df['beta'],
        mode='lines',
        name='Beta OLS',
        line=dict(color='#3b82f6', width=2),
    ))
    
    fig.add_trace(go.Scatter(
        x=beta_theil_df['date'],
        y=beta_theil_df['beta'],
        mode='lines',
        name='Beta Theil-Sen',
        line=dict(color='#f59e0b', width=2),
    ))
    
    fig.add_hline(y=1, line_dash="dash", line_color="#ef4444", opacity=0.5)
    
    fig.update_layout(
        title=f'Comparación Beta OLS vs Theil-Sen: {asset1_name} vs {asset2_name}',
        xaxis_title='Fecha',
        yaxis_title='Beta',
        template='plotly_dark',
        hovermode='x unified',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_spread_comparison(prices1, prices2, beta, asset1_name, asset2_name):
    """Comparación de spread con beta=1 vs beta estimado"""
    spread_simple = calculate_log_ratio_spread(prices1, prices2)
    spread_beta = calculate_spread_with_beta(prices1, prices2, beta)
    
    # Normalizar para comparar
    zscore_simple = (spread_simple - spread_simple.mean()) / spread_simple.std()
    zscore_beta = (spread_beta - spread_beta.mean()) / spread_beta.std()
    
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=('Spread (β=1) vs Spread (β estimado)', 'Z-Score Comparación'),
                        vertical_spacing=0.12)
    
    # Spreads
    fig.add_trace(go.Scatter(
        x=spread_simple.index,
        y=spread_simple,
        mode='lines',
        name='Spread (β=1)',
        line=dict(color='#3b82f6', width=1.5),
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=spread_beta.index,
        y=spread_beta,
        mode='lines',
        name=f'Spread (β={beta:.3f})',
        line=dict(color='#10b981', width=1.5),
    ), row=1, col=1)
    
    # Z-Scores
    fig.add_trace(go.Scatter(
        x=zscore_simple.index,
        y=zscore_simple,
        mode='lines',
        name='Z-Score (β=1)',
        line=dict(color='#3b82f6', width=1.5),
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=zscore_beta.index,
        y=zscore_beta,
        mode='lines',
        name=f'Z-Score (β={beta:.3f})',
        line=dict(color='#10b981', width=1.5),
    ), row=2, col=1)
    
    # Líneas de referencia para Z-Score
    fig.add_hline(y=2, line_dash="dot", line_color="#f59e0b", opacity=0.5, row=2, col=1)
    fig.add_hline(y=-2, line_dash="dot", line_color="#f59e0b", opacity=0.5, row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", row=2, col=1)
    
    fig.update_layout(
        title=f'Comparación de Spreads: {asset1_name} vs {asset2_name}',
        template='plotly_dark',
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_beta_distribution(beta_df):
    """Histograma de distribución del beta"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=beta_df['beta'].dropna(),
        nbinsx=50,
        name='Distribución Beta',
        marker_color='#8b5cf6',
        opacity=0.7
    ))
    
    # Línea vertical en beta = 1
    mean_beta = beta_df['beta'].mean()
    fig.add_vline(x=1, line_dash="dash", line_color="#ef4444", 
                  annotation_text="β = 1", annotation_position="top")
    fig.add_vline(x=mean_beta, line_dash="solid", line_color="#10b981", 
                  annotation_text=f"Media: {mean_beta:.3f}", annotation_position="top")
    
    fig.update_layout(
        title='Distribución Histórica del Beta',
        xaxis_title='Beta',
        yaxis_title='Frecuencia',
        template='plotly_dark',
        height=400,
    )
    
    return fig


def plot_price_comparison(df, asset1, asset2, asset1_name, asset2_name):
    """Gráfico comparativo de precios"""
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


def plot_multiple_rolling_correlations(df, pairs_list, window=10):
    """Múltiples gráficos de correlación"""
    fig = make_subplots(
        rows=(len(pairs_list) + 1) // 2, 
        cols=2,
        subplot_titles=[f"{ASSETS[p['asset1']]['label']} vs {ASSETS[p['asset2']]['label']}" 
                       for p in pairs_list],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    for idx, pair in enumerate(pairs_list):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        corr_df = calculate_rolling_correlation(df, pair['asset1'], pair['asset2'], window=window, step=1)
        
        fig.add_trace(
            go.Scatter(
                x=corr_df['date'],
                y=corr_df['correlation'],
                mode='lines',
                name=f"{ASSETS[pair['asset1']]['label'][:10]} vs {ASSETS[pair['asset2']]['label'][:10]}",
                line=dict(width=1.5),
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="#666666", opacity=0.5, row=row, col=col)
        fig.add_hline(y=0.5, line_dash="dot", line_color="#10b981", opacity=0.3, row=row, col=col)
        fig.add_hline(y=-0.5, line_dash="dot", line_color="#ef4444", opacity=0.3, row=row, col=col)
        
        fig.update_yaxes(range=[-1, 1], row=row, col=col)
    
    fig.update_layout(
        height=300 * ((len(pairs_list) + 1) // 2),
        template='plotly_dark',
        showlegend=False
    )
    
    return fig


def plot_seasonality_monthly(monthly_data, title, ylabel):
    """Gráfico de patrones mensuales"""
    months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
              'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    fig = go.Figure()
    
    if isinstance(monthly_data, pd.DataFrame) and 'mean' in monthly_data.columns:
        fig.add_trace(go.Scatter(
            x=months,
            y=monthly_data['mean'],
            mode='lines+markers',
            name='Media',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=10)
        ))
        
        if 'std' in monthly_data.columns:
            fig.add_trace(go.Scatter(
                x=months + months[::-1],
                y=(monthly_data['mean'] + monthly_data['std']).tolist() + 
                  (monthly_data['mean'] - monthly_data['std']).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
    else:
        fig.add_trace(go.Bar(
            x=months,
            y=monthly_data.values,
            marker_color='#3b82f6'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Mes',
        yaxis_title=ylabel,
        template='plotly_dark',
        height=400,
        hovermode='x'
    )
    
    return fig


def plot_seasonality_quarterly(quarterly_data, title, ylabel):
    """Gráfico de patrones trimestrales"""
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    
    fig = go.Figure()
    
    if isinstance(quarterly_data, pd.DataFrame) and 'mean' in quarterly_data.columns:
        fig.add_trace(go.Bar(
            x=quarters,
            y=quarterly_data['mean'],
            name='Media',
            marker_color=['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
        ))
    else:
        fig.add_trace(go.Bar(
            x=quarters,
            y=quarterly_data.values,
            marker_color=['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Trimestre',
        yaxis_title=ylabel,
        template='plotly_dark',
        height=400
    )
    
    return fig


def plot_seasonality_yearly(yearly_data, title, ylabel):
    """Gráfico de evolución anual"""
    fig = go.Figure()
    
    if isinstance(yearly_data, pd.DataFrame) and 'mean' in yearly_data.columns:
        fig.add_trace(go.Scatter(
            x=yearly_data.index,
            y=yearly_data['mean'],
            mode='lines+markers',
            name='Media',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=10)
        ))
        
        if 'std' in yearly_data.columns:
            fig.add_trace(go.Scatter(
                x=yearly_data.index.tolist() + yearly_data.index.tolist()[::-1],
                y=(yearly_data['mean'] + yearly_data['std']).tolist() + 
                  (yearly_data['mean'] - yearly_data['std']).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
    else:
        fig.add_trace(go.Bar(
            x=yearly_data.index,
            y=yearly_data.values,
            marker_color='#3b82f6'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Año',
        yaxis_title=ylabel,
        template='plotly_dark',
        height=400
    )
    
    return fig


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

st.title("📊 Pairs Trading - Beta & Correlation Analysis")
st.markdown("**Análisis de Beta, correlaciones múltiples y estacionalidad para pares de activos**")
st.info("📊 **10 años de historia** | 🔢 **Cálculo de Beta (OLS/Theil-Sen)** | 📈 **Pearson/Spearman/Kendall**")

# ============================================================================
# SIDEBAR - GESTIÓN DE DATOS
# ============================================================================

st.sidebar.header("💾 Gestión de Datos")

cache_info = get_cache_info()

if cache_info:
    st.sidebar.success("✅ Datos en cache")
    st.sidebar.metric("Última actualización", cache_info['last_update'].strftime('%Y-%m-%d %H:%M'))
    st.sidebar.metric("Total activos", cache_info['total_assets'])
    st.sidebar.metric("Período histórico", f"{cache_info['date_range']['start']} → {cache_info['date_range']['end']}")
    
    days_old = (datetime.now() - cache_info['last_update']).days
    if days_old > 0:
        st.sidebar.warning(f"⏰ Datos de hace {days_old} días")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Actualizar", key='btn_update_data'):
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
        if st.button("🗑️ Borrar", key='btn_delete_cache'):
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
    
    if st.sidebar.button("📥 Descargar Datos (10 años)", type="primary", key='btn_download_assets'):
        with st.spinner(f"Descargando {len(ASSETS)} activos..."):
            all_data, metadata = download_all_assets(delay=3, start_date='2015-01-01')
        
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
    ### 👋 Bienvenido al Análisis de Beta y Correlaciones
    
    **Activos disponibles ({len(ASSETS)}):**
    - 📊 {len([a for a in ASSETS.values() if a['category'] == 'Indices'])} Índices globales
    - 💱 {len([a for a in ASSETS.values() if a['category'] == 'Forex'])} Pares de divisas (incluye DXY)
    - 🏆 {len([a for a in ASSETS.values() if a['category'] == 'Commodities'])} Commodities
    - ₿ {len([a for a in ASSETS.values() if a['category'] == 'Crypto'])} Criptomonedas
    
    **Nuevas Características:**
    - 🔢 **Cálculo de Beta** (OLS y Theil-Sen)
    - 📊 **Spread con Beta**: ln(P1) - β·ln(P2)
    - 📈 **Múltiples Correlaciones**: Pearson, Spearman, Kendall
    - 🎯 **Análisis de Estabilidad del Beta**
    - 📅 **10 años de datos históricos**
    
    **Para comenzar:**
    1. Presiona "📥 Descargar Datos (10 años)"
    """)
    st.stop()

# ============================================================================
# PARÁMETROS
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parámetros de Análisis")

min_correlation = st.sidebar.slider("Correlación Mínima", 0.3, 0.9, 0.5, 0.05)
max_cv = st.sidebar.slider("Máx. CV (estabilidad)", 0.2, 0.8, 0.4, 0.05)
rolling_window = st.sidebar.slider("Window Rolling Correlation", 10, 200, 30, 5)
beta_window = st.sidebar.slider("Window Beta Rolling", 50, 300, 100, 10)
lookback_analysis = st.sidebar.slider("Lookback para Análisis", 50, 200, 100, 10)

# Crear DataFrame
df_all_prices = merge_asset_data(st.session_state.all_asset_data)

if df_all_prices.empty:
    st.error("No hay datos suficientes")
    st.stop()

years_available = (df_all_prices.index[-1] - df_all_prices.index[0]).days / 365.25

st.success(f"✅ {len(df_all_prices)} días ({years_available:.1f} años) | {df_all_prices.index[0].date()} → {df_all_prices.index[-1].date()}")
st.info(f"📊 {len(df_all_prices.columns)} activos disponibles | 🔢 Beta Analysis | 📈 Multi-Correlation")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Búsqueda de Pares",
    "📊 Análisis Individual",
    "🔢 Análisis de Beta",
    "📈 Comparación de Pares",
    "📅 Estacionalidad"
])

# ============================================================================
# TAB 1: BÚSQUEDA
# ============================================================================

with tab1:
    st.header("🔍 Búsqueda de Mejores Pares")
    st.info("""
    **Criterios Estadísticos (Mejorados con Beta):**
    - ✅ Estabilidad de Correlación (30 pts)
    - ✅ Mean Reversion - Hurst < 0.5 (25 pts)
    - ✅ Estacionariedad - ADF test (20 pts) - **Bonus si spread con β es más estacionario**
    - ✅ Cointegración (15 pts)
    - ✅ **Beta cercano a 1** (5 pts) - Simplicidad
    - ✅ **Beta estable** (5 pts) - OLS ≈ Theil-Sen
    """)
    
    if st.button("🚀 Buscar Pares", type="primary"):
        
        st.markdown("### 📈 Correlación POSITIVA...")
        with st.spinner("Analizando con cálculo de Beta..."):
            positive_pairs = find_best_pairs(
                df_all_prices,
                correlation_type='positive',
                min_correlation=min_correlation,
                max_cv=max_cv,
                lookback=lookback_analysis,
                include_beta=True
            )
        
        st.markdown("### 📉 Correlación NEGATIVA...")
        with st.spinner("Analizando con cálculo de Beta..."):
            negative_pairs = find_best_pairs(
                df_all_prices,
                correlation_type='negative',
                min_correlation=min_correlation,
                max_cv=max_cv,
                lookback=lookback_analysis,
                include_beta=True
            )
        
        st.session_state.positive_pairs = positive_pairs
        st.session_state.negative_pairs = negative_pairs
        st.success("✅ Búsqueda completada!")
    
    if 'positive_pairs' in st.session_state and 'negative_pairs' in st.session_state:
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Top 20 - Correlación POSITIVA")
            
            if len(st.session_state.positive_pairs) > 0:
                display_pos = st.session_state.positive_pairs.head(20).copy()
                display_pos['Activo 1'] = display_pos['asset1'].apply(lambda x: ASSETS[x]['label'])
                display_pos['Activo 2'] = display_pos['asset2'].apply(lambda x: ASSETS[x]['label'])
                display_pos['✓ Estacionario'] = display_pos['stationary_beta'].apply(lambda x: '✅' if x else '❌')
                display_pos['✓ Cointegrado'] = display_pos['cointegrated'].apply(lambda x: '✅' if x else '❌')
                
                table_pos = display_pos[['Activo 1', 'Activo 2', 'score', 'mean_correlation', 
                                         'beta_ols', 'beta_theil', 'corr_stability_cv', 
                                         'hurst_beta', 'half_life_beta', 'years_data',
                                         '✓ Estacionario', '✓ Cointegrado']].rename(columns={
                    'score': 'Score',
                    'mean_correlation': 'Corr',
                    'beta_ols': 'β OLS',
                    'beta_theil': 'β Theil',
                    'corr_stability_cv': 'CV',
                    'hurst_beta': 'Hurst',
                    'half_life_beta': 'HL',
                    'years_data': 'Años'
                })
                
                st.dataframe(
                    table_pos.style.format({
                        'Score': '{:.1f}',
                        'Corr': '{:.3f}',
                        'β OLS': '{:.3f}',
                        'β Theil': '{:.3f}',
                        'CV': '{:.3f}',
                        'Hurst': '{:.3f}',
                        'HL': '{:.1f}',
                        'Años': '{:.1f}'
                    }),
                    height=600
                )
                
                st.metric("Total pares", len(st.session_state.positive_pairs))
                
                pair_options_pos = [f"{row['Activo 1']} / {row['Activo 2']}" 
                                   for _, row in display_pos.iterrows()]
                
                selected_pos_pair = st.selectbox(
                    "Seleccionar para análisis",
                    options=pair_options_pos,
                    key='select_pos_pair'
                )
                
                if st.button("📊 Analizar", key='btn_analyze_pos'):
                    idx = pair_options_pos.index(selected_pos_pair)
                    selected_row = display_pos.iloc[idx]
                    st.session_state.selected_asset1 = selected_row['asset1']
                    st.session_state.selected_asset2 = selected_row['asset2']
                    st.session_state.run_analysis = True
                    st.success(f"✅ {selected_pos_pair}")
                    st.info("👉 Ve a 'Análisis Individual' o 'Análisis de Beta'")
            else:
                st.warning("No se encontraron pares")
        
        with col2:
            st.markdown("### 📉 Top 20 - Correlación NEGATIVA")
            
            if len(st.session_state.negative_pairs) > 0:
                display_neg = st.session_state.negative_pairs.head(20).copy()
                display_neg['Activo 1'] = display_neg['asset1'].apply(lambda x: ASSETS[x]['label'])
                display_neg['Activo 2'] = display_neg['asset2'].apply(lambda x: ASSETS[x]['label'])
                display_neg['✓ Estacionario'] = display_neg['stationary_beta'].apply(lambda x: '✅' if x else '❌')
                display_neg['✓ Cointegrado'] = display_neg['cointegrated'].apply(lambda x: '✅' if x else '❌')
                
                table_neg = display_neg[['Activo 1', 'Activo 2', 'score', 'mean_correlation', 
                                         'beta_ols', 'beta_theil', 'corr_stability_cv', 
                                         'hurst_beta', 'half_life_beta', 'years_data',
                                         '✓ Estacionario', '✓ Cointegrado']].rename(columns={
                    'score': 'Score',
                    'mean_correlation': 'Corr',
                    'beta_ols': 'β OLS',
                    'beta_theil': 'β Theil',
                    'corr_stability_cv': 'CV',
                    'hurst_beta': 'Hurst',
                    'half_life_beta': 'HL',
                    'years_data': 'Años'
                })
                
                st.dataframe(
                    table_neg.style.format({
                        'Score': '{:.1f}',
                        'Corr': '{:.3f}',
                        'β OLS': '{:.3f}',
                        'β Theil': '{:.3f}',
                        'CV': '{:.3f}',
                        'Hurst': '{:.3f}',
                        'HL': '{:.1f}',
                        'Años': '{:.1f}'
                    }),
                    height=600
                )
                
                st.metric("Total pares", len(st.session_state.negative_pairs))
                
                pair_options_neg = [f"{row['Activo 1']} / {row['Activo 2']}" 
                                   for _, row in display_neg.iterrows()]
                
                selected_neg_pair = st.selectbox(
                    "Seleccionar para análisis",
                    options=pair_options_neg,
                    key='select_neg_pair'
                )
                
                if st.button("📊 Analizar", key='btn_analyze_neg'):
                    idx = pair_options_neg.index(selected_neg_pair)
                    selected_row = display_neg.iloc[idx]
                    st.session_state.selected_asset1 = selected_row['asset1']
                    st.session_state.selected_asset2 = selected_row['asset2']
                    st.session_state.run_analysis = True
                    st.success(f"✅ {selected_neg_pair}")
                    st.info("👉 Ve a 'Análisis Individual' o 'Análisis de Beta'")
            else:
                st.warning("No se encontraron pares")

# ============================================================================
# TAB 2: ANÁLISIS INDIVIDUAL
# ============================================================================

with tab2:
    st.header("📊 Análisis Individual de Par")
    
    available_assets = list(st.session_state.all_asset_data.keys())
    
    default_asset1 = st.session_state.get('selected_asset1', available_assets[0])
    default_asset2 = st.session_state.get('selected_asset2', available_assets[1] if len(available_assets) > 1 else available_assets[0])
    
    if default_asset2 == default_asset1 and len(available_assets) > 1:
        default_asset2 = available_assets[1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        asset1 = st.selectbox(
            "Activo 1",
            options=available_assets,
            index=available_assets.index(default_asset1) if default_asset1 in available_assets else 0,
            format_func=lambda x: ASSETS[x]['label'],
            key='detail_asset1'
        )
    
    with col2:
        asset2_options = [a for a in available_assets if a != asset1]
        asset2 = st.selectbox(
            "Activo 2",
            options=asset2_options,
            index=asset2_options.index(default_asset2) if default_asset2 in asset2_options else 0,
            format_func=lambda x: ASSETS[x]['label'],
            key='detail_asset2'
        )
    
    if st.button("🔄 Analizar", type="primary", key='btn_analyze_individual'):
        st.session_state.run_analysis = True
    
    if st.session_state.get('run_analysis', False):
        
        prices1 = df_all_prices[asset1]
        prices2 = df_all_prices[asset2]
        
        years_data = (prices1.index[-1] - prices1.index[0]).days / 365.25
        
        st.markdown("### 📅 Período")
        col1, col2, col3 = st.columns(3)
        col1.metric("Inicio", prices1.index[0].strftime('%Y-%m-%d'))
        col2.metric("Fin", prices1.index[-1].strftime('%Y-%m-%d'))
        col3.metric("Años", f"{years_data:.1f}")
        
        # Múltiples Correlaciones
        st.markdown("### 📈 Comparación de Correlaciones (Pearson vs Spearman vs Kendall)")
        
        all_corr_df = calculate_all_correlations(prices1, prices2, window=rolling_window)
        st.plotly_chart(
            plot_all_correlations(all_corr_df, ASSETS[asset1]['label'], ASSETS[asset2]['label']),
            use_container_width=True
        )
        
        # Análisis de divergencia
        divergence = analyze_correlation_divergence(all_corr_df)
        if divergence:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Pearson (media)", f"{divergence['pearson_mean']:.3f}")
            col2.metric("Spearman (media)", f"{divergence['spearman_mean']:.3f}")
            col3.metric("Kendall (media)", f"{divergence['kendall_mean']:.3f}")
            col4.metric("Recomendación", divergence['recommendation'],
                       delta="Lineal ✅" if divergence['is_linear'] else "No Lineal ⚠️")
        
        # Rolling Correlation (Pearson tradicional)
        st.markdown("### 📈 Rolling Correlation (Pearson)")
        corr_df = calculate_rolling_correlation(df_all_prices, asset1, asset2, window=rolling_window, step=1)
        st.plotly_chart(
            plot_rolling_correlation(corr_df, ASSETS[asset1]['label'], ASSETS[asset2]['label']),
            use_container_width=True
        )
        
        # Métricas
        st.markdown("### 📊 Métricas")
        
        col1, col2, col3, col4 = st.columns(4)
        
        current_corr = corr_df['correlation'].iloc[-1]
        mean_corr = corr_df['correlation'].mean()
        max_corr = corr_df['correlation'].max()
        min_corr = corr_df['correlation'].min()
        
        col1.metric("Actual", f"{current_corr:.4f}")
        col2.metric("Media", f"{mean_corr:.4f}")
        col3.metric("Máx", f"{max_corr:.4f}")
        col4.metric("Mín", f"{min_corr:.4f}")
        
        # Precios
        st.markdown("### 📉 Comparación de Precios")
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
        col3.metric("% Fuerte + (>0.5)", f"{strong_pos/total*100:.1f}%")
        col4.metric("% Fuerte - (<-0.5)", f"{strong_neg/total*100:.1f}%")
        
        # Períodos históricos
        st.markdown("### 📅 Períodos Históricos")
        
        historical_df = calculate_historical_periods(df_all_prices, asset1, asset2)
        
        if len(historical_df) > 0:
            st.dataframe(
                historical_df.style.format({
                    'correlation': '{:.3f}',
                    'beta_ols': '{:.3f}',
                    'spread_mean': '{:.4f}',
                    'spread_std': '{:.4f}',
                    'spread_min': '{:.4f}',
                    'spread_max': '{:.4f}'
                }),
                use_container_width=True
            )
        
        st.session_state.run_analysis = False
    
    else:
        st.info("👆 Selecciona activos y presiona **Analizar**")

# ============================================================================
# TAB 3: ANÁLISIS DE BETA
# ============================================================================

with tab3:
    st.header("🔢 Análisis de Beta (Hedge Ratio)")
    
    st.info("""
    **¿Qué es el Beta?**
    
    El Beta (β) es el **hedge ratio** que determina cuántas unidades del Activo 2 necesitas 
    para replicar el Activo 1:
    
    **Spread = ln(P₁) - β × ln(P₂)**
    
    - **β = 1**: Relación 1:1 (simplificación)
    - **β > 1**: Activo 2 es menos volátil que Activo 1
    - **β < 1**: Activo 2 es más volátil que Activo 1
    
    **Métodos:**
    - **OLS**: Mínimos cuadrados ordinarios (sensible a outliers)
    - **Theil-Sen**: Robusto a outliers (usa mediana de pendientes)
    """)
    
    available_assets = list(st.session_state.all_asset_data.keys())
    
    default_beta_asset1 = st.session_state.get('selected_asset1', available_assets[0])
    default_beta_asset2 = st.session_state.get('selected_asset2', available_assets[1] if len(available_assets) > 1 else available_assets[0])
    
    if default_beta_asset2 == default_beta_asset1 and len(available_assets) > 1:
        default_beta_asset2 = available_assets[1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        beta_asset1 = st.selectbox(
            "Activo 1 (Y)",
            options=available_assets,
            index=available_assets.index(default_beta_asset1) if default_beta_asset1 in available_assets else 0,
            format_func=lambda x: ASSETS[x]['label'],
            key='beta_asset1'
        )
    
    with col2:
        beta_asset2_options = [a for a in available_assets if a != beta_asset1]
        beta_asset2 = st.selectbox(
            "Activo 2 (X)",
            options=beta_asset2_options,
            index=beta_asset2_options.index(default_beta_asset2) if default_beta_asset2 in beta_asset2_options else 0,
            format_func=lambda x: ASSETS[x]['label'],
            key='beta_asset2'
        )
    
    if st.button("🔄 Calcular Beta", type="primary", key='btn_calc_beta'):
        
        prices1 = df_all_prices[beta_asset1]
        prices2 = df_all_prices[beta_asset2]
        
        # Beta estático
        st.markdown("### 📊 Beta Estático (Todo el período)")
        
        beta_ols, alpha_ols = calculate_beta_ols(prices1, prices2)
        beta_theil, alpha_theil = calculate_beta_theil_sen(prices1, prices2)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("β OLS", f"{beta_ols:.4f}")
        col2.metric("β Theil-Sen", f"{beta_theil:.4f}")
        col3.metric("Divergencia", f"{abs(beta_ols - beta_theil):.4f}",
                   delta="Estable ✅" if abs(beta_ols - beta_theil) < 0.1 else "Inestable ⚠️")
        col4.metric("Distancia de β=1", f"{abs(beta_ols - 1):.4f}",
                   delta="Simple ✅" if abs(beta_ols - 1) < 0.15 else "Usar β")
        
        st.markdown("---")
        
        # Beta Rolling
        st.markdown("### 📈 Beta Rolling (Dinámico)")
        
        beta_ols_rolling = calculate_rolling_beta(prices1, prices2, window=beta_window, method='ols')
        beta_theil_rolling = calculate_rolling_beta(prices1, prices2, window=beta_window, method='theil_sen')
        
        # Gráfico comparativo
        st.plotly_chart(
            plot_beta_comparison(beta_ols_rolling, beta_theil_rolling, 
                                ASSETS[beta_asset1]['label'], ASSETS[beta_asset2]['label']),
            use_container_width=True
        )
        
        # Análisis de estabilidad
        st.markdown("### 🎯 Estabilidad del Beta")
        
        stability_ols = analyze_beta_stability(beta_ols_rolling)
        stability_theil = analyze_beta_stability(beta_theil_rolling)
        
        if stability_ols and stability_theil:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### OLS")
                st.metric("Media", f"{stability_ols['mean']:.4f}")
                st.metric("Desv. Std", f"{stability_ols['std']:.4f}")
                st.metric("CV", f"{stability_ols['cv']:.4f}")
                st.metric("Rango", f"{stability_ols['range']:.4f}")
                st.metric("% > 1", f"{stability_ols['pct_above_1']:.1f}%")
                st.metric("% < 1", f"{stability_ols['pct_below_1']:.1f}%")
            
            with col2:
                st.markdown("#### Theil-Sen")
                st.metric("Media", f"{stability_theil['mean']:.4f}")
                st.metric("Desv. Std", f"{stability_theil['std']:.4f}")
                st.metric("CV", f"{stability_theil['cv']:.4f}")
                st.metric("Rango", f"{stability_theil['range']:.4f}")
                st.metric("% > 1", f"{stability_theil['pct_above_1']:.1f}%")
                st.metric("% < 1", f"{stability_theil['pct_below_1']:.1f}%")
        
        # Distribución del Beta
        st.markdown("### 📊 Distribución del Beta")
        st.plotly_chart(plot_beta_distribution(beta_ols_rolling), use_container_width=True)
        
        st.markdown("---")
        
        # Comparación de Spreads
        st.markdown("### 🔄 Comparación de Spreads: β=1 vs β estimado")
        
        st.plotly_chart(
            plot_spread_comparison(prices1, prices2, beta_ols,
                                  ASSETS[beta_asset1]['label'], ASSETS[beta_asset2]['label']),
            use_container_width=True
        )
        
        # Tests de estacionariedad comparativos
        st.markdown("### 📋 Tests de Estacionariedad (Spread)")
        
        spread_simple = calculate_log_ratio_spread(prices1, prices2)
        spread_beta = calculate_spread_with_beta(prices1, prices2, beta_ols)
        
        adf_simple = adf_test(spread_simple)
        adf_beta = adf_test(spread_beta)
        
        hurst_simple = calculate_hurst_exponent(spread_simple.dropna())
        hurst_beta = calculate_hurst_exponent(spread_beta.dropna())
        
        hl_simple = calculate_half_life(spread_simple)
        hl_beta = calculate_half_life(spread_beta)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Spread (β = 1)")
            st.metric("ADF p-value", f"{adf_simple['pvalue']:.4f}",
                     delta="Estacionario ✅" if adf_simple['stationary'] else "No Estacionario ❌")
            st.metric("Hurst", f"{hurst_simple:.4f}",
                     delta="Mean Reverting ✅" if hurst_simple < 0.5 else "Trending ⚠️")
            st.metric("Half-Life", f"{hl_simple:.1f} días" if not np.isnan(hl_simple) else "N/A")
        
        with col2:
            st.markdown(f"#### Spread (β = {beta_ols:.3f})")
            st.metric("ADF p-value", f"{adf_beta['pvalue']:.4f}",
                     delta="Estacionario ✅" if adf_beta['stationary'] else "No Estacionario ❌")
            st.metric("Hurst", f"{hurst_beta:.4f}",
                     delta="Mean Reverting ✅" if hurst_beta < 0.5 else "Trending ⚠️")
            st.metric("Half-Life", f"{hl_beta:.1f} días" if not np.isnan(hl_beta) else "N/A")
        
        # Recomendación
        st.markdown("---")
        st.markdown("### 💡 Recomendación")
        
        use_beta = (
            (adf_beta['pvalue'] < adf_simple['pvalue']) or
            (hurst_beta < hurst_simple) or
            (abs(beta_ols - 1) > 0.15)
        )
        
        if use_beta:
            st.success(f"""
            **✅ Usar Beta = {beta_ols:.3f}**
            
            Razones:
            - El spread con β estimado es {'más estacionario' if adf_beta['pvalue'] < adf_simple['pvalue'] else 'similar en estacionariedad'}
            - Hurst con β: {hurst_beta:.3f} vs sin β: {hurst_simple:.3f}
            - Beta está {'lejos' if abs(beta_ols - 1) > 0.15 else 'cerca'} de 1 (distancia: {abs(beta_ols - 1):.3f})
            
            **Fórmula para tu EA:**
            ```
            spread = ln(P1) - {beta_ols:.3f} × ln(P2)
            ```
            """)
        else:
            st.info(f"""
            **ℹ️ Puedes usar Beta = 1 (simplificación)**
            
            Razones:
            - Beta está cerca de 1 (valor: {beta_ols:.3f})
            - Diferencia de estacionariedad es mínima
            - Simplicidad sin pérdida significativa de calidad
            
            **Fórmula simplificada:**
            ```
            spread = ln(P1) - ln(P2)
            ```
            """)
    
    else:
        st.info("👆 Selecciona activos y presiona **Calcular Beta**")

# ============================================================================
# TAB 4: COMPARACIÓN
# ============================================================================

with tab4:
    st.header("📈 Comparación de Pares")
    
    if 'positive_pairs' in st.session_state and 'negative_pairs' in st.session_state:
        
        st.markdown("### 📈 Top 10 - Correlación Positiva")
        
        if len(st.session_state.positive_pairs) > 0:
            top_pos = st.session_state.positive_pairs.head(10).to_dict('records')
            
            with st.spinner("Generando gráficos..."):
                fig_pos = plot_multiple_rolling_correlations(df_all_prices, top_pos, window=rolling_window)
                st.plotly_chart(fig_pos, use_container_width=True)
        else:
            st.info("No hay pares")
        
        st.markdown("---")
        
        st.markdown("### 📉 Top 10 - Correlación Negativa")
        
        if len(st.session_state.negative_pairs) > 0:
            top_neg = st.session_state.negative_pairs.head(10).to_dict('records')
            
            with st.spinner("Generando gráficos..."):
                fig_neg = plot_multiple_rolling_correlations(df_all_prices, top_neg, window=rolling_window)
                st.plotly_chart(fig_neg, use_container_width=True)
        else:
            st.info("No hay pares")
        
        st.markdown("---")
        
        st.markdown("### 📊 Comparación de Métricas (con Beta)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(st.session_state.positive_pairs) > 0:
                st.markdown("#### Positivos: Score vs Beta")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.positive_pairs['score'],
                    y=st.session_state.positive_pairs['beta_ols'],
                    mode='markers',
                    marker=dict(
                        size=st.session_state.positive_pairs['years_data'] * 3,
                        color=st.session_state.positive_pairs['mean_correlation'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Corr")
                    ),
                    text=[f"{ASSETS[row['asset1']]['label']} / {ASSETS[row['asset2']]['label']}<br>β={row['beta_ols']:.3f}" 
                          for _, row in st.session_state.positive_pairs.iterrows()],
                    hovertemplate='<b>%{text}</b><br>Score: %{x:.1f}<br>Beta: %{y:.3f}<extra></extra>'
                ))
                
                fig.add_hline(y=1, line_dash="dash", line_color="#ef4444", opacity=0.5,
                             annotation_text="β=1")
                
                fig.update_layout(
                    title='Score vs Beta OLS',
                    xaxis_title='Score',
                    yaxis_title='Beta OLS',
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if len(st.session_state.negative_pairs) > 0:
                st.markdown("#### Negativos: Score vs Beta")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.negative_pairs['score'],
                    y=st.session_state.negative_pairs['beta_ols'],
                    mode='markers',
                    marker=dict(
                        size=st.session_state.negative_pairs['years_data'] * 3,
                        color=st.session_state.negative_pairs['mean_correlation'],
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title="Corr")
                    ),
                    text=[f"{ASSETS[row['asset1']]['label']} / {ASSETS[row['asset2']]['label']}<br>β={row['beta_ols']:.3f}" 
                          for _, row in st.session_state.negative_pairs.iterrows()],
                    hovertemplate='<b>%{text}</b><br>Score: %{x:.1f}<br>Beta: %{y:.3f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Score vs Beta OLS',
                    xaxis_title='Score',
                    yaxis_title='Beta OLS',
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Ejecuta la búsqueda primero")

# ============================================================================
# TAB 5: ESTACIONALIDAD
# ============================================================================

with tab5:
    st.header("📅 Análisis de Estacionalidad")
    st.info("Identifica patrones estacionales en correlación y spread")
    
    available_assets = list(st.session_state.all_asset_data.keys())
    
    default_season_asset1 = st.session_state.get('selected_asset1', available_assets[0])
    default_season_asset2 = st.session_state.get('selected_asset2', available_assets[1] if len(available_assets) > 1 else available_assets[0])
    
    if default_season_asset2 == default_season_asset1 and len(available_assets) > 1:
        default_season_asset2 = available_assets[1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        season_asset1 = st.selectbox(
            "Activo 1",
            options=available_assets,
            index=available_assets.index(default_season_asset1) if default_season_asset1 in available_assets else 0,
            format_func=lambda x: ASSETS[x]['label'],
            key='season_asset1'
        )
    
    with col2:
        season_asset2_options = [a for a in available_assets if a != season_asset1]
        season_asset2 = st.selectbox(
            "Activo 2",
            options=season_asset2_options,
            index=season_asset2_options.index(default_season_asset2) if default_season_asset2 in season_asset2_options else 0,
            format_func=lambda x: ASSETS[x]['label'],
            key='season_asset2'
        )
    
    if st.button("🔄 Analizar Estacionalidad", type="primary"):
        
        with st.spinner("Analizando..."):
            seasonality = analyze_seasonality(df_all_prices, season_asset1, season_asset2, lookback_analysis)
        
        st.success("✅ Completado")
        
        # Mensual
        st.markdown("### 📅 Análisis Mensual")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Correlación por Mes")
            fig_monthly_corr = plot_seasonality_monthly(
                seasonality['monthly_corr'],
                'Correlación Media por Mes',
                'Correlación'
            )
            st.plotly_chart(fig_monthly_corr, use_container_width=True)
        
        with col2:
            st.markdown("#### Volatilidad del Spread")
            fig_monthly_vol = plot_seasonality_monthly(
                seasonality['monthly_spread_vol'],
                'Volatilidad del Spread por Mes',
                'Volatilidad'
            )
            st.plotly_chart(fig_monthly_vol, use_container_width=True)
        
        # Tabla mensual
        st.markdown("#### 📊 Estadísticas Mensuales")
        
        monthly_stats = seasonality['monthly_corr'].copy()
        monthly_stats.index = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                              'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        
        st.dataframe(
            monthly_stats.style.format({
                'mean': '{:.3f}',
                'std': '{:.3f}',
                'min': '{:.3f}',
                'max': '{:.3f}'
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Trimestral
        st.markdown("### 📊 Análisis Trimestral")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Correlación por Trimestre")
            fig_quarterly_corr = plot_seasonality_quarterly(
                seasonality['quarterly_corr'],
                'Correlación por Trimestre',
                'Correlación'
            )
            st.plotly_chart(fig_quarterly_corr, use_container_width=True)
        
        with col2:
            st.markdown("#### Volatilidad del Spread")
            fig_quarterly_vol = plot_seasonality_quarterly(
                seasonality['quarterly_spread_vol'],
                'Volatilidad por Trimestre',
                'Volatilidad'
            )
            st.plotly_chart(fig_quarterly_vol, use_container_width=True)
        
        st.markdown("---")
        
        # Anual
        st.markdown("### 📈 Análisis Anual")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Correlación por Año")
            fig_yearly_corr = plot_seasonality_yearly(
                seasonality['yearly_corr'],
                'Correlación por Año',
                'Correlación'
            )
            st.plotly_chart(fig_yearly_corr, use_container_width=True)
        
        with col2:
            st.markdown("#### Volatilidad del Spread")
            fig_yearly_vol = plot_seasonality_yearly(
                seasonality['yearly_spread_vol'],
                'Volatilidad por Año',
                'Volatilidad'
            )
            st.plotly_chart(fig_yearly_vol, use_container_width=True)
        
        # Tabla anual
        st.markdown("#### 📊 Estadísticas Anuales")
        
        st.dataframe(
            seasonality['yearly_corr'].style.format({
                'mean': '{:.3f}',
                'std': '{:.3f}',
                'min': '{:.3f}',
                'max': '{:.3f}'
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Períodos históricos con Beta
        st.markdown("### 📅 Períodos Históricos (con Beta)")
        
        historical_df = calculate_historical_periods(df_all_prices, season_asset1, season_asset2)
        
        if len(historical_df) > 0:
            st.dataframe(
                historical_df.style.format({
                    'correlation': '{:.3f}',
                    'beta_ols': '{:.3f}',
                    'spread_mean': '{:.4f}',
                    'spread_std': '{:.4f}',
                    'spread_min': '{:.4f}',
                    'spread_max': '{:.4f}'
                }),
                use_container_width=True
            )
            
            # Gráfico de correlación y beta por período
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Correlación por Período', 'Beta por Período'))
            
            fig.add_trace(go.Bar(
                x=historical_df['period'],
                y=historical_df['correlation'],
                marker_color=['#10b981' if c > 0 else '#ef4444' for c in historical_df['correlation']],
                text=historical_df['correlation'].round(3),
                textposition='auto',
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Bar(
                x=historical_df['period'],
                y=historical_df['beta_ols'],
                marker_color='#8b5cf6',
                text=historical_df['beta_ols'].round(3),
                textposition='auto',
                showlegend=False
            ), row=1, col=2)
            
            fig.add_hline(y=1, line_dash="dash", line_color="#ef4444", opacity=0.5, row=1, col=2)
            
            fig.update_layout(
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Recomendaciones
        st.markdown("### 💡 Recomendaciones")
        
        best_months = seasonality['monthly_corr']['mean'].abs().nlargest(3)
        worst_months = seasonality['monthly_corr']['mean'].abs().nsmallest(3)
        
        months_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                       'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("**✅ Mejores Meses**")
            for month_num in best_months.index:
                st.write(f"- **{months_names[month_num-1]}**: {best_months[month_num]:.3f}")
        
        with col2:
            st.warning("**⚠️ Meses con Menor Correlación**")
            for month_num in worst_months.index:
                st.write(f"- **{months_names[month_num-1]}**: {worst_months[month_num]:.3f}")
    
    else:
        st.info("👆 Selecciona activos y presiona **Analizar Estacionalidad**")

# Footer
st.sidebar.markdown("---")
st.sidebar.header("📚 Guía")
st.sidebar.markdown("""
**Flujo:**
1. 🔍 Búsqueda de pares
2. 📊 Análisis individual
3. 🔢 **Análisis de Beta** (NUEVO)
4. 📈 Comparación múltiple
5. 📅 Estacionalidad

**Nuevas Funcionalidades:**
- 🔢 Beta OLS y Theil-Sen
- 📈 Pearson, Spearman, Kendall
- 🔄 Spread con β estimado
- 📊 Comparación β=1 vs β
- 🎯 Análisis de estabilidad

**10 años de datos**
**DXY incluido**
""")

st.sidebar.info("📊 Análisis estadístico con Beta")
