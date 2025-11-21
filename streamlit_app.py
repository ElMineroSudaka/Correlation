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

def calculate_rolling_correlation(df, asset1, asset2, window=30, step=1):
    """Calcula la correlación móvil entre dos activos con step diario"""
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
                           max_cv=0.4, lookback=100):
    """
    Encuentra los mejores pares usando SOLO criterios estadísticos fundamentales
    NO usa win rate ni cantidad de señales
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
            
            # Calcular spread y correlación
            spread = calculate_log_ratio_spread(p1, p2)
            corr = calculate_correlation(p1, p2, window=lookback)
            
            mean_corr = corr.mean()
            
            # Filtrar según tipo de correlación
            if correlation_type == 'positive':
                if mean_corr < min_correlation:
                    continue
            else:  # negative
                if mean_corr > -min_correlation:
                    continue
            
            # Tests estadísticos
            adf_result = adf_test(spread)
            hurst = calculate_hurst_exponent(spread.dropna())
            half_life = calculate_half_life(spread)
            coint_result = test_cointegration(p1, p2)
            
            # Estabilidad de correlación
            stability = calculate_correlation_stability(corr)
            
            if stability['mean_cv'] > max_cv:
                continue
            
            # Volatilidad del spread
            spread_vol = calculate_spread_volatility(spread)
            
            # SCORE BASADO EN PROPIEDADES ESTADÍSTICAS FUNDAMENTALES
            score = 0
            
            # 35 pts: Estabilidad de correlación (CRÍTICO)
            if stability['mean_cv'] < 0.15:
                score += 35
            elif stability['mean_cv'] < 0.25:
                score += 25
            elif stability['mean_cv'] < 0.35:
                score += 15
            else:
                score += 5
            
            # 30 pts: Mean Reversion (Hurst Exponent)
            if hurst < 0.35:
                score += 30
            elif hurst < 0.45:
                score += 20
            elif hurst < 0.5:
                score += 10
            else:
                score += 0
            
            # 20 pts: Estacionariedad (ADF Test)
            if adf_result['stationary']:
                if adf_result['pvalue'] < 0.01:
                    score += 20
                elif adf_result['pvalue'] < 0.05:
                    score += 15
            
            # 15 pts: Cointegración
            if coint_result['cointegrated']:
                if coint_result['pvalue'] < 0.01:
                    score += 15
                elif coint_result['pvalue'] < 0.05:
                    score += 10
            
            # Penalización por volatilidad extrema del spread
            if spread_vol > spread.std() * 2:
                score *= 0.8
            
            # Penalización por half-life muy largo (>100 días)
            if not np.isnan(half_life) and half_life > 100:
                score *= 0.9
            
            positive_corr_pct = (corr > 0).sum() / len(corr) * 100
            suggest_invert = positive_corr_pct < 50
            
            returns1 = np.log(p1 / p1.shift(1)).dropna()
            returns2 = np.log(p2 / p2.shift(1)).dropna()
            cond_corr = calculate_conditional_correlation(returns1, returns2)
            
            candidates.append({
                'asset1': asset1,
                'asset2': asset2,
                'score': score,
                'mean_correlation': mean_corr,
                'corr_stability_cv': stability['mean_cv'],
                'hurst': hurst,
                'half_life': half_life,
                'adf_pvalue': adf_result['pvalue'],
                'stationary': adf_result['stationary'],
                'cointegrated': coint_result['cointegrated'],
                'coint_pvalue': coint_result['pvalue'],
                'spread_volatility': spread_vol,
                'suggest_invert_trades': suggest_invert,
                'positive_corr_pct': positive_corr_pct,
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
# FUNCIONES DE BACKTESTING
# ============================================================================

def backtest_pairs_strategy(prices1, prices2, zscore_threshold=3.3, lookback=100, 
                            correlation_threshold=0.5, position_size=10000, 
                            stop_loss_pct=None, take_profit_pct=None):
    """
    Backtest completo de estrategia de pairs trading
    """
    # Calcular spread y z-score
    spread = calculate_log_ratio_spread(prices1, prices2)
    zscore = calculate_zscore(spread, window=lookback)
    corr = calculate_correlation(prices1, prices2, window=lookback)
    
    # Alinear índices
    common_idx = zscore.index.intersection(corr.index)
    zscore = zscore.loc[common_idx]
    corr = corr.loc[common_idx]
    spread = spread.loc[common_idx]
    prices1_aligned = prices1.loc[common_idx]
    prices2_aligned = prices2.loc[common_idx]
    
    # Inicializar variables
    position = 0  # 0: sin posición, 1: LONG spread, -1: SHORT spread
    entry_price1 = 0
    entry_price2 = 0
    entry_spread = 0
    trades = []
    equity_curve = [position_size]
    current_capital = position_size
    
    valid_corr = corr.abs() >= correlation_threshold
    
    for i in range(len(common_idx)):
        date = common_idx[i]
        z = zscore.iloc[i]
        s = spread.iloc[i]
        p1 = prices1_aligned.iloc[i]
        p2 = prices2_aligned.iloc[i]
        
        # Verificar stop loss / take profit si hay posición abierta
        if position != 0 and entry_spread != 0:
            spread_change_pct = (s - entry_spread) / abs(entry_spread) * 100
            
            # Stop Loss
            if stop_loss_pct is not None:
                if position == 1 and spread_change_pct < -stop_loss_pct:
                    # Cerrar LONG por stop loss
                    pnl = (s - entry_spread) * (position_size / 2)
                    current_capital += pnl
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'type': 'LONG',
                        'entry_spread': entry_spread,
                        'exit_spread': s,
                        'pnl': pnl,
                        'exit_reason': 'Stop Loss'
                    })
                    
                    position = 0
                    
                elif position == -1 and spread_change_pct > stop_loss_pct:
                    # Cerrar SHORT por stop loss
                    pnl = -(s - entry_spread) * (position_size / 2)
                    current_capital += pnl
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'type': 'SHORT',
                        'entry_spread': entry_spread,
                        'exit_spread': s,
                        'pnl': pnl,
                        'exit_reason': 'Stop Loss'
                    })
                    
                    position = 0
            
            # Take Profit
            if take_profit_pct is not None and position != 0:
                if position == 1 and spread_change_pct > take_profit_pct:
                    # Cerrar LONG por take profit
                    pnl = (s - entry_spread) * (position_size / 2)
                    current_capital += pnl
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'type': 'LONG',
                        'entry_spread': entry_spread,
                        'exit_spread': s,
                        'pnl': pnl,
                        'exit_reason': 'Take Profit'
                    })
                    
                    position = 0
                    
                elif position == -1 and spread_change_pct < -take_profit_pct:
                    # Cerrar SHORT por take profit
                    pnl = -(s - entry_spread) * (position_size / 2)
                    current_capital += pnl
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'type': 'SHORT',
                        'entry_spread': entry_spread,
                        'exit_spread': s,
                        'pnl': pnl,
                        'exit_reason': 'Take Profit'
                    })
                    
                    position = 0
        
        # Señales de entrada/salida
        if not valid_corr.iloc[i]:
            # Si correlación no es válida, cerrar posición
            if position != 0:
                if position == 1:
                    pnl = (s - entry_spread) * (position_size / 2)
                else:
                    pnl = -(s - entry_spread) * (position_size / 2)
                
                current_capital += pnl
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'type': 'LONG' if position == 1 else 'SHORT',
                    'entry_spread': entry_spread,
                    'exit_spread': s,
                    'pnl': pnl,
                    'exit_reason': 'Correlation Break'
                })
                
                position = 0
        else:
            # Señal SHORT (Z-score muy alto, esperamos que baje)
            if z > zscore_threshold and position == 0:
                position = -1
                entry_spread = s
                entry_date = date
                entry_price1 = p1
                entry_price2 = p2
            
            # Señal LONG (Z-score muy bajo, esperamos que suba)
            elif z < -zscore_threshold and position == 0:
                position = 1
                entry_spread = s
                entry_date = date
                entry_price1 = p1
                entry_price2 = p2
            
            # Cerrar posición cuando Z-score vuelve a 0
            elif abs(z) < 0.5 and position != 0:
                if position == 1:
                    pnl = (s - entry_spread) * (position_size / 2)
                else:
                    pnl = -(s - entry_spread) * (position_size / 2)
                
                current_capital += pnl
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'type': 'LONG' if position == 1 else 'SHORT',
                    'entry_spread': entry_spread,
                    'exit_spread': s,
                    'pnl': pnl,
                    'exit_reason': 'Mean Reversion'
                })
                
                position = 0
        
        equity_curve.append(current_capital)
    
    # Cerrar posición final si existe
    if position != 0:
        s = spread.iloc[-1]
        if position == 1:
            pnl = (s - entry_spread) * (position_size / 2)
        else:
            pnl = -(s - entry_spread) * (position_size / 2)
        
        current_capital += pnl
        
        trades.append({
            'entry_date': entry_date,
            'exit_date': common_idx[-1],
            'type': 'LONG' if position == 1 else 'SHORT',
            'entry_spread': entry_spread,
            'exit_spread': s,
            'pnl': pnl,
            'exit_reason': 'End of Data'
        })
        
        equity_curve.append(current_capital)
    
    # Calcular métricas
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) > 0:
        total_pnl = trades_df['pnl'].sum()
        total_return_pct = (total_pnl / position_size) * 100
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = (len(winning_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else np.inf
        
        # Drawdown
        equity_series = pd.Series(equity_curve)
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio
        returns = equity_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        
        # Calmar Ratio
        years = (common_idx[-1] - common_idx[0]).days / 365.25
        annual_return = (total_return_pct / years) if years > 0 else 0
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
    else:
        total_pnl = 0
        total_return_pct = 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        max_drawdown = 0
        sharpe = 0
        calmar = 0
    
    metrics = {
        'total_trades': len(trades_df),
        'total_pnl': total_pnl,
        'total_return_pct': total_return_pct,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar
    }
    
    return trades_df, equity_curve, metrics, zscore

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def plot_rolling_correlation(corr_df, asset1_name, asset2_name):
    """Crea un gráfico interactivo de la correlación móvil"""
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

def plot_multiple_rolling_correlations(df, pairs_list, window=10):
    """Crea gráficos de rolling correlation para múltiples pares"""
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

def plot_backtest_results(equity_curve, trades_df, zscore, initial_capital):
    """Visualiza resultados del backtest"""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Equity Curve', 'Z-Score con Trades'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Equity Curve
    dates = zscore.index
    equity_dates = dates.tolist() + [dates[-1]]  # Añadir fecha final
    
    fig.add_trace(go.Scatter(
        x=equity_dates,
        y=equity_curve,
        name='Equity',
        line=dict(color='#10b981', width=2),
        fill='tonexty',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ), row=1, col=1)
    
    fig.add_hline(y=initial_capital, line_dash="dash", line_color="#666666", 
                  annotation_text=f"Capital Inicial: ${initial_capital:,.0f}", row=1, col=1)
    
    # Z-Score
    fig.add_trace(go.Scatter(
        x=zscore.index,
        y=zscore,
        name='Z-Score',
        line=dict(color='#3b82f6', width=1.5)
    ), row=2, col=1)
    
    # Marcar trades en el Z-Score
    if len(trades_df) > 0:
        long_entries = trades_df[trades_df['type'] == 'LONG']
        short_entries = trades_df[trades_df['type'] == 'SHORT']
        
        if len(long_entries) > 0:
            # Encontrar z-scores en fechas de entrada
            long_zscores = [zscore.loc[date] if date in zscore.index else np.nan 
                           for date in long_entries['entry_date']]
            
            fig.add_trace(go.Scatter(
                x=long_entries['entry_date'],
                y=long_zscores,
                mode='markers',
                name='LONG Entry',
                marker=dict(color='#10b981', size=10, symbol='triangle-up')
            ), row=2, col=1)
        
        if len(short_entries) > 0:
            short_zscores = [zscore.loc[date] if date in zscore.index else np.nan 
                            for date in short_entries['entry_date']]
            
            fig.add_trace(go.Scatter(
                x=short_entries['entry_date'],
                y=short_zscores,
                mode='markers',
                name='SHORT Entry',
                marker=dict(color='#ef4444', size=10, symbol='triangle-down')
            ), row=2, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", row=2, col=1)
    
    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Capital ($)", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    
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
    
    if st.sidebar.button("📥 Descargar Todos los Activos", type="primary", key='btn_download_assets'):
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
    """)
    st.stop()

# ============================================================================
# PARÁMETROS DEL EA
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parámetros del EA")

lookback = st.sidebar.slider("InpLookback", 50, 200, 100, 10, key='param_lookback')
zscore_threshold = st.sidebar.slider("InpZScoreThreshold", 1.5, 5.0, 3.3, 0.1, key='param_zscore')
correlation_threshold = st.sidebar.slider("InpCorrelationThreshold", 0.3, 0.9, 0.5, 0.05, key='param_corr')

st.sidebar.subheader("Filtros de Búsqueda")
max_cv = st.sidebar.slider("Máx. CV (estabilidad)", 0.2, 0.8, 0.4, 0.05, key='param_cv')

st.sidebar.subheader("Rolling Correlation")
rolling_window = st.sidebar.slider("Window (días)", 10, 200, 30, 5, key='param_rolling_window')

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Backtesting")
position_size = st.sidebar.number_input("Tamaño Posición ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
stop_loss_pct = st.sidebar.number_input("Stop Loss (%)", min_value=0.0, max_value=50.0, value=0.0, step=0.5, help="0 = sin stop loss")
take_profit_pct = st.sidebar.number_input("Take Profit (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, help="0 = sin take profit")

# Crear DataFrame con TODOS los activos
df_all_prices = merge_asset_data(st.session_state.all_asset_data)

if df_all_prices.empty:
    st.error("No hay datos suficientes")
    st.stop()

st.success(f"✅ {len(df_all_prices)} días | {df_all_prices.index[0].date()} → {df_all_prices.index[-1].date()}")
st.info(f"📊 Usando {len(df_all_prices.columns)} activos para análisis")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Búsqueda de Pares",
    "📊 Análisis Individual",
    "📈 Análisis Detallado Top Pares",
    "🎯 Backtesting"
])

# ============================================================================
# TAB 1: BÚSQUEDA DE PARES
# ============================================================================

with tab1:
    st.header("🔍 Búsqueda de Mejores Pares")
    st.info("""
    **Criterios de Selección (Enfoque Estadístico):**
    - ✅ **Estabilidad de Correlación** (35 pts): Baja variación en el tiempo
    - ✅ **Mean Reversion** (30 pts): Hurst < 0.5 (tendencia a revertir)
    - ✅ **Estacionariedad** (20 pts): ADF test significativo
    - ✅ **Cointegración** (15 pts): Relación de largo plazo
    
    ❌ **NO se usa**: Cantidad de señales ni win rate (esos son para backtesting)
    """)
    
    if st.button("🚀 Buscar Mejores Pares", type="primary", key='btn_search_pairs'):
        
        st.markdown("### 📈 Buscando Pares con Correlación POSITIVA...")
        with st.spinner("Analizando correlaciones positivas..."):
            positive_pairs = find_best_pairs_for_ea(
                df_all_prices,
                correlation_type='positive',
                min_correlation=correlation_threshold,
                max_cv=max_cv,
                lookback=lookback
            )
        
        st.markdown("### 📉 Buscando Pares con Correlación NEGATIVA...")
        with st.spinner("Analizando correlaciones negativas..."):
            negative_pairs = find_best_pairs_for_ea(
                df_all_prices,
                correlation_type='negative',
                min_correlation=correlation_threshold,
                max_cv=max_cv,
                lookback=lookback
            )
        
        st.session_state.positive_pairs = positive_pairs
        st.session_state.negative_pairs = negative_pairs
        st.success("✅ Búsqueda completada!")
    
    if 'positive_pairs' in st.session_state and 'negative_pairs' in st.session_state:
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Top 20 Pares - Correlación POSITIVA")
            
            if len(st.session_state.positive_pairs) > 0:
                display_pos = st.session_state.positive_pairs.head(20).copy()
                display_pos['Activo 1'] = display_pos['asset1'].apply(lambda x: ASSETS[x]['label'])
                display_pos['Activo 2'] = display_pos['asset2'].apply(lambda x: ASSETS[x]['label'])
                display_pos['InpInvertTrades'] = 'false'
                display_pos['Estacionario'] = display_pos['stationary'].apply(lambda x: '✅' if x else '❌')
                display_pos['Cointegrado'] = display_pos['cointegrated'].apply(lambda x: '✅' if x else '❌')
                
                table_pos = display_pos[['Activo 1', 'Activo 2', 'score', 'mean_correlation', 
                                         'corr_stability_cv', 'hurst', 'half_life',
                                         'Estacionario', 'Cointegrado', 'InpInvertTrades']].rename(columns={
                    'score': 'Score',
                    'mean_correlation': 'Corr',
                    'corr_stability_cv': 'CV',
                    'hurst': 'Hurst',
                    'half_life': 'Half-Life'
                })
                
                st.dataframe(
                    table_pos.style.format({
                        'Score': '{:.1f}',
                        'Corr': '{:.3f}',
                        'CV': '{:.3f}',
                        'Hurst': '{:.3f}',
                        'Half-Life': '{:.1f}'
                    }),
                    width='stretch',
                    height=600
                )
                
                st.metric("Total pares positivos", len(st.session_state.positive_pairs))
                
                pair_options_pos = [f"{row['Activo 1']} / {row['Activo 2']}" 
                                   for _, row in display_pos.iterrows()]
                
                selected_pos_pair = st.selectbox(
                    "Seleccionar par para análisis",
                    options=pair_options_pos,
                    key='select_pos_pair'
                )
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button("📊 Analizar", key='btn_analyze_pos'):
                        idx = pair_options_pos.index(selected_pos_pair)
                        selected_row = display_pos.iloc[idx]
                        st.session_state.selected_asset1 = selected_row['asset1']
                        st.session_state.selected_asset2 = selected_row['asset2']
                        st.session_state.run_analysis = True
                        st.success(f"✅ {selected_pos_pair}")
                        st.info("👉 Ve a 'Análisis Individual'")
                
                with col_b:
                    if st.button("🎯 Backtest", key='btn_backtest_pos'):
                        idx = pair_options_pos.index(selected_pos_pair)
                        selected_row = display_pos.iloc[idx]
                        st.session_state.backtest_asset1 = selected_row['asset1']
                        st.session_state.backtest_asset2 = selected_row['asset2']
                        st.session_state.run_backtest = True
                        st.success(f"✅ {selected_pos_pair}")
                        st.info("👉 Ve a 'Backtesting'")
                
            else:
                st.warning("No se encontraron pares con correlación positiva")
        
        with col2:
            st.markdown("### 📉 Top 20 Pares - Correlación NEGATIVA")
            
            if len(st.session_state.negative_pairs) > 0:
                display_neg = st.session_state.negative_pairs.head(20).copy()
                display_neg['Activo 1'] = display_neg['asset1'].apply(lambda x: ASSETS[x]['label'])
                display_neg['Activo 2'] = display_neg['asset2'].apply(lambda x: ASSETS[x]['label'])
                display_neg['InpInvertTrades'] = 'true'
                display_neg['Estacionario'] = display_neg['stationary'].apply(lambda x: '✅' if x else '❌')
                display_neg['Cointegrado'] = display_neg['cointegrated'].apply(lambda x: '✅' if x else '❌')
                
                table_neg = display_neg[['Activo 1', 'Activo 2', 'score', 'mean_correlation', 
                                         'corr_stability_cv', 'hurst', 'half_life',
                                         'Estacionario', 'Cointegrado', 'InpInvertTrades']].rename(columns={
                    'score': 'Score',
                    'mean_correlation': 'Corr',
                    'corr_stability_cv': 'CV',
                    'hurst': 'Hurst',
                    'half_life': 'Half-Life'
                })
                
                st.dataframe(
                    table_neg.style.format({
                        'Score': '{:.1f}',
                        'Corr': '{:.3f}',
                        'CV': '{:.3f}',
                        'Hurst': '{:.3f}',
                        'Half-Life': '{:.1f}'
                    }),
                    width='stretch',
                    height=600
                )
                
                st.metric("Total pares negativos", len(st.session_state.negative_pairs))
                
                pair_options_neg = [f"{row['Activo 1']} / {row['Activo 2']}" 
                                   for _, row in display_neg.iterrows()]
                
                selected_neg_pair = st.selectbox(
                    "Seleccionar par para análisis",
                    options=pair_options_neg,
                    key='select_neg_pair'
                )
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button("📊 Analizar", key='btn_analyze_neg'):
                        idx = pair_options_neg.index(selected_neg_pair)
                        selected_row = display_neg.iloc[idx]
                        st.session_state.selected_asset1 = selected_row['asset1']
                        st.session_state.selected_asset2 = selected_row['asset2']
                        st.session_state.run_analysis = True
                        st.success(f"✅ {selected_neg_pair}")
                        st.info("👉 Ve a 'Análisis Individual'")
                
                with col_b:
                    if st.button("🎯 Backtest", key='btn_backtest_neg'):
                        idx = pair_options_neg.index(selected_neg_pair)
                        selected_row = display_neg.iloc[idx]
                        st.session_state.backtest_asset1 = selected_row['asset1']
                        st.session_state.backtest_asset2 = selected_row['asset2']
                        st.session_state.run_backtest = True
                        st.success(f"✅ {selected_neg_pair}")
                        st.info("👉 Ve a 'Backtesting'")
                
            else:
                st.warning("No se encontraron pares con correlación negativa")

# ============================================================================
# TAB 2: ANÁLISIS INDIVIDUAL
# ============================================================================

with tab2:
    st.header("📊 Análisis Individual de Par")
    
    # Selección de activos (con valores por defecto si hay par seleccionado)
    available_assets = list(st.session_state.all_asset_data.keys())
    
    default_asset1 = st.session_state.get('selected_asset1', available_assets[0])
    default_asset2 = st.session_state.get('selected_asset2', available_assets[1] if len(available_assets) > 1 else available_assets[0])
    
    # Asegurar que asset2 no sea igual a asset1
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
    
    # Configuración del análisis
    st.markdown("### ⚙️ Configuración del Análisis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        correlation_type_analysis = st.radio(
            "Tipo de Correlación a Buscar",
            options=['Positiva', 'Negativa'],
            help="**Positiva**: Pares que se mueven juntos\n**Negativa**: Pares que se mueven en direcciones opuestas"
        )
    
    with col2:
        invert_trades = st.checkbox(
            "InpInvertTrades",
            value=False,
            help="Invierte las señales de trading del EA"
        )
    
    with col3:
        if st.button("🔄 Actualizar Análisis", type="primary", key='btn_analyze'):
            st.session_state.run_analysis = True
    
    # Solo ejecutar análisis si se presionó el botón O si viene de selección de tabla
    if st.session_state.get('run_analysis', False):
        
        prices1 = df_all_prices[asset1]
        prices2 = df_all_prices[asset2]
        
        # Rolling Correlation
        st.markdown("### 📈 Rolling Correlation")
        corr_df = calculate_rolling_correlation(df_all_prices, asset1, asset2, window=rolling_window, step=1)
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
        else:
            if mean_corr <= -correlation_threshold:
                st.success(f"✅ Este par tiene correlación NEGATIVA fuerte ({mean_corr:.3f} <= {-correlation_threshold})")
            else:
                st.warning(f"⚠️ Este par NO tiene correlación negativa suficiente ({mean_corr:.3f} > {-correlation_threshold})")
        
        # Configuración sugerida para el EA
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
InpInvertTrades = {str(suggest_invert_based_on_corr).lower()}
InpLookback = {lookback}
InpZScoreThresholdLong = {zscore_threshold}
InpZScoreThresholdShort = {zscore_threshold}
InpCorrelationThreshold = {correlation_threshold}
            """, language="c++")
        
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
        
        # Resetear flag
        st.session_state.run_analysis = False
    
    else:
        st.info("👆 Configura el análisis y presiona **'🔄 Actualizar Análisis'** para ver los resultados, o selecciona un par desde la pestaña de búsqueda")

# ============================================================================
# TAB 3: ANÁLISIS DETALLADO TOP PARES
# ============================================================================

with tab3:
    st.header("📈 Análisis Detallado Top Pares")
    
    if 'positive_pairs' in st.session_state and 'negative_pairs' in st.session_state:
        
        # Rolling Correlation Top 10 Positivos
        st.markdown("### 📈 Rolling Correlation - Top 10 Pares Positivos")
        
        if len(st.session_state.positive_pairs) > 0:
            top_pos = st.session_state.positive_pairs.head(10).to_dict('records')
            
            with st.spinner("Generando gráficos de correlación..."):
                fig_pos = plot_multiple_rolling_correlations(df_all_prices, top_pos, window=rolling_window)
                st.plotly_chart(fig_pos, use_container_width=True)
        else:
            st.info("No hay pares con correlación positiva")
        
        st.markdown("---")
        
        # Rolling Correlation Top 10 Negativos
        st.markdown("### 📉 Rolling Correlation - Top 10 Pares Negativos")
        
        if len(st.session_state.negative_pairs) > 0:
            top_neg = st.session_state.negative_pairs.head(10).to_dict('records')
            
            with st.spinner("Generando gráficos de correlación..."):
                fig_neg = plot_multiple_rolling_correlations(df_all_prices, top_neg, window=rolling_window)
                st.plotly_chart(fig_neg, use_container_width=True)
        else:
            st.info("No hay pares con correlación negativa")
        
        st.markdown("---")
        
        # Comparación de Métricas
        st.markdown("### 📊 Comparación de Métricas - Todos los Pares Encontrados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(st.session_state.positive_pairs) > 0:
                st.markdown("#### 📈 Pares Positivos")
                
                # Gráfico de Score vs Hurst
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.positive_pairs['score'],
                    y=st.session_state.positive_pairs['hurst'],
                    mode='markers',
                    marker=dict(
                        size=st.session_state.positive_pairs['corr_stability_cv'] * 100,
                        color=st.session_state.positive_pairs['mean_correlation'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Corr")
                    ),
                    text=[f"{ASSETS[row['asset1']]['label']} / {ASSETS[row['asset2']]['label']}" 
                          for _, row in st.session_state.positive_pairs.iterrows()],
                    hovertemplate='<b>%{text}</b><br>Score: %{x:.1f}<br>Hurst: %{y:.3f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Score vs Hurst (tamaño = CV)',
                    xaxis_title='Score',
                    yaxis_title='Hurst Exponent',
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if len(st.session_state.negative_pairs) > 0:
                st.markdown("#### 📉 Pares Negativos")
                
                # Gráfico de Score vs Hurst
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.negative_pairs['score'],
                    y=st.session_state.negative_pairs['hurst'],
                    mode='markers',
                    marker=dict(
                        size=st.session_state.negative_pairs['corr_stability_cv'] * 100,
                        color=st.session_state.negative_pairs['mean_correlation'],
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title="Corr")
                    ),
                    text=[f"{ASSETS[row['asset1']]['label']} / {ASSETS[row['asset2']]['label']}" 
                          for _, row in st.session_state.negative_pairs.iterrows()],
                    hovertemplate='<b>%{text}</b><br>Score: %{x:.1f}<br>Hurst: %{y:.3f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Score vs Hurst (tamaño = CV)',
                    xaxis_title='Score',
                    yaxis_title='Hurst Exponent',
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Primero ejecuta la búsqueda de pares en la pestaña 'Búsqueda de Pares'")

# ============================================================================
# TAB 4: BACKTESTING
# ============================================================================

with tab4:
    st.header("🎯 Backtesting de Estrategia Pairs Trading")
    
    # Selección de activos
    available_assets = list(st.session_state.all_asset_data.keys())
    
    default_backtest_asset1 = st.session_state.get('backtest_asset1', available_assets[0])
    default_backtest_asset2 = st.session_state.get('backtest_asset2', available_assets[1] if len(available_assets) > 1 else available_assets[0])
    
    if default_backtest_asset2 == default_backtest_asset1 and len(available_assets) > 1:
        default_backtest_asset2 = available_assets[1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        backtest_asset1 = st.selectbox(
            "Activo 1",
            options=available_assets,
            index=available_assets.index(default_backtest_asset1) if default_backtest_asset1 in available_assets else 0,
            format_func=lambda x: ASSETS[x]['label'],
            key='backtest_asset1_select'
        )
    
    with col2:
        backtest_asset2_options = [a for a in available_assets if a != backtest_asset1]
        backtest_asset2 = st.selectbox(
            "Activo 2",
            options=backtest_asset2_options,
            index=backtest_asset2_options.index(default_backtest_asset2) if default_backtest_asset2 in backtest_asset2_options else 0,
            format_func=lambda x: ASSETS[x]['label'],
            key='backtest_asset2_select'
        )
    
    st.markdown("### ⚙️ Configuración del Backtest")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Z-Score Threshold", f"±{zscore_threshold}")
        st.metric("Lookback", f"{lookback} días")
    
    with col2:
        st.metric("Correlación Min", f"{correlation_threshold:.2f}")
        st.metric("Tamaño Posición", f"${position_size:,.0f}")
    
    with col3:
        if stop_loss_pct > 0:
            st.metric("Stop Loss", f"{stop_loss_pct}%")
        else:
            st.metric("Stop Loss", "Desactivado")
        
        if take_profit_pct > 0:
            st.metric("Take Profit", f"{take_profit_pct}%")
        else:
            st.metric("Take Profit", "Desactivado")
    
    if st.button("🚀 Ejecutar Backtest", type="primary", key='btn_run_backtest'):
        st.session_state.run_backtest = True
    
    if st.session_state.get('run_backtest', False):
        
        with st.spinner("Ejecutando backtest..."):
            prices1 = df_all_prices[backtest_asset1]
            prices2 = df_all_prices[backtest_asset2]
            
            # Ejecutar backtest
            trades_df, equity_curve, metrics, zscore = backtest_pairs_strategy(
                prices1, prices2,
                zscore_threshold=zscore_threshold,
                lookback=lookback,
                correlation_threshold=correlation_threshold,
                position_size=position_size,
                stop_loss_pct=stop_loss_pct if stop_loss_pct > 0 else None,
                take_profit_pct=take_profit_pct if take_profit_pct > 0 else None
            )
        
        st.success("✅ Backtest completado!")
        
        # Mostrar métricas principales
        st.markdown("### 📊 Resultados del Backtest")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Trades", metrics['total_trades'])
        col2.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        col3.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        col4.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total PnL", f"${metrics['total_pnl']:,.2f}")
        col2.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
        col3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        col4.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.2f}")
        
        col1, col2 = st.columns(2)
        
        col1.metric("Avg Win", f"${metrics['avg_win']:,.2f}")
        col2.metric("Avg Loss", f"${metrics['avg_loss']:,.2f}")
        
        # Gráficos
        st.markdown("### 📈 Gráficos")
        
        fig = plot_backtest_results(equity_curve, trades_df, zscore, position_size)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de trades
        st.markdown("### 📋 Historial de Trades")
        
        if len(trades_df) > 0:
            display_trades = trades_df.copy()
            display_trades['entry_date'] = pd.to_datetime(display_trades['entry_date']).dt.strftime('%Y-%m-%d')
            display_trades['exit_date'] = pd.to_datetime(display_trades['exit_date']).dt.strftime('%Y-%m-%d')
            display_trades['duration'] = (pd.to_datetime(trades_df['exit_date']) - pd.to_datetime(trades_df['entry_date'])).dt.days
            
            st.dataframe(
                display_trades.style.format({
                    'entry_spread': '{:.4f}',
                    'exit_spread': '{:.4f}',
                    'pnl': '${:,.2f}'
                }),
                width='stretch',
                height=400
            )
            
            # Descargar resultados
            csv = trades_df.to_csv(index=False)
            st.download_button(
                "📥 Descargar Trades (CSV)",
                csv,
                f"backtest_{ASSETS[backtest_asset1]['label']}_{ASSETS[backtest_asset2]['label']}.csv",
                "text/csv"
            )
        else:
            st.warning("No se generaron trades en el período analizado")
        
        # Análisis de distribución de PnL
        if len(trades_df) > 0:
            st.markdown("### 📊 Distribución de PnL")
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=trades_df['pnl'],
                nbinsx=30,
                marker_color='#3b82f6',
                name='PnL Distribution'
            ))
            
            fig.update_layout(
                title='Distribución de PnL por Trade',
                xaxis_title='PnL ($)',
                yaxis_title='Frecuencia',
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.session_state.run_backtest = False
    
    else:
        st.info("👆 Configura los parámetros y presiona **'🚀 Ejecutar Backtest'**, o selecciona un par desde la pestaña de búsqueda")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Guía Rápida")
st.sidebar.markdown("""
**Flujo de Trabajo:**
1. 🔍 **Búsqueda**: Encuentra pares (criterios estadísticos)
2. 📊 **Individual**: Analiza correlación
3. 📈 **Top Pares**: Compara gráficos
4. 🎯 **Backtesting**: Simula estrategia

**Nuevo Scoring:** Basado en propiedades estadísticas, NO en señales
""")

st.sidebar.success("✨ Diseñado para EA MQL5")
