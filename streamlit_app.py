import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests
import warnings
import pickle
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Pairs Trading - Correlation Analysis",
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTES - VENTANAS FIJAS
# ============================================================================

ROLLING_WINDOW = 30  # Ventana fija para rolling correlation
LOOKBACK_ANALYSIS = 30  # Ventana fija para análisis

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
# FUNCIONES DE ANÁLISIS
# ============================================================================

def calculate_log_ratio_spread(prices1, prices2):
    """Calcula spread usando log-ratio"""
    spread = np.log(prices1) - np.log(prices2)
    return spread.dropna()

def calculate_rolling_correlation(df, asset1, asset2, window=ROLLING_WINDOW, step=1):
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
# FUNCIONES DE LEAD-LAG
# ============================================================================

def calculate_cross_correlation(returns1, returns2, max_lag=20):
    """
    Calcula cross-correlation entre dos series de retornos.
    Lag positivo = returns1 lidera a returns2
    Lag negativo = returns2 lidera a returns1
    """
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        if lag < 0:
            # returns2 lidera (shift returns1 hacia adelante)
            corr = returns1.shift(-lag).corr(returns2)
        elif lag > 0:
            # returns1 lidera (shift returns2 hacia adelante)
            corr = returns1.corr(returns2.shift(lag))
        else:
            corr = returns1.corr(returns2)
        correlations.append(corr)
    
    return pd.DataFrame({
        'lag': list(lags),
        'correlation': correlations
    })

def calculate_rolling_lead_lag(returns1, returns2, window=60, max_lag=10):
    """
    Calcula el lag óptimo de forma rolling para detectar cambios en la relación lead-lag.
    """
    optimal_lags = []
    max_correlations = []
    dates = []
    
    for i in range(window, len(returns1)):
        r1_window = returns1.iloc[i-window:i]
        r2_window = returns2.iloc[i-window:i]
        
        best_lag = 0
        best_corr = r1_window.corr(r2_window)
        
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                continue
            if lag < 0:
                corr = r1_window.shift(-lag).corr(r2_window)
            else:
                corr = r1_window.corr(r2_window.shift(lag))
            
            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        
        optimal_lags.append(best_lag)
        max_correlations.append(best_corr)
        dates.append(returns1.index[i])
    
    return pd.DataFrame({
        'date': dates,
        'optimal_lag': optimal_lags,
        'max_correlation': max_correlations
    })

def granger_causality_test(returns1, returns2, max_lag=5):
    """
    Test de causalidad de Granger bidireccional.
    Retorna p-values para ambas direcciones.
    """
    results = {
        'asset1_causes_asset2': {},
        'asset2_causes_asset1': {}
    }
    
    # Preparar datos
    df_test = pd.DataFrame({
        'r1': returns1,
        'r2': returns2
    }).dropna()
    
    if len(df_test) < max_lag * 10:
        return None
    
    try:
        # Test: ¿r1 causa r2?
        test1 = grangercausalitytests(df_test[['r2', 'r1']], maxlag=max_lag, verbose=False)
        for lag in range(1, max_lag + 1):
            results['asset1_causes_asset2'][lag] = test1[lag][0]['ssr_ftest'][1]
        
        # Test: ¿r2 causa r1?
        test2 = grangercausalitytests(df_test[['r1', 'r2']], maxlag=max_lag, verbose=False)
        for lag in range(1, max_lag + 1):
            results['asset2_causes_asset1'][lag] = test2[lag][0]['ssr_ftest'][1]
        
        return results
    except Exception as e:
        return None

def calculate_impulse_response(returns1, returns2, periods=20):
    """
    Calcula una aproximación simple de impulse response.
    Mide cómo un shock en un activo afecta al otro en períodos futuros.
    """
    # Normalizar retornos
    r1_norm = (returns1 - returns1.mean()) / returns1.std()
    r2_norm = (returns2 - returns2.mean()) / returns2.std()
    
    # Identificar shocks grandes (> 2 std)
    shock_threshold = 2.0
    
    # Shocks en asset1 -> respuesta en asset2
    shock_dates_1 = r1_norm[r1_norm.abs() > shock_threshold].index
    responses_1_to_2 = []
    
    for t in range(periods):
        responses = []
        for shock_date in shock_dates_1:
            try:
                shock_idx = returns2.index.get_loc(shock_date)
                if shock_idx + t < len(returns2):
                    responses.append(r2_norm.iloc[shock_idx + t])
            except:
                continue
        if responses:
            responses_1_to_2.append(np.mean(responses))
        else:
            responses_1_to_2.append(np.nan)
    
    # Shocks en asset2 -> respuesta en asset1
    shock_dates_2 = r2_norm[r2_norm.abs() > shock_threshold].index
    responses_2_to_1 = []
    
    for t in range(periods):
        responses = []
        for shock_date in shock_dates_2:
            try:
                shock_idx = returns1.index.get_loc(shock_date)
                if shock_idx + t < len(returns1):
                    responses.append(r1_norm.iloc[shock_idx + t])
            except:
                continue
        if responses:
            responses_2_to_1.append(np.mean(responses))
        else:
            responses_2_to_1.append(np.nan)
    
    return {
        'periods': list(range(periods)),
        'asset1_shock_to_asset2': responses_1_to_2,
        'asset2_shock_to_asset1': responses_2_to_1,
        'n_shocks_asset1': len(shock_dates_1),
        'n_shocks_asset2': len(shock_dates_2)
    }

def find_lead_lag_pairs(df, min_correlation=0.3, max_lag=10):
    """
    Busca pares con relaciones lead-lag significativas.
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
            status_text.text(f"Analizando {pair_idx}/{total_pairs}: {ASSETS[asset1]['label'][:15]} vs {ASSETS[asset2]['label'][:15]}")
            
            prices1 = df[asset1].dropna()
            prices2 = df[asset2].dropna()
            
            common_idx = prices1.index.intersection(prices2.index)
            if len(common_idx) < 252:
                continue
            
            p1 = prices1.loc[common_idx]
            p2 = prices2.loc[common_idx]
            
            returns1 = np.log(p1 / p1.shift(1)).dropna()
            returns2 = np.log(p2 / p2.shift(1)).dropna()
            
            common_ret_idx = returns1.index.intersection(returns2.index)
            returns1 = returns1.loc[common_ret_idx]
            returns2 = returns2.loc[common_ret_idx]
            
            # Cross-correlation
            cross_corr = calculate_cross_correlation(returns1, returns2, max_lag=max_lag)
            
            # Encontrar lag óptimo
            idx_max = cross_corr['correlation'].abs().idxmax()
            optimal_lag = cross_corr.loc[idx_max, 'lag']
            max_corr = cross_corr.loc[idx_max, 'correlation']
            
            # Correlación contemporánea
            contemp_corr = cross_corr[cross_corr['lag'] == 0]['correlation'].values[0]
            
            # Solo incluir si hay lead-lag significativo
            if abs(max_corr) < min_correlation:
                continue
            
            # Granger causality
            granger = granger_causality_test(returns1, returns2, max_lag=5)
            
            if granger:
                # Mejor p-value para cada dirección
                best_pval_1_to_2 = min(granger['asset1_causes_asset2'].values())
                best_pval_2_to_1 = min(granger['asset2_causes_asset1'].values())
            else:
                best_pval_1_to_2 = np.nan
                best_pval_2_to_1 = np.nan
            
            # Determinar líder
            if optimal_lag > 0:
                leader = asset1
                follower = asset2
                lead_days = optimal_lag
            elif optimal_lag < 0:
                leader = asset2
                follower = asset1
                lead_days = -optimal_lag
            else:
                leader = None
                follower = None
                lead_days = 0
            
            candidates.append({
                'asset1': asset1,
                'asset2': asset2,
                'leader': leader,
                'follower': follower,
                'optimal_lag': optimal_lag,
                'lead_days': lead_days,
                'max_correlation': max_corr,
                'contemp_correlation': contemp_corr,
                'correlation_improvement': abs(max_corr) - abs(contemp_corr),
                'granger_pval_1_to_2': best_pval_1_to_2,
                'granger_pval_2_to_1': best_pval_2_to_1,
                'granger_significant_1_to_2': best_pval_1_to_2 < 0.05 if not np.isnan(best_pval_1_to_2) else False,
                'granger_significant_2_to_1': best_pval_2_to_1 < 0.05 if not np.isnan(best_pval_2_to_1) else False,
                'years_data': (common_idx[-1] - common_idx[0]).days / 365.25
            })
    
    progress_bar.empty()
    status_text.empty()
    
    if not candidates:
        return pd.DataFrame()
    
    df_result = pd.DataFrame(candidates)
    
    # Score basado en:
    # - Mejora de correlación con lag
    # - Significancia de Granger
    # - Lead-lag claro (no contemporáneo)
    df_result['score'] = (
        df_result['correlation_improvement'] * 50 +
        (df_result['granger_significant_1_to_2'] | df_result['granger_significant_2_to_1']).astype(int) * 30 +
        (df_result['lead_days'] > 0).astype(int) * 20
    )
    
    return df_result.sort_values('score', ascending=False)

# ============================================================================
# FUNCIONES DE ANÁLISIS DE ESTACIONALIDAD
# ============================================================================

def analyze_seasonality(df, asset1, asset2, lookback=LOOKBACK_ANALYSIS):
    """Analiza patrones estacionales en la correlación y el spread"""
    prices1 = df[asset1]
    prices2 = df[asset2]
    
    # Calcular spread
    spread = calculate_log_ratio_spread(prices1, prices2)
    
    # Calcular correlación rolling
    returns1 = np.log(prices1 / prices1.shift(1))
    returns2 = np.log(prices2 / prices2.shift(1))
    corr_rolling = returns1.rolling(lookback).corr(returns2)
    
    # Crear DataFrame con fechas
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
    
    # Análisis mensual
    monthly_corr = corr_df.groupby('month')['correlation'].agg(['mean', 'std', 'min', 'max'])
    monthly_spread_vol = spread_df.groupby('month')['volatility'].mean()
    
    # Análisis trimestral
    quarterly_corr = corr_df.groupby('quarter')['correlation'].agg(['mean', 'std', 'min', 'max'])
    quarterly_spread_vol = spread_df.groupby('quarter')['volatility'].mean()
    
    # Análisis anual
    yearly_corr = corr_df.groupby('year')['correlation'].agg(['mean', 'std', 'min', 'max'])
    yearly_spread_vol = spread_df.groupby('year')['volatility'].mean()
    
    # Análisis de volatilidad estacional del spread
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
    
    # Definir períodos históricos relevantes
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
            
            if len(period_spread) > 30:  # Mínimo 30 días
                period_corr = period_p1.corr(period_p2)
                
                periods.append({
                    'period': label,
                    'start': start,
                    'end': end,
                    'days': len(period_spread),
                    'correlation': period_corr,
                    'spread_mean': period_spread.mean(),
                    'spread_std': period_spread.std(),
                    'spread_min': period_spread.min(),
                    'spread_max': period_spread.max()
                })
        except:
            continue
    
    return pd.DataFrame(periods)

def find_best_pairs(df, correlation_type='positive', min_correlation=0.5, 
                    max_cv=0.4, lookback=LOOKBACK_ANALYSIS):
    """Encuentra los mejores pares usando criterios estadísticos"""
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
            
            # Calcular correlación
            mean_corr = p1.corr(p2)
            
            # Filtrar según tipo de correlación
            if correlation_type == 'positive':
                if mean_corr < min_correlation:
                    continue
            else:  # negative
                if mean_corr > -min_correlation:
                    continue
            
            # Calcular spread
            spread = calculate_log_ratio_spread(p1, p2)
            
            # Tests estadísticos
            adf_result = adf_test(spread)
            hurst = calculate_hurst_exponent(spread.dropna())
            half_life = calculate_half_life(spread)
            coint_result = test_cointegration(p1, p2)
            
            # Estabilidad de correlación
            corr_rolling = calculate_rolling_correlation(df, asset1, asset2, window=lookback)
            corr_series = pd.Series(corr_rolling['correlation'].values, index=corr_rolling['date'])
            stability = calculate_correlation_stability(corr_series)
            
            if stability['mean_cv'] > max_cv:
                continue
            
            # Volatilidad del spread
            spread_vol = calculate_spread_volatility(spread)
            
            # SCORE
            score = 0
            
            # Estabilidad de correlación
            if stability['mean_cv'] < 0.15:
                score += 35
            elif stability['mean_cv'] < 0.25:
                score += 25
            elif stability['mean_cv'] < 0.35:
                score += 15
            else:
                score += 5
            
            # Mean Reversion
            if hurst < 0.35:
                score += 30
            elif hurst < 0.45:
                score += 20
            elif hurst < 0.5:
                score += 10
            
            # Estacionariedad
            if adf_result['stationary']:
                if adf_result['pvalue'] < 0.01:
                    score += 20
                elif adf_result['pvalue'] < 0.05:
                    score += 15
            
            # Cointegración
            if coint_result['cointegrated']:
                if coint_result['pvalue'] < 0.01:
                    score += 15
                elif coint_result['pvalue'] < 0.05:
                    score += 10
            
            if spread_vol > spread.std() * 2:
                score *= 0.8
            
            if not np.isnan(half_life) and half_life > 100:
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
                'hurst': hurst,
                'half_life': half_life,
                'adf_pvalue': adf_result['pvalue'],
                'stationary': adf_result['stationary'],
                'cointegrated': coint_result['cointegrated'],
                'coint_pvalue': coint_result['pvalue'],
                'spread_volatility': spread_vol,
                'corr_positive_markets': cond_corr['positive_markets'],
                'corr_negative_markets': cond_corr['negative_markets'],
                'corr_high_volatility': cond_corr['high_volatility'],
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
        title=f'Rolling Correlation ({ROLLING_WINDOW}d): {asset1_name} vs {asset2_name}',
        xaxis_title='Fecha',
        yaxis_title='Correlación',
        yaxis=dict(range=[-1, 1]),
        template='plotly_dark',
        hovermode='x unified',
        height=400,
        showlegend=True
    )
    
    return fig

def plot_multiple_rolling_correlations(df, pairs_list, window=ROLLING_WINDOW):
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

def plot_cross_correlation(cross_corr_df, asset1_name, asset2_name):
    """Gráfico de cross-correlation"""
    fig = go.Figure()
    
    colors = ['#ef4444' if lag < 0 else '#10b981' if lag > 0 else '#3b82f6' 
              for lag in cross_corr_df['lag']]
    
    fig.add_trace(go.Bar(
        x=cross_corr_df['lag'],
        y=cross_corr_df['correlation'],
        marker_color=colors,
        hovertemplate='Lag: %{x}<br>Correlación: %{y:.4f}<extra></extra>'
    ))
    
    # Marcar lag óptimo
    idx_max = cross_corr_df['correlation'].abs().idxmax()
    optimal_lag = cross_corr_df.loc[idx_max, 'lag']
    max_corr = cross_corr_df.loc[idx_max, 'correlation']
    
    fig.add_vline(x=optimal_lag, line_dash="dash", line_color="#f59e0b", 
                  annotation_text=f"Óptimo: {optimal_lag}d", annotation_position="top")
    
    fig.update_layout(
        title=f'Cross-Correlation: {asset1_name} vs {asset2_name}',
        xaxis_title=f'Lag (días) - Positivo: {asset1_name} lidera | Negativo: {asset2_name} lidera',
        yaxis_title='Correlación',
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    
    return fig

def plot_rolling_lead_lag(rolling_ll_df, asset1_name, asset2_name):
    """Gráfico de lead-lag rolling"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Lag Óptimo Rolling', 'Correlación Máxima'),
        vertical_spacing=0.15
    )
    
    # Lag óptimo
    colors = ['#ef4444' if lag < 0 else '#10b981' if lag > 0 else '#3b82f6' 
              for lag in rolling_ll_df['optimal_lag']]
    
    fig.add_trace(go.Scatter(
        x=rolling_ll_df['date'],
        y=rolling_ll_df['optimal_lag'],
        mode='lines',
        name='Lag Óptimo',
        line=dict(color='#3b82f6', width=1.5)
    ), row=1, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", row=1, col=1)
    
    # Correlación máxima
    fig.add_trace(go.Scatter(
        x=rolling_ll_df['date'],
        y=rolling_ll_df['max_correlation'],
        mode='lines',
        name='Correlación Máx',
        line=dict(color='#10b981', width=1.5)
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", row=2, col=1)
    
    fig.update_layout(
        title=f'Lead-Lag Rolling: {asset1_name} vs {asset2_name}',
        height=500,
        template='plotly_dark',
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Lag (días)", row=1, col=1)
    fig.update_yaxes(title_text="Correlación", row=2, col=1)
    
    return fig

def plot_granger_causality(granger_results, asset1_name, asset2_name):
    """Gráfico de Granger causality p-values"""
    if granger_results is None:
        return None
    
    lags = list(granger_results['asset1_causes_asset2'].keys())
    pvals_1_to_2 = list(granger_results['asset1_causes_asset2'].values())
    pvals_2_to_1 = list(granger_results['asset2_causes_asset1'].values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lags,
        y=pvals_1_to_2,
        mode='lines+markers',
        name=f'{asset1_name} → {asset2_name}',
        line=dict(color='#10b981', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=lags,
        y=pvals_2_to_1,
        mode='lines+markers',
        name=f'{asset2_name} → {asset1_name}',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_hline(y=0.05, line_dash="dash", line_color="#ef4444", 
                  annotation_text="p=0.05", annotation_position="right")
    fig.add_hline(y=0.01, line_dash="dot", line_color="#f59e0b", 
                  annotation_text="p=0.01", annotation_position="right")
    
    fig.update_layout(
        title='Test de Causalidad de Granger (p-values)',
        xaxis_title='Lag (días)',
        yaxis_title='p-value',
        yaxis=dict(type='log'),
        template='plotly_dark',
        height=400,
        legend=dict(x=0.7, y=0.95)
    )
    
    return fig

def plot_impulse_response(ir_results, asset1_name, asset2_name):
    """Gráfico de impulse response"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ir_results['periods'],
        y=ir_results['asset1_shock_to_asset2'],
        mode='lines+markers',
        name=f'Shock {asset1_name} → {asset2_name}',
        line=dict(color='#10b981', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=ir_results['periods'],
        y=ir_results['asset2_shock_to_asset1'],
        mode='lines+markers',
        name=f'Shock {asset2_name} → {asset1_name}',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#666666")
    
    fig.update_layout(
        title=f'Impulse Response (Shocks > 2σ)',
        xaxis_title='Períodos después del shock',
        yaxis_title='Respuesta normalizada',
        template='plotly_dark',
        height=400,
        legend=dict(x=0.6, y=0.95)
    )
    
    return fig

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

st.title("📊 Pairs Trading - Correlation Analysis")
st.markdown("**Análisis de correlaciones, estacionalidad y Lead-Lag para pares de activos**")
st.info(f"📊 **10 años de historia** | 🔍 **Ventana fija: {ROLLING_WINDOW} días** | 📈 **Análisis Lead-Lag**")

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
    ### 👋 Bienvenido al Análisis de Correlaciones
    
    **Activos disponibles ({len(ASSETS)}):**
    - 📊 {len([a for a in ASSETS.values() if a['category'] == 'Indices'])} Índices globales
    - 💱 {len([a for a in ASSETS.values() if a['category'] == 'Forex'])} Pares de divisas (incluye DXY)
    - 🏆 {len([a for a in ASSETS.values() if a['category'] == 'Commodities'])} Commodities
    - ₿ {len([a for a in ASSETS.values() if a['category'] == 'Crypto'])} Criptomonedas
    
    **Características:**
    - 📅 **10 años de datos históricos** (2015-2025)
    - 📊 **Ventana fija de {ROLLING_WINDOW} días**
    - 📈 **Análisis Lead-Lag** completo
    - 🎯 **Test de Granger Causality**
    
    **Para comenzar:**
    1. Presiona "📥 Descargar Datos (10 años)"
    """)
    st.stop()

# ============================================================================
# PARÁMETROS (SIMPLIFICADOS)
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parámetros de Análisis")

min_correlation = st.sidebar.slider("Correlación Mínima", 0.3, 0.9, 0.5, 0.05)
max_cv = st.sidebar.slider("Máx. CV (estabilidad)", 0.2, 0.8, 0.4, 0.05)

st.sidebar.info(f"📊 Ventana fija: **{ROLLING_WINDOW} días**")

# Crear DataFrame
df_all_prices = merge_asset_data(st.session_state.all_asset_data)

if df_all_prices.empty:
    st.error("No hay datos suficientes")
    st.stop()

years_available = (df_all_prices.index[-1] - df_all_prices.index[0]).days / 365.25

st.success(f"✅ {len(df_all_prices)} días ({years_available:.1f} años) | {df_all_prices.index[0].date()} → {df_all_prices.index[-1].date()}")
st.info(f"📊 {len(df_all_prices.columns)} activos disponibles | 💱 DXY incluido")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Búsqueda de Pares",
    "📊 Análisis Individual",
    "📈 Comparación de Pares",
    "📅 Estacionalidad",
    "⏱️ Lead-Lag"
])

# ============================================================================
# TAB 1: BÚSQUEDA
# ============================================================================

with tab1:
    st.header("🔍 Búsqueda de Mejores Pares")
    st.info("""
    **Criterios Estadísticos:**
    - ✅ Estabilidad de Correlación (35 pts)
    - ✅ Mean Reversion - Hurst < 0.5 (30 pts)
    - ✅ Estacionariedad - ADF test (20 pts)
    - ✅ Cointegración (15 pts)
    """)
    
    if st.button("🚀 Buscar Pares", type="primary"):
        
        st.markdown("### 📈 Correlación POSITIVA...")
        with st.spinner("Analizando..."):
            positive_pairs = find_best_pairs(
                df_all_prices,
                correlation_type='positive',
                min_correlation=min_correlation,
                max_cv=max_cv,
                lookback=LOOKBACK_ANALYSIS
            )
        
        st.markdown("### 📉 Correlación NEGATIVA...")
        with st.spinner("Analizando..."):
            negative_pairs = find_best_pairs(
                df_all_prices,
                correlation_type='negative',
                min_correlation=min_correlation,
                max_cv=max_cv,
                lookback=LOOKBACK_ANALYSIS
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
                display_pos['✓ Estacionario'] = display_pos['stationary'].apply(lambda x: '✅' if x else '❌')
                display_pos['✓ Cointegrado'] = display_pos['cointegrated'].apply(lambda x: '✅' if x else '❌')
                
                table_pos = display_pos[['Activo 1', 'Activo 2', 'score', 'mean_correlation', 
                                         'corr_stability_cv', 'hurst', 'half_life', 'years_data',
                                         '✓ Estacionario', '✓ Cointegrado']].rename(columns={
                    'score': 'Score',
                    'mean_correlation': 'Corr',
                    'corr_stability_cv': 'CV',
                    'hurst': 'Hurst',
                    'half_life': 'Half-Life',
                    'years_data': 'Años'
                })
                
                st.dataframe(
                    table_pos.style.format({
                        'Score': '{:.1f}',
                        'Corr': '{:.3f}',
                        'CV': '{:.3f}',
                        'Hurst': '{:.3f}',
                        'Half-Life': '{:.1f}',
                        'Años': '{:.1f}'
                    }),
                    width='stretch',
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
                    st.info("👉 Ve a 'Análisis Individual', 'Estacionalidad' o 'Lead-Lag'")
            else:
                st.warning("No se encontraron pares")
        
        with col2:
            st.markdown("### 📉 Top 20 - Correlación NEGATIVA")
            
            if len(st.session_state.negative_pairs) > 0:
                display_neg = st.session_state.negative_pairs.head(20).copy()
                display_neg['Activo 1'] = display_neg['asset1'].apply(lambda x: ASSETS[x]['label'])
                display_neg['Activo 2'] = display_neg['asset2'].apply(lambda x: ASSETS[x]['label'])
                display_neg['✓ Estacionario'] = display_neg['stationary'].apply(lambda x: '✅' if x else '❌')
                display_neg['✓ Cointegrado'] = display_neg['cointegrated'].apply(lambda x: '✅' if x else '❌')
                
                table_neg = display_neg[['Activo 1', 'Activo 2', 'score', 'mean_correlation', 
                                         'corr_stability_cv', 'hurst', 'half_life', 'years_data',
                                         '✓ Estacionario', '✓ Cointegrado']].rename(columns={
                    'score': 'Score',
                    'mean_correlation': 'Corr',
                    'corr_stability_cv': 'CV',
                    'hurst': 'Hurst',
                    'half_life': 'Half-Life',
                    'years_data': 'Años'
                })
                
                st.dataframe(
                    table_neg.style.format({
                        'Score': '{:.1f}',
                        'Corr': '{:.3f}',
                        'CV': '{:.3f}',
                        'Hurst': '{:.3f}',
                        'Half-Life': '{:.1f}',
                        'Años': '{:.1f}'
                    }),
                    width='stretch',
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
                    st.info("👉 Ve a 'Análisis Individual', 'Estacionalidad' o 'Lead-Lag'")
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
    
    if st.button("🔄 Analizar", type="primary"):
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
        
        # Rolling Correlation
        st.markdown(f"### 📈 Rolling Correlation ({ROLLING_WINDOW}d)")
        corr_df = calculate_rolling_correlation(df_all_prices, asset1, asset2, window=ROLLING_WINDOW, step=1)
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
        
        # Estabilidad
        st.markdown("### 🎯 Estabilidad")
        
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
        col3.metric("Desv. Std", f"{stability['std_corr']:.3f}")
        
        # Períodos históricos
        st.markdown("### 📅 Períodos Históricos")
        
        historical_df = calculate_historical_periods(df_all_prices, asset1, asset2)
        
        if len(historical_df) > 0:
            st.dataframe(
                historical_df.style.format({
                    'correlation': '{:.3f}',
                    'spread_mean': '{:.4f}',
                    'spread_std': '{:.4f}',
                    'spread_min': '{:.4f}',
                    'spread_max': '{:.4f}'
                }),
                width='stretch'
            )
        
        st.session_state.run_analysis = False
    
    else:
        st.info("👆 Selecciona activos y presiona **Analizar**")

# ============================================================================
# TAB 3: COMPARACIÓN
# ============================================================================

with tab3:
    st.header("📈 Comparación de Pares")
    
    if 'positive_pairs' in st.session_state and 'negative_pairs' in st.session_state:
        
        st.markdown("### 📈 Top 10 - Correlación Positiva")
        
        if len(st.session_state.positive_pairs) > 0:
            top_pos = st.session_state.positive_pairs.head(10).to_dict('records')
            
            with st.spinner("Generando gráficos..."):
                fig_pos = plot_multiple_rolling_correlations(df_all_prices, top_pos, window=ROLLING_WINDOW)
                st.plotly_chart(fig_pos, use_container_width=True)
        else:
            st.info("No hay pares")
        
        st.markdown("---")
        
        st.markdown("### 📉 Top 10 - Correlación Negativa")
        
        if len(st.session_state.negative_pairs) > 0:
            top_neg = st.session_state.negative_pairs.head(10).to_dict('records')
            
            with st.spinner("Generando gráficos..."):
                fig_neg = plot_multiple_rolling_correlations(df_all_prices, top_neg, window=ROLLING_WINDOW)
                st.plotly_chart(fig_neg, use_container_width=True)
        else:
            st.info("No hay pares")
        
        st.markdown("---")
        
        st.markdown("### 📊 Comparación de Métricas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(st.session_state.positive_pairs) > 0:
                st.markdown("#### Positivos")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.positive_pairs['score'],
                    y=st.session_state.positive_pairs['hurst'],
                    mode='markers',
                    marker=dict(
                        size=st.session_state.positive_pairs['years_data'] * 3,
                        color=st.session_state.positive_pairs['mean_correlation'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Corr")
                    ),
                    text=[f"{ASSETS[row['asset1']]['label']} / {ASSETS[row['asset2']]['label']}<br>{row['years_data']:.1f} años" 
                          for _, row in st.session_state.positive_pairs.iterrows()],
                    hovertemplate='<b>%{text}</b><br>Score: %{x:.1f}<br>Hurst: %{y:.3f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Score vs Hurst',
                    xaxis_title='Score',
                    yaxis_title='Hurst',
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if len(st.session_state.negative_pairs) > 0:
                st.markdown("#### Negativos")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.negative_pairs['score'],
                    y=st.session_state.negative_pairs['hurst'],
                    mode='markers',
                    marker=dict(
                        size=st.session_state.negative_pairs['years_data'] * 3,
                        color=st.session_state.negative_pairs['mean_correlation'],
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title="Corr")
                    ),
                    text=[f"{ASSETS[row['asset1']]['label']} / {ASSETS[row['asset2']]['label']}<br>{row['years_data']:.1f} años" 
                          for _, row in st.session_state.negative_pairs.iterrows()],
                    hovertemplate='<b>%{text}</b><br>Score: %{x:.1f}<br>Hurst: %{y:.3f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Score vs Hurst',
                    xaxis_title='Score',
                    yaxis_title='Hurst',
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Ejecuta la búsqueda primero")

# ============================================================================
# TAB 4: ESTACIONALIDAD
# ============================================================================

with tab4:
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
            seasonality = analyze_seasonality(df_all_prices, season_asset1, season_asset2, LOOKBACK_ANALYSIS)
        
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
            width='stretch'
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
            width='stretch'
        )
        
        st.markdown("---")
        
        # Períodos históricos
        st.markdown("### 📅 Períodos Históricos")
        
        historical_df = calculate_historical_periods(df_all_prices, season_asset1, season_asset2)
        
        if len(historical_df) > 0:
            st.dataframe(
                historical_df.style.format({
                    'correlation': '{:.3f}',
                    'spread_mean': '{:.4f}',
                    'spread_std': '{:.4f}',
                    'spread_min': '{:.4f}',
                    'spread_max': '{:.4f}'
                }),
                width='stretch'
            )
            
            # Gráfico
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=historical_df['period'],
                y=historical_df['correlation'],
                marker_color=['#10b981' if c > 0 else '#ef4444' for c in historical_df['correlation']],
                text=historical_df['correlation'].round(3),
                textposition='auto'
            ))
            
            fig.update_layout(
                title='Correlación por Período',
                xaxis_title='Período',
                yaxis_title='Correlación',
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

# ============================================================================
# TAB 5: LEAD-LAG
# ============================================================================

with tab5:
    st.header("⏱️ Análisis Lead-Lag")
    st.info("""
    **Análisis de relaciones temporales entre activos:**
    - 📊 **Cross-Correlation**: Correlación con diferentes lags
    - 📈 **Rolling Lead-Lag**: Evolución temporal del lag óptimo
    - 🔬 **Granger Causality**: Test estadístico de causalidad
    - 💥 **Impulse Response**: Respuesta a shocks
    """)
    
    # Sub-tabs para Lead-Lag
    ll_tab1, ll_tab2 = st.tabs(["📊 Análisis Individual", "🔍 Búsqueda de Pares Lead-Lag"])
    
    with ll_tab1:
        st.markdown("### 📊 Análisis Lead-Lag Individual")
        
        available_assets = list(st.session_state.all_asset_data.keys())
        
        default_ll_asset1 = st.session_state.get('selected_asset1', available_assets[0])
        default_ll_asset2 = st.session_state.get('selected_asset2', available_assets[1] if len(available_assets) > 1 else available_assets[0])
        
        if default_ll_asset2 == default_ll_asset1 and len(available_assets) > 1:
            default_ll_asset2 = available_assets[1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ll_asset1 = st.selectbox(
                "Activo 1",
                options=available_assets,
                index=available_assets.index(default_ll_asset1) if default_ll_asset1 in available_assets else 0,
                format_func=lambda x: ASSETS[x]['label'],
                key='ll_asset1'
            )
        
        with col2:
            ll_asset2_options = [a for a in available_assets if a != ll_asset1]
            ll_asset2 = st.selectbox(
                "Activo 2",
                options=ll_asset2_options,
                index=ll_asset2_options.index(default_ll_asset2) if default_ll_asset2 in ll_asset2_options else 0,
                format_func=lambda x: ASSETS[x]['label'],
                key='ll_asset2'
            )
        
        with col3:
            max_lag = st.number_input("Max Lag (días)", min_value=5, max_value=30, value=15, key='ll_max_lag')
        
        if st.button("🔄 Analizar Lead-Lag", type="primary", key='btn_ll_analyze'):
            
            with st.spinner("Calculando..."):
                prices1 = df_all_prices[ll_asset1].dropna()
                prices2 = df_all_prices[ll_asset2].dropna()
                
                common_idx = prices1.index.intersection(prices2.index)
                p1 = prices1.loc[common_idx]
                p2 = prices2.loc[common_idx]
                
                returns1 = np.log(p1 / p1.shift(1)).dropna()
                returns2 = np.log(p2 / p2.shift(1)).dropna()
                
                common_ret_idx = returns1.index.intersection(returns2.index)
                returns1 = returns1.loc[common_ret_idx]
                returns2 = returns2.loc[common_ret_idx]
                
                asset1_name = ASSETS[ll_asset1]['label']
                asset2_name = ASSETS[ll_asset2]['label']
            
            st.success("✅ Análisis completado")
            
            # Cross-Correlation
            st.markdown("### 📊 Cross-Correlation")
            
            cross_corr = calculate_cross_correlation(returns1, returns2, max_lag=max_lag)
            
            st.plotly_chart(
                plot_cross_correlation(cross_corr, asset1_name, asset2_name),
                use_container_width=True
            )
            
            # Métricas de Cross-Correlation
            idx_max = cross_corr['correlation'].abs().idxmax()
            optimal_lag = cross_corr.loc[idx_max, 'lag']
            max_corr = cross_corr.loc[idx_max, 'correlation']
            contemp_corr = cross_corr[cross_corr['lag'] == 0]['correlation'].values[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Lag Óptimo", f"{optimal_lag} días")
            col2.metric("Corr. Óptima", f"{max_corr:.4f}")
            col3.metric("Corr. Contemporánea", f"{contemp_corr:.4f}")
            col4.metric("Mejora", f"{abs(max_corr) - abs(contemp_corr):.4f}")
            
            # Interpretación
            if optimal_lag > 0:
                st.success(f"📈 **{asset1_name}** LIDERA a **{asset2_name}** por {optimal_lag} días")
            elif optimal_lag < 0:
                st.success(f"📈 **{asset2_name}** LIDERA a **{asset1_name}** por {-optimal_lag} días")
            else:
                st.info("📊 **Relación contemporánea** - No hay lead-lag significativo")
            
            st.markdown("---")
            
            # Rolling Lead-Lag
            st.markdown("### 📈 Lead-Lag Rolling (60 días)")
            
            with st.spinner("Calculando rolling lead-lag..."):
                rolling_ll = calculate_rolling_lead_lag(returns1, returns2, window=60, max_lag=10)
            
            st.plotly_chart(
                plot_rolling_lead_lag(rolling_ll, asset1_name, asset2_name),
                use_container_width=True
            )
            
            # Estadísticas del rolling
            col1, col2, col3, col4 = st.columns(4)
            
            mean_lag = rolling_ll['optimal_lag'].mean()
            std_lag = rolling_ll['optimal_lag'].std()
            pct_positive = (rolling_ll['optimal_lag'] > 0).mean() * 100
            pct_negative = (rolling_ll['optimal_lag'] < 0).mean() * 100
            
            col1.metric("Lag Medio", f"{mean_lag:.2f} días")
            col2.metric("Std Lag", f"{std_lag:.2f}")
            col3.metric(f"% {asset1_name[:10]} lidera", f"{pct_positive:.1f}%")
            col4.metric(f"% {asset2_name[:10]} lidera", f"{pct_negative:.1f}%")
            
            st.markdown("---")
            
            # Granger Causality
            st.markdown("### 🔬 Test de Causalidad de Granger")
            
            with st.spinner("Ejecutando test de Granger..."):
                granger_results = granger_causality_test(returns1, returns2, max_lag=5)
            
            if granger_results:
                fig_granger = plot_granger_causality(granger_results, asset1_name, asset2_name)
                if fig_granger:
                    st.plotly_chart(fig_granger, use_container_width=True)
                
                # Interpretación Granger
                best_pval_1_to_2 = min(granger_results['asset1_causes_asset2'].values())
                best_pval_2_to_1 = min(granger_results['asset2_causes_asset1'].values())
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if best_pval_1_to_2 < 0.05:
                        st.success(f"✅ **{asset1_name}** Granger-causa a **{asset2_name}** (p={best_pval_1_to_2:.4f})")
                    else:
                        st.warning(f"❌ **{asset1_name}** NO Granger-causa a **{asset2_name}** (p={best_pval_1_to_2:.4f})")
                
                with col2:
                    if best_pval_2_to_1 < 0.05:
                        st.success(f"✅ **{asset2_name}** Granger-causa a **{asset1_name}** (p={best_pval_2_to_1:.4f})")
                    else:
                        st.warning(f"❌ **{asset2_name}** NO Granger-causa a **{asset1_name}** (p={best_pval_2_to_1:.4f})")
                
                # Tabla de p-values
                granger_df = pd.DataFrame({
                    'Lag': list(granger_results['asset1_causes_asset2'].keys()),
                    f'{asset1_name[:15]} → {asset2_name[:15]}': list(granger_results['asset1_causes_asset2'].values()),
                    f'{asset2_name[:15]} → {asset1_name[:15]}': list(granger_results['asset2_causes_asset1'].values())
                })
                
                st.dataframe(
                    granger_df.style.format({
                        f'{asset1_name[:15]} → {asset2_name[:15]}': '{:.4f}',
                        f'{asset2_name[:15]} → {asset1_name[:15]}': '{:.4f}'
                    }),
                    width='stretch'
                )
            else:
                st.warning("No se pudo ejecutar el test de Granger (datos insuficientes)")
            
            st.markdown("---")
            
            # Impulse Response
            st.markdown("### 💥 Impulse Response")
            
            with st.spinner("Calculando impulse response..."):
                ir_results = calculate_impulse_response(returns1, returns2, periods=20)
            
            st.plotly_chart(
                plot_impulse_response(ir_results, asset1_name, asset2_name),
                use_container_width=True
            )
            
            col1, col2 = st.columns(2)
            col1.metric(f"Shocks en {asset1_name[:15]}", ir_results['n_shocks_asset1'])
            col2.metric(f"Shocks en {asset2_name[:15]}", ir_results['n_shocks_asset2'])
        
        else:
            st.info("👆 Selecciona activos y presiona **Analizar Lead-Lag**")
    
    with ll_tab2:
        st.markdown("### 🔍 Búsqueda de Pares con Lead-Lag Significativo")
        
        st.info("""
        **Criterios de búsqueda:**
        - Mejora de correlación con lag vs contemporánea
        - Significancia en test de Granger
        - Lead-lag consistente (no contemporáneo)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            ll_min_corr = st.slider("Correlación Mínima", 0.2, 0.8, 0.3, 0.05, key='ll_search_min_corr')
        
        with col2:
            ll_search_max_lag = st.number_input("Max Lag Búsqueda", min_value=5, max_value=20, value=10, key='ll_search_max_lag')
        
        if st.button("🚀 Buscar Pares Lead-Lag", type="primary", key='btn_ll_search'):
            
            with st.spinner("Buscando pares con lead-lag significativo..."):
                lead_lag_pairs = find_lead_lag_pairs(
                    df_all_prices,
                    min_correlation=ll_min_corr,
                    max_lag=ll_search_max_lag
                )
            
            st.session_state.lead_lag_pairs = lead_lag_pairs
            st.success(f"✅ Encontrados {len(lead_lag_pairs)} pares con lead-lag")
        
        if 'lead_lag_pairs' in st.session_state and len(st.session_state.lead_lag_pairs) > 0:
            
            display_ll = st.session_state.lead_lag_pairs.head(30).copy()
            
            display_ll['Activo 1'] = display_ll['asset1'].apply(lambda x: ASSETS[x]['label'])
            display_ll['Activo 2'] = display_ll['asset2'].apply(lambda x: ASSETS[x]['label'])
            display_ll['Líder'] = display_ll['leader'].apply(lambda x: ASSETS[x]['label'] if x else '-')
            display_ll['✓ Granger 1→2'] = display_ll['granger_significant_1_to_2'].apply(lambda x: '✅' if x else '❌')
            display_ll['✓ Granger 2→1'] = display_ll['granger_significant_2_to_1'].apply(lambda x: '✅' if x else '❌')
            
            table_ll = display_ll[[
                'Activo 1', 'Activo 2', 'Líder', 'lead_days', 'score',
                'max_correlation', 'contemp_correlation', 'correlation_improvement',
                '✓ Granger 1→2', '✓ Granger 2→1', 'years_data'
            ]].rename(columns={
                'lead_days': 'Lead (días)',
                'score': 'Score',
                'max_correlation': 'Corr Óptima',
                'contemp_correlation': 'Corr Contemp',
                'correlation_improvement': 'Mejora',
                'years_data': 'Años'
            })
            
            st.dataframe(
                table_ll.style.format({
                    'Lead (días)': '{:.0f}',
                    'Score': '{:.1f}',
                    'Corr Óptima': '{:.3f}',
                    'Corr Contemp': '{:.3f}',
                    'Mejora': '{:.3f}',
                    'Años': '{:.1f}'
                }),
                width='stretch',
                height=600
            )
            
            st.metric("Total pares encontrados", len(st.session_state.lead_lag_pairs))
            
            # Seleccionar para análisis
            pair_options_ll = [f"{row['Activo 1']} / {row['Activo 2']}" 
                              for _, row in display_ll.iterrows()]
            
            selected_ll_pair = st.selectbox(
                "Seleccionar para análisis detallado",
                options=pair_options_ll,
                key='select_ll_pair'
            )
            
            if st.button("📊 Analizar Detalle", key='btn_analyze_ll_detail'):
                idx = pair_options_ll.index(selected_ll_pair)
                selected_row = display_ll.iloc[idx]
                st.session_state.selected_asset1 = selected_row['asset1']
                st.session_state.selected_asset2 = selected_row['asset2']
                st.success(f"✅ {selected_ll_pair} - Ve al tab 'Análisis Individual' arriba")
        
        elif 'lead_lag_pairs' in st.session_state:
            st.warning("No se encontraron pares con lead-lag significativo")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Guía")
st.sidebar.markdown(f"""
**Flujo:**
1. 🔍 Búsqueda de pares
2. 📊 Análisis individual
3. 📈 Comparación múltiple
4. 📅 Estacionalidad
5. ⏱️ **Lead-Lag** (NUEVO)

**Ventana fija: {ROLLING_WINDOW} días**
**DXY incluido**
""")

st.sidebar.info("📊 Análisis estadístico puro")
