import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import correlate
from scipy.stats import spearmanr
import warnings
import pickle
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Lead-Lag Analysis - Pairs Trading",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1e2130; padding: 15px; border-radius: 10px;}
    h1, h2, h3 {color: #ffffff;}
    .leader-card {
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
# FUNCIONES DE LEAD-LAG ANALYSIS
# ============================================================================

def calculate_cross_correlation(returns1, returns2, max_lag=20):
    """
    Calcula correlación cruzada entre dos series para diferentes lags.
    
    Lag positivo: returns1 lidera (returns2 sigue)
    Lag negativo: returns2 lidera (returns1 sigue)
    
    Returns:
        DataFrame con lags y correlaciones
    """
    correlations = []
    
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            # returns1 adelantado (lidera)
            r1 = returns1.iloc[:-lag].values
            r2 = returns2.iloc[lag:].values
        elif lag < 0:
            # returns2 adelantado (lidera)
            r1 = returns1.iloc[-lag:].values
            r2 = returns2.iloc[:lag].values
        else:
            r1 = returns1.values
            r2 = returns2.values
        
        if len(r1) > 10:
            corr = np.corrcoef(r1, r2)[0, 1]
        else:
            corr = np.nan
        
        correlations.append({
            'lag': lag,
            'correlation': corr
        })
    
    return pd.DataFrame(correlations)


def calculate_rolling_lead_lag(returns1, returns2, window=60, max_lag=10):
    """
    Calcula el lag óptimo en ventana rolling para detectar cambios de liderazgo.
    
    Returns:
        DataFrame con fecha, lag óptimo, correlación máxima, y líder
    """
    results = []
    
    for i in range(window + max_lag, len(returns1)):
        window_r1 = returns1.iloc[i-window-max_lag:i]
        window_r2 = returns2.iloc[i-window-max_lag:i]
        
        best_lag = 0
        best_corr = -1
        
        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                r1 = window_r1.iloc[max_lag:-lag if lag > 0 else None].values
                r2 = window_r2.iloc[max_lag+lag:].values
            elif lag < 0:
                r1 = window_r1.iloc[max_lag-lag:].values
                r2 = window_r2.iloc[max_lag:lag if lag < 0 else None].values
            else:
                r1 = window_r1.iloc[max_lag:].values
                r2 = window_r2.iloc[max_lag:].values
            
            if len(r1) > 10 and len(r2) > 10:
                min_len = min(len(r1), len(r2))
                corr = abs(np.corrcoef(r1[:min_len], r2[:min_len])[0, 1])
                
                if corr > best_corr:
                    best_corr = corr
                    # Calcular correlación con signo para el mejor lag
                    best_lag = lag
                    signed_corr = np.corrcoef(r1[:min_len], r2[:min_len])[0, 1]
        
        # Determinar líder
        if best_lag > 0:
            leader = 'asset1'
        elif best_lag < 0:
            leader = 'asset2'
        else:
            leader = 'simultaneous'
        
        results.append({
            'date': returns1.index[i],
            'optimal_lag': best_lag,
            'max_correlation': best_corr,
            'signed_correlation': signed_corr if 'signed_corr' in dir() else best_corr,
            'leader': leader
        })
    
    return pd.DataFrame(results)


def calculate_granger_causality_simple(returns1, returns2, max_lag=5):
    """
    Versión simplificada de test de causalidad de Granger.
    Compara R² de modelos con y sin lags del otro activo.
    
    Returns:
        Dict con estadísticas de causalidad en ambas direcciones
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    # Crear matrices de lags
    n = len(returns1)
    
    # returns1 -> returns2 (¿returns1 causa returns2?)
    X_base = np.column_stack([returns2.shift(i).values for i in range(1, max_lag + 1)])
    X_full = np.column_stack([
        X_base,
        *[returns1.shift(i).values for i in range(1, max_lag + 1)]
    ])
    y = returns2.values
    
    # Eliminar NaN
    valid_idx = ~np.isnan(X_full).any(axis=1) & ~np.isnan(y)
    X_base_clean = X_base[valid_idx][:, ~np.isnan(X_base[valid_idx]).any(axis=0)]
    X_full_clean = X_full[valid_idx]
    y_clean = y[valid_idx]
    
    if len(y_clean) < max_lag * 3:
        return None
    
    # Modelos
    model_base = LinearRegression().fit(X_base_clean, y_clean)
    model_full = LinearRegression().fit(X_full_clean, y_clean)
    
    r2_base_1to2 = r2_score(y_clean, model_base.predict(X_base_clean))
    r2_full_1to2 = r2_score(y_clean, model_full.predict(X_full_clean))
    
    # returns2 -> returns1 (¿returns2 causa returns1?)
    X_base = np.column_stack([returns1.shift(i).values for i in range(1, max_lag + 1)])
    X_full = np.column_stack([
        X_base,
        *[returns2.shift(i).values for i in range(1, max_lag + 1)]
    ])
    y = returns1.values
    
    valid_idx = ~np.isnan(X_full).any(axis=1) & ~np.isnan(y)
    X_base_clean = X_base[valid_idx][:, ~np.isnan(X_base[valid_idx]).any(axis=0)]
    X_full_clean = X_full[valid_idx]
    y_clean = y[valid_idx]
    
    model_base = LinearRegression().fit(X_base_clean, y_clean)
    model_full = LinearRegression().fit(X_full_clean, y_clean)
    
    r2_base_2to1 = r2_score(y_clean, model_base.predict(X_base_clean))
    r2_full_2to1 = r2_score(y_clean, model_full.predict(X_full_clean))
    
    # Mejora en R²
    improvement_1to2 = r2_full_1to2 - r2_base_1to2
    improvement_2to1 = r2_full_2to1 - r2_base_2to1
    
    return {
        'asset1_causes_asset2': {
            'r2_base': r2_base_1to2,
            'r2_full': r2_full_1to2,
            'improvement': improvement_1to2,
            'causes': improvement_1to2 > 0.01  # Umbral arbitrario
        },
        'asset2_causes_asset1': {
            'r2_base': r2_base_2to1,
            'r2_full': r2_full_2to1,
            'improvement': improvement_2to1,
            'causes': improvement_2to1 > 0.01
        },
        'dominant_leader': 'asset1' if improvement_1to2 > improvement_2to1 else 'asset2',
        'bidirectional': improvement_1to2 > 0.01 and improvement_2to1 > 0.01
    }


def analyze_leadership_stability(lead_lag_df):
    """
    Analiza la estabilidad del liderazgo en el tiempo.
    """
    if len(lead_lag_df) < 10:
        return None
    
    # Conteo de liderazgo
    leadership_counts = lead_lag_df['leader'].value_counts()
    total = len(lead_lag_df)
    
    # Estadísticas de lag
    lag_stats = lead_lag_df['optimal_lag'].describe()
    
    # Cambios de liderazgo
    leadership_changes = (lead_lag_df['leader'] != lead_lag_df['leader'].shift(1)).sum()
    
    # Rachas de liderazgo
    lead_lag_df['leadership_streak'] = (lead_lag_df['leader'] != lead_lag_df['leader'].shift(1)).cumsum()
    streak_lengths = lead_lag_df.groupby('leadership_streak').size()
    
    return {
        'leadership_distribution': {
            'asset1_pct': leadership_counts.get('asset1', 0) / total * 100,
            'asset2_pct': leadership_counts.get('asset2', 0) / total * 100,
            'simultaneous_pct': leadership_counts.get('simultaneous', 0) / total * 100,
        },
        'lag_statistics': {
            'mean': lag_stats['mean'],
            'std': lag_stats['std'],
            'median': lag_stats['50%'],
            'min': lag_stats['min'],
            'max': lag_stats['max'],
        },
        'stability': {
            'total_changes': leadership_changes,
            'change_frequency': leadership_changes / total * 100,
            'avg_streak_length': streak_lengths.mean(),
            'max_streak_length': streak_lengths.max(),
        },
        'dominant_leader': leadership_counts.idxmax() if len(leadership_counts) > 0 else 'unknown'
    }


def calculate_lead_lag_by_regime(returns1, returns2, max_lag=10):
    """
    Calcula lead-lag en diferentes regímenes de mercado:
    - Alta volatilidad vs baja volatilidad
    - Mercados alcistas vs bajistas
    """
    # Volatilidad rolling
    vol = returns1.rolling(20).std()
    vol_median = vol.median()
    
    high_vol_mask = vol > vol_median
    low_vol_mask = vol <= vol_median
    
    # Mercados alcistas/bajistas (basado en retornos acumulados de 20 días)
    cum_returns = returns1.rolling(20).sum()
    bull_mask = cum_returns > 0
    bear_mask = cum_returns <= 0
    
    results = {}
    
    for regime_name, mask in [
        ('high_volatility', high_vol_mask),
        ('low_volatility', low_vol_mask),
        ('bull_market', bull_mask),
        ('bear_market', bear_mask)
    ]:
        r1_regime = returns1[mask].dropna()
        r2_regime = returns2[mask].dropna()
        
        if len(r1_regime) > max_lag * 3:
            # Encontrar lag óptimo para este régimen
            common_idx = r1_regime.index.intersection(r2_regime.index)
            r1_aligned = r1_regime.loc[common_idx]
            r2_aligned = r2_regime.loc[common_idx]
            
            best_lag = 0
            best_corr = -1
            
            for lag in range(-max_lag, max_lag + 1):
                if lag > 0:
                    r1 = r1_aligned.iloc[:-lag].values
                    r2 = r2_aligned.iloc[lag:].values
                elif lag < 0:
                    r1 = r1_aligned.iloc[-lag:].values
                    r2 = r2_aligned.iloc[:lag].values
                else:
                    r1 = r1_aligned.values
                    r2 = r2_aligned.values
                
                if len(r1) > 10:
                    corr = abs(np.corrcoef(r1, r2)[0, 1])
                    if corr > best_corr:
                        best_corr = corr
                        best_lag = lag
            
            results[regime_name] = {
                'optimal_lag': best_lag,
                'max_correlation': best_corr,
                'leader': 'asset1' if best_lag > 0 else ('asset2' if best_lag < 0 else 'simultaneous'),
                'n_observations': len(r1_aligned)
            }
        else:
            results[regime_name] = None
    
    return results


def find_pairs_with_lead_lag(df, min_correlation=0.4, max_lag=10, lookback=252):
    """
    Encuentra pares con relaciones lead-lag significativas.
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
            if len(common_idx) < lookback:
                continue
            
            # Usar solo últimos N días
            common_idx = common_idx[-lookback:]
            
            returns1 = np.log(prices1.loc[common_idx] / prices1.loc[common_idx].shift(1)).dropna()
            returns2 = np.log(prices2.loc[common_idx] / prices2.loc[common_idx].shift(1)).dropna()
            
            common_idx_returns = returns1.index.intersection(returns2.index)
            returns1 = returns1.loc[common_idx_returns]
            returns2 = returns2.loc[common_idx_returns]
            
            if len(returns1) < 100:
                continue
            
            # Correlación base
            base_corr = returns1.corr(returns2)
            
            if abs(base_corr) < min_correlation:
                continue
            
            # Cross-correlation
            cross_corr = calculate_cross_correlation(returns1, returns2, max_lag)
            
            # Encontrar lag óptimo
            best_row = cross_corr.loc[cross_corr['correlation'].abs().idxmax()]
            optimal_lag = int(best_row['lag'])
            max_corr = best_row['correlation']
            
            # Mejora sobre correlación base
            improvement = abs(max_corr) - abs(base_corr)
            
            # Determinar líder (guardamos tanto el identificador como el asset real)
            if optimal_lag > 0:
                leader_id = 'asset1'
                leader = asset1
                follower = asset2
            elif optimal_lag < 0:
                leader_id = 'asset2'
                leader = asset2
                follower = asset1
            else:
                leader_id = 'simultaneous'
                leader = 'simultaneous'
                follower = 'simultaneous'
            
            # Score basado en:
            # - Mejora de correlación con lag
            # - Consistencia del lag
            # - Magnitud de la correlación
            
            score = 0
            
            # Mejora de correlación
            if improvement > 0.05:
                score += 30
            elif improvement > 0.02:
                score += 20
            elif improvement > 0.01:
                score += 10
            
            # Magnitud de correlación
            if abs(max_corr) > 0.7:
                score += 30
            elif abs(max_corr) > 0.5:
                score += 20
            elif abs(max_corr) > 0.3:
                score += 10
            
            # Lag significativo (no cero)
            if abs(optimal_lag) > 0 and abs(optimal_lag) <= 5:
                score += 20  # Lag pequeño y significativo
            elif abs(optimal_lag) > 5:
                score += 10  # Lag más grande
            
            # Calcular estabilidad del lead-lag
            if len(returns1) >= 100:
                lead_lag_rolling = calculate_rolling_lead_lag(returns1, returns2, window=60, max_lag=5)
                if len(lead_lag_rolling) > 0:
                    stability = analyze_leadership_stability(lead_lag_rolling)
                    if stability:
                        # Bonus por estabilidad
                        if stability['stability']['change_frequency'] < 20:
                            score += 15
                        elif stability['stability']['change_frequency'] < 40:
                            score += 8
                else:
                    stability = None
            else:
                stability = None
            
            # Determinar el porcentaje del líder
            if stability and leader_id != 'simultaneous':
                leader_pct = stability['leadership_distribution'][f'{leader_id}_pct']
            else:
                leader_pct = 0
            
            candidates.append({
                'asset1': asset1,
                'asset2': asset2,
                'score': score,
                'base_correlation': base_corr,
                'optimal_lag': optimal_lag,
                'max_correlation': max_corr,
                'improvement': improvement,
                'leader': leader,
                'follower': follower,
                'leader_pct': leader_pct,
                'change_frequency': stability['stability']['change_frequency'] if stability else np.nan,
                'avg_streak': stability['stability']['avg_streak_length'] if stability else np.nan,
            })
    
    progress_bar.empty()
    status_text.empty()
    
    if len(candidates) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(candidates).sort_values('score', ascending=False)


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def plot_cross_correlation(cross_corr_df, asset1_name, asset2_name):
    """Gráfico de correlación cruzada"""
    fig = go.Figure()
    
    colors = ['#ef4444' if lag < 0 else '#10b981' if lag > 0 else '#3b82f6' 
              for lag in cross_corr_df['lag']]
    
    fig.add_trace(go.Bar(
        x=cross_corr_df['lag'],
        y=cross_corr_df['correlation'],
        marker_color=colors,
        hovertemplate='Lag: %{x}<br>Correlación: %{y:.4f}<extra></extra>'
    ))
    
    # Línea en lag = 0
    fig.add_vline(x=0, line_dash="dash", line_color="#ffffff", opacity=0.5)
    
    # Encontrar máximo
    max_idx = cross_corr_df['correlation'].abs().idxmax()
    max_lag = cross_corr_df.loc[max_idx, 'lag']
    max_corr = cross_corr_df.loc[max_idx, 'correlation']
    
    fig.add_annotation(
        x=max_lag,
        y=max_corr,
        text=f"Óptimo: Lag={max_lag}, ρ={max_corr:.3f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#f59e0b",
        font=dict(color="#f59e0b")
    )
    
    fig.update_layout(
        title=f'Correlación Cruzada: {asset1_name} vs {asset2_name}<br><sup>Lag > 0: {asset1_name} lidera | Lag < 0: {asset2_name} lidera</sup>',
        xaxis_title='Lag (días)',
        yaxis_title='Correlación',
        template='plotly_dark',
        height=400,
    )
    
    return fig


def plot_rolling_lead_lag(lead_lag_df, asset1_name, asset2_name):
    """Gráfico de lag óptimo rolling"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Lag Óptimo (Rolling)', 'Correlación Máxima'),
        row_heights=[0.6, 0.4]
    )
    
    # Colorear por líder
    colors = ['#10b981' if l == 'asset1' else '#ef4444' if l == 'asset2' else '#3b82f6' 
              for l in lead_lag_df['leader']]
    
    # Lag óptimo
    fig.add_trace(go.Scatter(
        x=lead_lag_df['date'],
        y=lead_lag_df['optimal_lag'],
        mode='lines+markers',
        marker=dict(color=colors, size=4),
        line=dict(color='#8b5cf6', width=1),
        name='Lag Óptimo',
        hovertemplate='%{x}<br>Lag: %{y}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="#ffffff", opacity=0.3, row=1, col=1)
    
    # Correlación
    fig.add_trace(go.Scatter(
        x=lead_lag_df['date'],
        y=lead_lag_df['max_correlation'],
        mode='lines',
        line=dict(color='#3b82f6', width=2),
        name='Correlación',
        hovertemplate='%{x}<br>Corr: %{y:.3f}<extra></extra>'
    ), row=2, col=1)
    
    fig.update_layout(
        title=f'Análisis Lead-Lag Rolling: {asset1_name} vs {asset2_name}',
        template='plotly_dark',
        height=500,
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Lag (días)", row=1, col=1)
    fig.update_yaxes(title_text="Correlación", row=2, col=1)
    
    return fig


def plot_leadership_distribution(stability_data, asset1_name, asset2_name):
    """Gráfico de distribución de liderazgo"""
    fig = go.Figure()
    
    labels = [asset1_name, asset2_name, 'Simultáneo']
    values = [
        stability_data['leadership_distribution']['asset1_pct'],
        stability_data['leadership_distribution']['asset2_pct'],
        stability_data['leadership_distribution']['simultaneous_pct']
    ]
    colors = ['#10b981', '#ef4444', '#3b82f6']
    
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.4,
        textinfo='label+percent',
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Distribución de Liderazgo',
        template='plotly_dark',
        height=400,
    )
    
    return fig


def plot_lag_histogram(lead_lag_df):
    """Histograma de distribución de lags"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=lead_lag_df['optimal_lag'],
        nbinsx=21,
        marker_color='#8b5cf6',
        opacity=0.7
    ))
    
    mean_lag = lead_lag_df['optimal_lag'].mean()
    fig.add_vline(x=mean_lag, line_dash="solid", line_color="#f59e0b",
                  annotation_text=f"Media: {mean_lag:.1f}", annotation_position="top")
    fig.add_vline(x=0, line_dash="dash", line_color="#ffffff", opacity=0.5)
    
    fig.update_layout(
        title='Distribución del Lag Óptimo',
        xaxis_title='Lag (días)',
        yaxis_title='Frecuencia',
        template='plotly_dark',
        height=350,
    )
    
    return fig


def plot_regime_lead_lag(regime_results, asset1_name, asset2_name):
    """Gráfico de lead-lag por régimen"""
    fig = go.Figure()
    
    regimes = []
    lags = []
    correlations = []
    colors = []
    
    regime_labels = {
        'high_volatility': 'Alta Volatilidad',
        'low_volatility': 'Baja Volatilidad',
        'bull_market': 'Mercado Alcista',
        'bear_market': 'Mercado Bajista'
    }
    
    for regime, data in regime_results.items():
        if data is not None:
            regimes.append(regime_labels.get(regime, regime))
            lags.append(data['optimal_lag'])
            correlations.append(data['max_correlation'])
            
            if data['leader'] == 'asset1':
                colors.append('#10b981')
            elif data['leader'] == 'asset2':
                colors.append('#ef4444')
            else:
                colors.append('#3b82f6')
    
    fig.add_trace(go.Bar(
        x=regimes,
        y=lags,
        marker_color=colors,
        text=[f"ρ={c:.2f}" for c in correlations],
        textposition='outside',
        hovertemplate='%{x}<br>Lag: %{y}<br>Corr: %{text}<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#ffffff", opacity=0.3)
    
    fig.update_layout(
        title=f'Lag Óptimo por Régimen de Mercado<br><sup>Verde: {asset1_name} lidera | Rojo: {asset2_name} lidera</sup>',
        xaxis_title='Régimen',
        yaxis_title='Lag Óptimo (días)',
        template='plotly_dark',
        height=400,
    )
    
    return fig


def plot_returns_with_lag(returns1, returns2, optimal_lag, asset1_name, asset2_name):
    """Visualiza retornos con el lag aplicado"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            'Retornos Originales (Sin Lag)',
            f'Retornos con Lag={optimal_lag} Aplicado'
        )
    )
    
    # Original
    fig.add_trace(go.Scatter(
        x=returns1.index,
        y=returns1,
        name=asset1_name,
        line=dict(color='#10b981', width=1),
        opacity=0.7
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=returns2.index,
        y=returns2,
        name=asset2_name,
        line=dict(color='#ef4444', width=1),
        opacity=0.7
    ), row=1, col=1)
    
    # Con lag
    if optimal_lag > 0:
        r1_lagged = returns1.iloc[:-optimal_lag]
        r2_lagged = returns2.iloc[optimal_lag:]
        r2_lagged.index = r1_lagged.index
    elif optimal_lag < 0:
        r1_lagged = returns1.iloc[-optimal_lag:]
        r2_lagged = returns2.iloc[:optimal_lag]
        r1_lagged.index = r2_lagged.index
    else:
        r1_lagged = returns1
        r2_lagged = returns2
    
    fig.add_trace(go.Scatter(
        x=r1_lagged.index,
        y=r1_lagged,
        name=f'{asset1_name} (original)',
        line=dict(color='#10b981', width=1),
        opacity=0.7,
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=r1_lagged.index,
        y=r2_lagged.values,
        name=f'{asset2_name} (shifted)',
        line=dict(color='#ef4444', width=1),
        opacity=0.7,
        showlegend=False
    ), row=2, col=1)
    
    fig.update_layout(
        title=f'Comparación de Retornos: Original vs Con Lag',
        template='plotly_dark',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

st.title("🔄 Lead-Lag Analysis - Pairs Trading")
st.markdown("**Detecta qué activo lidera y cuál sigue usando correlación cruzada**")
st.info("📊 **Cross-Correlation** | 🔄 **Rolling Lead-Lag** | 📈 **Análisis por Régimen**")

# ============================================================================
# SIDEBAR - GESTIÓN DE DATOS
# ============================================================================

st.sidebar.header("💾 Gestión de Datos")

cache_info = get_cache_info()

if cache_info:
    st.sidebar.success("✅ Datos en cache")
    st.sidebar.metric("Última actualización", cache_info['last_update'].strftime('%Y-%m-%d %H:%M'))
    st.sidebar.metric("Total activos", cache_info['total_assets'])
    
    if 'all_asset_data' not in st.session_state:
        with st.spinner("Cargando datos desde cache..."):
            data, metadata = load_data_from_cache()
            if data and metadata:
                st.session_state.all_asset_data = data
                st.session_state.metadata = metadata
else:
    st.sidebar.warning("⚠️ No hay datos descargados")
    
    if st.sidebar.button("📥 Descargar Datos", type="primary"):
        with st.spinner(f"Descargando {len(ASSETS)} activos..."):
            all_data, metadata = download_all_assets(delay=3, start_date='2015-01-01')
        
        if len(all_data) > 0:
            if save_data_to_cache(all_data, metadata):
                st.success(f"✅ Descargados {len(all_data)} activos")
                st.session_state.all_asset_data = all_data
                st.session_state.metadata = metadata
                st.rerun()

if 'all_asset_data' not in st.session_state:
    st.info("""
    ### 👋 Lead-Lag Analysis
    
    **¿Qué es Lead-Lag?**
    
    En pairs trading, uno de los activos puede **liderar** (moverse primero) mientras el otro **sigue** 
    (reacciona después). Detectar esto permite:
    
    - 🎯 Predecir movimientos del activo rezagado
    - ⚡ Mejores puntos de entrada
    - 📊 Entender la dinámica del par
    
    **Herramientas disponibles:**
    - 📈 Correlación cruzada con lags
    - 🔄 Detección rolling de liderazgo
    - 📊 Análisis por régimen (volatilidad, tendencia)
    - 🎯 Búsqueda de pares con lead-lag significativo
    
    👉 Descarga los datos para comenzar
    """)
    st.stop()

# ============================================================================
# PARÁMETROS
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parámetros")

max_lag = st.sidebar.slider("Lag Máximo (días)", 5, 30, 15, 1)
rolling_window = st.sidebar.slider("Window Rolling", 30, 120, 60, 10)
min_correlation = st.sidebar.slider("Correlación Mínima (búsqueda)", 0.2, 0.7, 0.4, 0.05)
lookback_days = st.sidebar.slider("Lookback (días)", 126, 504, 252, 21)

# Crear DataFrame
df_all_prices = merge_asset_data(st.session_state.all_asset_data)

if df_all_prices.empty:
    st.error("No hay datos suficientes")
    st.stop()

st.success(f"✅ {len(df_all_prices)} días | {len(df_all_prices.columns)} activos")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3 = st.tabs([
    "🔍 Búsqueda Lead-Lag",
    "📊 Análisis Individual",
    "📈 Análisis por Régimen"
])

# ============================================================================
# TAB 1: BÚSQUEDA
# ============================================================================

with tab1:
    st.header("🔍 Búsqueda de Pares con Lead-Lag Significativo")
    
    st.info("""
    **Criterios de Score:**
    - 📈 Mejora de correlación con lag (hasta 30 pts)
    - 🎯 Magnitud de correlación (hasta 30 pts)
    - ⏱️ Lag significativo (hasta 20 pts)
    - 🔒 Estabilidad del liderazgo (hasta 15 pts)
    """)
    
    if st.button("🚀 Buscar Pares con Lead-Lag", type="primary"):
        with st.spinner("Analizando relaciones lead-lag..."):
            lead_lag_pairs = find_pairs_with_lead_lag(
                df_all_prices,
                min_correlation=min_correlation,
                max_lag=max_lag,
                lookback=lookback_days
            )
        
        st.session_state.lead_lag_pairs = lead_lag_pairs
        st.success("✅ Búsqueda completada!")
    
    if 'lead_lag_pairs' in st.session_state and len(st.session_state.lead_lag_pairs) > 0:
        
        st.markdown("### 🏆 Top Pares con Lead-Lag")
        
        display_df = st.session_state.lead_lag_pairs.head(30).copy()
        display_df['Líder'] = display_df['leader'].apply(
            lambda x: ASSETS[x]['label'] if x in ASSETS else x
        )
        display_df['Seguidor'] = display_df['follower'].apply(
            lambda x: ASSETS[x]['label'] if x in ASSETS else x
        )
        display_df['Activo 1'] = display_df['asset1'].apply(lambda x: ASSETS[x]['label'])
        display_df['Activo 2'] = display_df['asset2'].apply(lambda x: ASSETS[x]['label'])
        
        # Indicador de dirección
        display_df['Dirección'] = display_df['optimal_lag'].apply(
            lambda x: '→' if x > 0 else ('←' if x < 0 else '↔')
        )
        
        table = display_df[[
            'Activo 1', 'Dirección', 'Activo 2', 'score', 
            'optimal_lag', 'base_correlation', 'max_correlation', 
            'improvement', 'leader_pct', 'change_frequency'
        ]].rename(columns={
            'score': 'Score',
            'optimal_lag': 'Lag',
            'base_correlation': 'Corr Base',
            'max_correlation': 'Corr Max',
            'improvement': 'Mejora',
            'leader_pct': '% Líder',
            'change_frequency': '% Cambios'
        })
        
        st.dataframe(
            table.style.format({
                'Score': '{:.1f}',
                'Lag': '{:d}',
                'Corr Base': '{:.3f}',
                'Corr Max': '{:.3f}',
                'Mejora': '{:.4f}',
                '% Líder': '{:.1f}',
                '% Cambios': '{:.1f}'
            }).background_gradient(subset=['Score'], cmap='Greens'),
            height=600,
            use_container_width=True
        )
        
        st.markdown("""
        **Interpretación:**
        - **→**: Activo 1 lidera (lag positivo)
        - **←**: Activo 2 lidera (lag negativo)
        - **↔**: Simultáneo (lag = 0)
        - **Mejora**: Ganancia de correlación al aplicar el lag óptimo
        """)
        
        # Selección para análisis
        st.markdown("---")
        st.markdown("### 🔬 Seleccionar para Análisis Detallado")
        
        pair_options = [f"{row['Activo 1']} {row['Dirección']} {row['Activo 2']} (Lag={row['Lag']})" 
                       for _, row in display_df.head(20).iterrows()]
        
        selected_pair = st.selectbox("Seleccionar par", options=pair_options)
        
        if st.button("📊 Analizar Este Par"):
            idx = pair_options.index(selected_pair)
            selected_row = display_df.iloc[idx]
            st.session_state.selected_lead_lag_asset1 = selected_row['asset1']
            st.session_state.selected_lead_lag_asset2 = selected_row['asset2']
            st.info("👉 Ve a 'Análisis Individual' o 'Análisis por Régimen'")
    
    else:
        if 'lead_lag_pairs' in st.session_state:
            st.warning("No se encontraron pares con las condiciones especificadas")

# ============================================================================
# TAB 2: ANÁLISIS INDIVIDUAL
# ============================================================================

with tab2:
    st.header("📊 Análisis Individual de Lead-Lag")
    
    available_assets = list(st.session_state.all_asset_data.keys())
    
    default_asset1 = st.session_state.get('selected_lead_lag_asset1', available_assets[0])
    default_asset2 = st.session_state.get('selected_lead_lag_asset2', 
                                          available_assets[1] if len(available_assets) > 1 else available_assets[0])
    
    col1, col2 = st.columns(2)
    
    with col1:
        asset1 = st.selectbox(
            "Activo 1",
            options=available_assets,
            index=available_assets.index(default_asset1) if default_asset1 in available_assets else 0,
            format_func=lambda x: ASSETS[x]['label'],
            key='ll_asset1'
        )
    
    with col2:
        asset2_options = [a for a in available_assets if a != asset1]
        asset2 = st.selectbox(
            "Activo 2",
            options=asset2_options,
            index=asset2_options.index(default_asset2) if default_asset2 in asset2_options else 0,
            format_func=lambda x: ASSETS[x]['label'],
            key='ll_asset2'
        )
    
    if st.button("🔄 Analizar Lead-Lag", type="primary", key='btn_analyze_ll'):
        
        prices1 = df_all_prices[asset1].dropna()
        prices2 = df_all_prices[asset2].dropna()
        
        common_idx = prices1.index.intersection(prices2.index)
        prices1 = prices1.loc[common_idx]
        prices2 = prices2.loc[common_idx]
        
        returns1 = np.log(prices1 / prices1.shift(1)).dropna()
        returns2 = np.log(prices2 / prices2.shift(1)).dropna()
        
        common_idx_returns = returns1.index.intersection(returns2.index)
        returns1 = returns1.loc[common_idx_returns]
        returns2 = returns2.loc[common_idx_returns]
        
        asset1_name = ASSETS[asset1]['label']
        asset2_name = ASSETS[asset2]['label']
        
        # 1. Cross-Correlation Estática
        st.markdown("### 📊 Correlación Cruzada")
        
        cross_corr = calculate_cross_correlation(returns1, returns2, max_lag)
        st.plotly_chart(
            plot_cross_correlation(cross_corr, asset1_name, asset2_name),
            use_container_width=True
        )
        
        # Métricas
        best_row = cross_corr.loc[cross_corr['correlation'].abs().idxmax()]
        optimal_lag = int(best_row['lag'])
        max_corr = best_row['correlation']
        base_corr = cross_corr[cross_corr['lag'] == 0]['correlation'].values[0]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Correlación Base (lag=0)", f"{base_corr:.4f}")
        col2.metric("Lag Óptimo", f"{optimal_lag} días")
        col3.metric("Correlación Máxima", f"{max_corr:.4f}")
        col4.metric("Mejora", f"{abs(max_corr) - abs(base_corr):.4f}")
        
        # Interpretación
        if optimal_lag > 0:
            st.success(f"**🎯 {asset1_name} LIDERA** → {asset2_name} sigue con {optimal_lag} días de retraso")
        elif optimal_lag < 0:
            st.success(f"**🎯 {asset2_name} LIDERA** → {asset1_name} sigue con {abs(optimal_lag)} días de retraso")
        else:
            st.info("**↔ Movimiento simultáneo** - No hay un líder claro")
        
        st.markdown("---")
        
        # 2. Rolling Lead-Lag
        st.markdown("### 🔄 Análisis Rolling (Dinámico)")
        
        with st.spinner("Calculando lead-lag rolling..."):
            lead_lag_rolling = calculate_rolling_lead_lag(
                returns1, returns2, 
                window=rolling_window, 
                max_lag=min(max_lag, 10)
            )
        
        if len(lead_lag_rolling) > 0:
            st.plotly_chart(
                plot_rolling_lead_lag(lead_lag_rolling, asset1_name, asset2_name),
                use_container_width=True
            )
            
            # Estadísticas de estabilidad
            stability = analyze_leadership_stability(lead_lag_rolling)
            
            if stability:
                st.markdown("### 📈 Estabilidad del Liderazgo")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(
                        plot_leadership_distribution(stability, asset1_name, asset2_name),
                        use_container_width=True
                    )
                
                with col2:
                    st.plotly_chart(
                        plot_lag_histogram(lead_lag_rolling),
                        use_container_width=True
                    )
                
                # Métricas de estabilidad
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric(
                    f"% {asset1_name[:15]} lidera",
                    f"{stability['leadership_distribution']['asset1_pct']:.1f}%"
                )
                col2.metric(
                    f"% {asset2_name[:15]} lidera",
                    f"{stability['leadership_distribution']['asset2_pct']:.1f}%"
                )
                col3.metric(
                    "Frecuencia de Cambios",
                    f"{stability['stability']['change_frequency']:.1f}%",
                    delta="Estable ✅" if stability['stability']['change_frequency'] < 30 else "Inestable ⚠️"
                )
                col4.metric(
                    "Racha Promedio",
                    f"{stability['stability']['avg_streak_length']:.1f} días"
                )
                
                st.markdown("---")
                
                # Estadísticas del lag
                st.markdown("### 📊 Estadísticas del Lag")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Media", f"{stability['lag_statistics']['mean']:.2f}")
                col2.metric("Mediana", f"{stability['lag_statistics']['median']:.1f}")
                col3.metric("Desv. Std", f"{stability['lag_statistics']['std']:.2f}")
                col4.metric("Mínimo", f"{stability['lag_statistics']['min']:.0f}")
                col5.metric("Máximo", f"{stability['lag_statistics']['max']:.0f}")
        
        st.markdown("---")
        
        # 3. Visualización de Retornos con Lag
        st.markdown("### 📉 Retornos: Original vs Con Lag Aplicado")
        
        st.plotly_chart(
            plot_returns_with_lag(returns1, returns2, optimal_lag, asset1_name, asset2_name),
            use_container_width=True
        )
        
        # Correlación antes y después
        if optimal_lag != 0:
            if optimal_lag > 0:
                r1_aligned = returns1.iloc[:-optimal_lag].values
                r2_aligned = returns2.iloc[optimal_lag:].values
            else:
                r1_aligned = returns1.iloc[-optimal_lag:].values
                r2_aligned = returns2.iloc[:optimal_lag].values
            
            corr_after = np.corrcoef(r1_aligned, r2_aligned)[0, 1]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Correlación Original", f"{base_corr:.4f}")
            col2.metric("Correlación con Lag", f"{corr_after:.4f}")
            col3.metric("Mejora", f"{(abs(corr_after) - abs(base_corr))*100:.2f}%")
        
        st.markdown("---")
        
        # 4. Causalidad de Granger (simplificada)
        st.markdown("### 🔬 Análisis de Causalidad (Granger Simplificado)")
        
        granger = calculate_granger_causality_simple(returns1, returns2, max_lag=5)
        
        if granger:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### {asset1_name} → {asset2_name}")
                r2_improve = granger['asset1_causes_asset2']['improvement'] * 100
                st.metric(
                    "Mejora R²",
                    f"{r2_improve:.2f}%",
                    delta="Causal ✅" if granger['asset1_causes_asset2']['causes'] else "No Causal"
                )
            
            with col2:
                st.markdown(f"#### {asset2_name} → {asset1_name}")
                r2_improve = granger['asset2_causes_asset1']['improvement'] * 100
                st.metric(
                    "Mejora R²",
                    f"{r2_improve:.2f}%",
                    delta="Causal ✅" if granger['asset2_causes_asset1']['causes'] else "No Causal"
                )
            
            # Conclusión
            if granger['bidirectional']:
                st.warning("⚠️ **Causalidad Bidireccional** - Ambos activos se influencian mutuamente")
            else:
                dominant = granger['dominant_leader']
                st.success(f"**🎯 Líder Dominante: {ASSETS[asset1]['label'] if dominant == 'asset1' else ASSETS[asset2]['label']}**")

# ============================================================================
# TAB 3: ANÁLISIS POR RÉGIMEN
# ============================================================================

with tab3:
    st.header("📈 Análisis Lead-Lag por Régimen de Mercado")
    
    st.info("""
    **¿Por qué importa el régimen?**
    
    El liderazgo puede cambiar según las condiciones del mercado:
    - 📈 **Mercado Alcista**: Un activo puede liderar en tendencias alcistas
    - 📉 **Mercado Bajista**: El liderazgo puede invertirse en caídas
    - 🌊 **Alta Volatilidad**: La dinámica puede cambiar en crisis
    - 😴 **Baja Volatilidad**: Comportamiento diferente en mercados calmados
    """)
    
    available_assets = list(st.session_state.all_asset_data.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        regime_asset1 = st.selectbox(
            "Activo 1",
            options=available_assets,
            format_func=lambda x: ASSETS[x]['label'],
            key='regime_asset1'
        )
    
    with col2:
        regime_asset2_options = [a for a in available_assets if a != regime_asset1]
        regime_asset2 = st.selectbox(
            "Activo 2",
            options=regime_asset2_options,
            format_func=lambda x: ASSETS[x]['label'],
            key='regime_asset2'
        )
    
    if st.button("📊 Analizar por Régimen", type="primary"):
        
        prices1 = df_all_prices[regime_asset1].dropna()
        prices2 = df_all_prices[regime_asset2].dropna()
        
        common_idx = prices1.index.intersection(prices2.index)
        prices1 = prices1.loc[common_idx]
        prices2 = prices2.loc[common_idx]
        
        returns1 = np.log(prices1 / prices1.shift(1)).dropna()
        returns2 = np.log(prices2 / prices2.shift(1)).dropna()
        
        common_idx_returns = returns1.index.intersection(returns2.index)
        returns1 = returns1.loc[common_idx_returns]
        returns2 = returns2.loc[common_idx_returns]
        
        asset1_name = ASSETS[regime_asset1]['label']
        asset2_name = ASSETS[regime_asset2]['label']
        
        # Análisis por régimen
        with st.spinner("Analizando por régimen..."):
            regime_results = calculate_lead_lag_by_regime(returns1, returns2, max_lag=max_lag)
        
        # Gráfico principal
        st.plotly_chart(
            plot_regime_lead_lag(regime_results, asset1_name, asset2_name),
            use_container_width=True
        )
        
        # Tabla detallada
        st.markdown("### 📋 Detalle por Régimen")
        
        regime_labels = {
            'high_volatility': '🌊 Alta Volatilidad',
            'low_volatility': '😴 Baja Volatilidad',
            'bull_market': '📈 Mercado Alcista',
            'bear_market': '📉 Mercado Bajista'
        }
        
        regime_data = []
        for regime, data in regime_results.items():
            if data:
                leader_name = asset1_name if data['leader'] == 'asset1' else (
                    asset2_name if data['leader'] == 'asset2' else 'Simultáneo'
                )
                regime_data.append({
                    'Régimen': regime_labels.get(regime, regime),
                    'Lag Óptimo': data['optimal_lag'],
                    'Correlación': data['max_correlation'],
                    'Líder': leader_name,
                    'Observaciones': data['n_observations']
                })
        
        if regime_data:
            regime_df = pd.DataFrame(regime_data)
            st.dataframe(
                regime_df.style.format({
                    'Lag Óptimo': '{:d}',
                    'Correlación': '{:.3f}',
                    'Observaciones': '{:,}'
                }),
                use_container_width=True
            )
        
        # Análisis
        st.markdown("### 💡 Interpretación")
        
        # Comparar volatilidad
        if regime_results.get('high_volatility') and regime_results.get('low_volatility'):
            hv = regime_results['high_volatility']
            lv = regime_results['low_volatility']
            
            if hv['leader'] != lv['leader']:
                st.warning(f"""
                **⚠️ Cambio de Liderazgo según Volatilidad:**
                - Alta volatilidad: **{asset1_name if hv['leader'] == 'asset1' else asset2_name}** lidera
                - Baja volatilidad: **{asset1_name if lv['leader'] == 'asset1' else asset2_name}** lidera
                
                Esto sugiere que la dinámica del par cambia en períodos de estrés.
                """)
            else:
                st.success(f"""
                **✅ Liderazgo Consistente en Volatilidad:**
                - **{asset1_name if hv['leader'] == 'asset1' else asset2_name}** lidera tanto en alta como baja volatilidad
                """)
        
        # Comparar tendencia
        if regime_results.get('bull_market') and regime_results.get('bear_market'):
            bull = regime_results['bull_market']
            bear = regime_results['bear_market']
            
            if bull['leader'] != bear['leader']:
                st.warning(f"""
                **⚠️ Cambio de Liderazgo según Tendencia:**
                - Mercado alcista: **{asset1_name if bull['leader'] == 'asset1' else asset2_name}** lidera
                - Mercado bajista: **{asset1_name if bear['leader'] == 'asset1' else asset2_name}** lidera
                
                El líder se invierte según la dirección del mercado.
                """)
            else:
                st.success(f"""
                **✅ Liderazgo Consistente en Tendencia:**
                - **{asset1_name if bull['leader'] == 'asset1' else asset2_name}** lidera en ambas direcciones
                """)
        
        # Recomendaciones para trading
        st.markdown("---")
        st.markdown("### 🎯 Recomendaciones para Trading")
        
        consistent_leader = None
        for regime, data in regime_results.items():
            if data and data['leader'] != 'simultaneous':
                if consistent_leader is None:
                    consistent_leader = data['leader']
                elif consistent_leader != data['leader']:
                    consistent_leader = 'mixed'
                    break
        
        if consistent_leader and consistent_leader != 'mixed':
            leader_name = asset1_name if consistent_leader == 'asset1' else asset2_name
            follower_name = asset2_name if consistent_leader == 'asset1' else asset1_name
            
            # Calcular lag promedio
            avg_lag = np.mean([data['optimal_lag'] for data in regime_results.values() if data])
            
            st.success(f"""
            **📊 Estrategia Sugerida:**
            
            1. **Líder Consistente**: {leader_name}
            2. **Seguidor**: {follower_name}
            3. **Lag Promedio**: {avg_lag:.1f} días
            
            **Implementación:**
            - Monitorear señales en {leader_name}
            - Esperar confirmación/divergencia en {follower_name}
            - Considerar el lag de ~{abs(avg_lag):.0f} días para timing de entradas
            """)
        else:
            st.info("""
            **⚠️ Sin Líder Consistente**
            
            El liderazgo varía según el régimen de mercado. Considera:
            - Adaptar la estrategia al régimen actual
            - Usar indicadores de régimen para switching
            - Mayor cautela en las señales
            """)

# Footer
st.sidebar.markdown("---")
st.sidebar.header("📚 Guía")
st.sidebar.markdown("""
**Conceptos Clave:**

**Lead-Lag**: Un activo se mueve 
primero (líder) y otro sigue 
(rezagado).

**Lag Positivo**: Activo 1 lidera
**Lag Negativo**: Activo 2 lidera

**Cross-Correlation**: 
Correlación calculada con 
diferentes desplazamientos 
temporales.

**Uso en Trading:**
- Predecir movimientos del 
  activo rezagado
- Mejorar timing de entradas
- Confirmar divergencias
""")
