import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import pickle
import os
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.stats.diagnostic import acorr_ljungbox

# Configuración de la página
st.set_page_config(
    page_title="Rolling Correlation Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Configuración de activos
ASSETS = {
    # Índices de Acciones (Risk On)
    'sp500': {'label': 'S&P 500', 'symbol': '^GSPC', 'color': '#3b82f6', 'risk': 'Risk On'},
    'nasdaq': {'label': 'NASDAQ', 'symbol': '^IXIC', 'color': '#8b5cf6', 'risk': 'Risk On'},
    'dow': {'label': 'Dow Jones', 'symbol': '^DJI', 'color': '#10b981', 'risk': 'Risk On'},
    'russell': {'label': 'Russell 2000', 'symbol': '^RUT', 'color': '#06b6d4', 'risk': 'Risk On'},
    'emerging': {'label': 'MSCI Emerging Markets', 'symbol': 'EEM', 'color': '#ec4899', 'risk': 'Risk On'},
    
    # Divisas (Risk On/Off)
    'dxy': {'label': 'DXY (Dólar Index)', 'symbol': 'DX-Y.NYB', 'color': '#f59e0b', 'risk': 'Risk Off'},
    'usdjpy': {'label': 'USD/JPY', 'symbol': 'JPY=X', 'color': '#ef4444', 'risk': 'Risk Off'},
    'usdchf': {'label': 'USD/CHF', 'symbol': 'CHF=X', 'color': '#dc2626', 'risk': 'Risk Off'},
    'audusd': {'label': 'AUD/USD', 'symbol': 'AUDUSD=X', 'color': '#10b981', 'risk': 'Risk On'},
    'nzdusd': {'label': 'NZD/USD', 'symbol': 'NZDUSD=X', 'color': '#059669', 'risk': 'Risk On'},
    'eurusd': {'label': 'EUR/USD', 'symbol': 'EURUSD=X', 'color': '#3b82f6', 'risk': 'Neutral'},
    
    # Metales Preciosos (Risk Off)
    'gold': {'label': 'Oro', 'symbol': 'GC=F', 'color': '#fbbf24', 'risk': 'Risk Off'},
    'silver': {'label': 'Plata', 'symbol': 'SI=F', 'color': '#d1d5db', 'risk': 'Risk On'},
    
    # Commodities (Risk On)
    'oil': {'label': 'Petróleo WTI', 'symbol': 'CL=F', 'color': '#000000', 'risk': 'Risk On'},
    'copper': {'label': 'Cobre', 'symbol': 'HG=F', 'color': '#c2410c', 'risk': 'Risk On'},
    'natgas': {'label': 'Gas Natural', 'symbol': 'NG=F', 'color': '#059669', 'risk': 'Risk On'},
    
    # Bonos (Risk Off)
    'us10y': {'label': 'Treasury 10Y', 'symbol': '^TNX', 'color': '#ef4444', 'risk': 'Risk Off'},
    'us2y': {'label': 'Treasury 2Y', 'symbol': '^IRX', 'color': '#dc2626', 'risk': 'Risk Off'},
    'tlt': {'label': 'TLT (20Y+ Treasury)', 'symbol': 'TLT', 'color': '#b91c1c', 'risk': 'Risk Off'},
    
    # Volatilidad (Risk Off)
    'vix': {'label': 'VIX (Volatilidad)', 'symbol': '^VIX', 'color': '#ec4899', 'risk': 'Risk Off'},
    
    # Criptomonedas (Risk On)
    'btc': {'label': 'Bitcoin', 'symbol': 'BTC-USD', 'color': '#f7931a', 'risk': 'Risk On'},
    'eth': {'label': 'Ethereum', 'symbol': 'ETH-USD', 'color': '#627eea', 'risk': 'Risk On'},
    
    # ETFs Sectoriales
    'qqq': {'label': 'QQQ (Nasdaq ETF)', 'symbol': 'QQQ', 'color': '#8b5cf6', 'risk': 'Risk On'},
    'iwm': {'label': 'IWM (Russell 2000 ETF)', 'symbol': 'IWM', 'color': '#06b6d4', 'risk': 'Risk On'},
    'eem': {'label': 'EEM (Emerging Markets)', 'symbol': 'EEM', 'color': '#ec4899', 'risk': 'Risk On'},
}

@st.cache_data(ttl=3600)
def fetch_asset_data(symbol, start_date='2000-01-01', end_date=None):
    """Descarga datos históricos de un activo"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            return None
        
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        prices = prices.dropna()
        
        return prices
        
    except Exception as e:
        st.error(f"Error descargando {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def download_selected_assets(selected_keys, delay=1):
    """Descarga solo los activos seleccionados"""
    all_data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, key in enumerate(selected_keys):
        asset_info = ASSETS[key]
        symbol = asset_info['symbol']
        
        status_text.text(f"Descargando {asset_info['label']} ({idx+1}/{len(selected_keys)})...")
        
        data = fetch_asset_data(symbol)
        
        if data is not None:
            all_data[key] = data
        
        progress_bar.progress((idx + 1) / len(selected_keys))
        
        if idx < len(selected_keys) - 1:
            time.sleep(delay)
    
    progress_bar.empty()
    status_text.empty()
    
    return all_data

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

def calculate_rolling_correlation(df, asset1, asset2, window=30, step=5):
    """Calcula la correlación móvil entre dos activos"""
    correlations = []
    dates = []
    
    for i in range(window, len(df), step):
        window_data = df.iloc[i-window:i]
        corr = window_data[asset1].corr(window_data[asset2])
        correlations.append(corr)
        dates.append(df.index[i])
    
    return pd.DataFrame({'date': dates, 'correlation': correlations})

def plot_rolling_correlation(corr_df, asset1_name, asset2_name, asset1_color, asset2_color):
    """Crea un gráfico interactivo de la correlación móvil"""
    fig = go.Figure()
    
    # Línea de correlación
    fig.add_trace(go.Scatter(
        x=corr_df['date'],
        y=corr_df['correlation'],
        mode='lines',
        name=f'{asset1_name} vs {asset2_name}',
        line=dict(color='#3b82f6', width=3),
        hovertemplate='%{x}<br>Correlación: %{y:.4f}<extra></extra>'
    ))
    
    # Líneas de referencia
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", 
                  annotation_text="Neutral", annotation_position="right")
    fig.add_hline(y=0.5, line_dash="dot", line_color="#10b981", opacity=0.5)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="#ef4444", opacity=0.5)
    
    # Sombreado de zonas
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
    
    # Normalizar precios (base 100)
    norm1 = (df[asset1] / df[asset1].iloc[0]) * 100
    norm2 = (df[asset2] / df[asset2].iloc[0]) * 100
    
    fig.add_trace(
        go.Scatter(x=df.index, y=norm1, name=asset1_name, 
                   line=dict(color=ASSETS[asset1]['color'], width=2)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=norm2, name=asset2_name, 
                   line=dict(color=ASSETS[asset2]['color'], width=2)),
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

def plot_correlation_heatmap(df, selected_assets):
    """Crea un heatmap de correlaciones entre todos los activos"""
    corr_matrix = df[selected_assets].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[ASSETS[a]['label'] for a in selected_assets],
        y=[ASSETS[a]['label'] for a in selected_assets],
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlación")
    ))
    
    fig.update_layout(
        title='Matriz de Correlaciones',
        template='plotly_dark',
        height=600,
        xaxis={'side': 'bottom'}
    )
    
    return fig

# =============================================================================
# FUNCIONES ESTADÍSTICAS AVANZADAS PARA TRADING
# =============================================================================

def calculate_returns(prices):
    """Calcula retornos logarítmicos"""
    return np.log(prices / prices.shift(1)).dropna()

def calculate_volatility(returns, window=30, method='historical'):
    """
    Calcula volatilidad con diferentes métodos
    - historical: Desviación estándar clásica
    - ewma: Exponentially Weighted Moving Average
    - parkinson: High-Low estimator
    - garman_klass: OHLC estimator
    """
    if method == 'historical':
        return returns.rolling(window).std() * np.sqrt(252)
    elif method == 'ewma':
        return returns.ewm(span=window).std() * np.sqrt(252)
    else:
        return returns.rolling(window).std() * np.sqrt(252)

def calculate_sharpe_ratio(returns, rf_rate=0.02, window=252):
    """Calcula el Sharpe Ratio rolling"""
    excess_returns = returns - rf_rate/252
    rolling_mean = excess_returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return (rolling_mean / rolling_std) * np.sqrt(252)

def calculate_sortino_ratio(returns, rf_rate=0.02, window=252):
    """Calcula el Sortino Ratio (penaliza solo downside volatility)"""
    excess_returns = returns - rf_rate/252
    rolling_mean = excess_returns.rolling(window).mean()
    downside_returns = returns[returns < 0]
    downside_std = returns.rolling(window).apply(lambda x: x[x < 0].std())
    return (rolling_mean / downside_std) * np.sqrt(252)

def calculate_max_drawdown(prices):
    """Calcula el Maximum Drawdown"""
    cumulative = (1 + calculate_returns(prices)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown

def calculate_calmar_ratio(returns, prices, window=252):
    """Calcula el Calmar Ratio (retorno/max drawdown)"""
    annual_return = returns.rolling(window).mean() * 252
    drawdown = calculate_max_drawdown(prices)
    max_dd = drawdown.rolling(window).min().abs()
    return annual_return / max_dd

def calculate_var(returns, confidence=0.95, window=252):
    """Calcula Value at Risk (VaR)"""
    return returns.rolling(window).quantile(1 - confidence)

def calculate_cvar(returns, confidence=0.95, window=252):
    """Calcula Conditional VaR (Expected Shortfall)"""
    var = calculate_var(returns, confidence, window)
    return returns.rolling(window).apply(
        lambda x: x[x <= x.quantile(1 - confidence)].mean()
    )

def calculate_beta(returns_asset, returns_market, window=252):
    """Calcula Beta vs mercado"""
    covariance = returns_asset.rolling(window).cov(returns_market)
    market_variance = returns_market.rolling(window).var()
    return covariance / market_variance

def calculate_alpha(returns_asset, returns_market, rf_rate=0.02, window=252):
    """Calcula Alpha (Jensen's Alpha)"""
    beta = calculate_beta(returns_asset, returns_market, window)
    asset_return = returns_asset.rolling(window).mean() * 252
    market_return = returns_market.rolling(window).mean() * 252
    return asset_return - (rf_rate + beta * (market_return - rf_rate))

def calculate_information_ratio(returns_asset, returns_benchmark, window=252):
    """Calcula Information Ratio"""
    active_return = returns_asset - returns_benchmark
    tracking_error = active_return.rolling(window).std() * np.sqrt(252)
    return (active_return.rolling(window).mean() * 252) / tracking_error

def calculate_omega_ratio(returns, threshold=0, window=252):
    """Calcula Omega Ratio"""
    def omega(ret_series):
        returns_above = ret_series[ret_series > threshold].sum()
        returns_below = abs(ret_series[ret_series < threshold].sum())
        return returns_above / returns_below if returns_below != 0 else np.nan
    
    return returns.rolling(window).apply(omega)

def test_cointegration(prices1, prices2):
    """Test de cointegración de Engle-Granger"""
    try:
        score, pvalue, _ = coint(prices1, prices2)
        return {'score': score, 'pvalue': pvalue, 'cointegrated': pvalue < 0.05}
    except:
        return {'score': np.nan, 'pvalue': np.nan, 'cointegrated': False}

def calculate_spread(prices1, prices2):
    """Calcula el spread entre dos activos (para pairs trading)"""
    # Regresión lineal para encontrar hedge ratio
    prices1_clean = prices1.dropna()
    prices2_clean = prices2.dropna()
    
    # Alinear índices
    common_idx = prices1_clean.index.intersection(prices2_clean.index)
    prices1_clean = prices1_clean.loc[common_idx]
    prices2_clean = prices2_clean.loc[common_idx]
    
    hedge_ratio = np.polyfit(prices2_clean, prices1_clean, 1)[0]
    spread = prices1_clean - hedge_ratio * prices2_clean
    return spread, hedge_ratio

def calculate_zscore(series, window=30):
    """Calcula Z-Score rolling"""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

def calculate_half_life(spread):
    """Calcula half-life del mean reversion"""
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    spread_lag = spread_lag.dropna()
    spread_diff = spread_diff.dropna()
    
    # Alinear
    common_idx = spread_lag.index.intersection(spread_diff.index)
    spread_lag = spread_lag.loc[common_idx]
    spread_diff = spread_diff.loc[common_idx]
    
    model = np.polyfit(spread_lag, spread_diff, 1)
    half_life = -np.log(2) / model[0] if model[0] < 0 else np.nan
    return half_life

def adf_test(series):
    """Augmented Dickey-Fuller test para estacionariedad"""
    try:
        result = adfuller(series.dropna())
        return {
            'adf_stat': result[0],
            'pvalue': result[1],
            'stationary': result[1] < 0.05
        }
    except:
        return {'adf_stat': np.nan, 'pvalue': np.nan, 'stationary': False}

def calculate_hurst_exponent(series, max_lag=100):
    """
    Calcula el Hurst Exponent
    H < 0.5: Mean reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    """
    lags = range(2, min(max_lag, len(series)//2))
    tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
    
    try:
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return np.nan

def calculate_rsi(prices, window=14):
    """Calcula Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calcula MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return pd.DataFrame({
        'macd': macd,
        'signal': signal_line,
        'histogram': histogram
    })

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calcula Bollinger Bands"""
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return pd.DataFrame({
        'upper': upper,
        'middle': sma,
        'lower': lower,
        'bandwidth': (upper - lower) / sma
    })

def calculate_skewness(returns, window=252):
    """Calcula Skewness rolling"""
    return returns.rolling(window).skew()

def calculate_kurtosis(returns, window=252):
    """Calcula Kurtosis rolling (excess kurtosis)"""
    return returns.rolling(window).kurt()

def jarque_bera_test(returns):
    """Test de normalidad Jarque-Bera"""
    try:
        jb_stat, pvalue = stats.jarque_bera(returns.dropna())
        return {'jb_stat': jb_stat, 'pvalue': pvalue, 'normal': pvalue > 0.05}
    except:
        return {'jb_stat': np.nan, 'pvalue': np.nan, 'normal': False}

def detect_correlation_regimes(corr_series, threshold=0.3):
    """
    Detecta regímenes de correlación
    1: Alta correlación positiva
    0: Correlación neutral
    -1: Alta correlación negativa
    """
    regimes = pd.Series(0, index=corr_series.index)
    regimes[corr_series > threshold] = 1
    regimes[corr_series < -threshold] = -1
    return regimes

def calculate_rolling_correlation_stats(df, asset1, asset2, window=30):
    """Calcula estadísticas completas de correlación rolling"""
    returns1 = calculate_returns(df[asset1])
    returns2 = calculate_returns(df[asset2])
    
    rolling_corr = returns1.rolling(window).corr(returns2)
    
    return pd.DataFrame({
        'correlation': rolling_corr,
        'corr_std': rolling_corr.rolling(window).std(),
        'corr_mean': rolling_corr.rolling(window).mean(),
        'regime': detect_correlation_regimes(rolling_corr)
    })

def calculate_tail_ratio(returns, window=252):
    """
    Tail Ratio: ratio entre el lado derecho y izquierdo de la distribución
    >1: Más ganancias extremas que pérdidas
    """
    right_tail = returns.rolling(window).quantile(0.95)
    left_tail = returns.rolling(window).quantile(0.05).abs()
    return right_tail / left_tail

def calculate_gain_to_pain_ratio(returns, window=252):
    """Gain to Pain Ratio: suma de retornos / suma de pérdidas absolutas"""
    sum_returns = returns.rolling(window).sum()
    sum_losses = returns[returns < 0].rolling(window).sum().abs()
    return sum_returns / sum_losses

def calculate_ulcer_index(prices, window=14):
    """
    Ulcer Index: mide la profundidad y duración de drawdowns
    Menor valor = menos estrés para el inversor
    """
    drawdown = calculate_max_drawdown(prices)
    squared_dd = drawdown ** 2
    ulcer = np.sqrt(squared_dd.rolling(window).mean()) * 100
    return ulcer

# =============================================================================
# FUNCIONES DE VISUALIZACIÓN PARA ESTADÍSTICAS
# =============================================================================

def plot_risk_metrics(df, asset, returns):
    """Gráfico de métricas de riesgo"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Volatilidad Histórica', 'Value at Risk (95%)', 'Maximum Drawdown'),
        vertical_spacing=0.1
    )
    
    vol = calculate_volatility(returns, window=30)
    var = calculate_var(returns, confidence=0.95, window=252)
    dd = calculate_max_drawdown(df[asset]) * 100
    
    fig.add_trace(go.Scatter(x=vol.index, y=vol, name='Volatilidad', 
                             line=dict(color='#f59e0b', width=2)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=var.index, y=var*100, name='VaR 95%',
                             line=dict(color='#ef4444', width=2)), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name='Drawdown',
                             fill='tozeroy', line=dict(color='#dc2626', width=2)), row=3, col=1)
    
    fig.update_layout(height=800, template='plotly_dark', showlegend=False)
    fig.update_yaxes(title_text="Volatilidad Anualizada", row=1, col=1)
    fig.update_yaxes(title_text="VaR (%)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
    
    return fig

def plot_performance_ratios(returns, returns_benchmark=None):
    """Gráfico de ratios de performance"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Sharpe Ratio Rolling', 'Sortino Ratio Rolling'),
        vertical_spacing=0.15
    )
    
    sharpe = calculate_sharpe_ratio(returns, window=252)
    sortino = calculate_sortino_ratio(returns, window=252)
    
    fig.add_trace(go.Scatter(x=sharpe.index, y=sharpe, name='Sharpe',
                             line=dict(color='#10b981', width=2)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=sortino.index, y=sortino, name='Sortino',
                             line=dict(color='#3b82f6', width=2)), row=2, col=1)
    
    # Líneas de referencia
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", row=1, col=1)
    fig.add_hline(y=1, line_dash="dot", line_color="#10b981", opacity=0.5, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", row=2, col=1)
    fig.add_hline(y=1, line_dash="dot", line_color="#3b82f6", opacity=0.5, row=2, col=1)
    
    fig.update_layout(height=600, template='plotly_dark', showlegend=False)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Sortino Ratio", row=2, col=1)
    
    return fig

def plot_spread_analysis(prices1, prices2, asset1_name, asset2_name):
    """Análisis completo de spread para pairs trading"""
    spread, hedge_ratio = calculate_spread(prices1, prices2)
    zscore = calculate_zscore(spread, window=30)
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f'Spread: {asset1_name} - {hedge_ratio:.4f} * {asset2_name}',
            'Z-Score del Spread',
            'Precios Normalizados'
        ),
        vertical_spacing=0.1
    )
    
    # Spread
    fig.add_trace(go.Scatter(x=spread.index, y=spread, name='Spread',
                             line=dict(color='#3b82f6', width=2)), row=1, col=1)
    
    # Z-Score
    fig.add_trace(go.Scatter(x=zscore.index, y=zscore, name='Z-Score',
                             line=dict(color='#8b5cf6', width=2)), row=2, col=1)
    
    # Bandas de trading
    fig.add_hline(y=2, line_dash="dash", line_color="#ef4444", row=2, col=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="#10b981", row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#666666", row=2, col=1)
    
    # Precios normalizados
    norm1 = (prices1 / prices1.iloc[0]) * 100
    norm2 = (prices2 / prices2.iloc[0]) * 100
    
    fig.add_trace(go.Scatter(x=norm1.index, y=norm1, name=asset1_name,
                             line=dict(color='#10b981', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=norm2.index, y=norm2, name=asset2_name,
                             line=dict(color='#ef4444', width=2)), row=3, col=1)
    
    fig.update_layout(height=900, template='plotly_dark')
    fig.update_yaxes(title_text="Spread", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    fig.update_yaxes(title_text="Base 100", row=3, col=1)
    
    return fig

def plot_distribution_analysis(returns):
    """Análisis de distribución de retornos"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Histograma de Retornos', 'Q-Q Plot', 'Skewness Rolling', 'Kurtosis Rolling'),
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Histograma
    fig.add_trace(go.Histogram(x=returns.dropna()*100, nbinsx=50, name='Retornos',
                               marker_color='#3b82f6'), row=1, col=1)
    
    # Q-Q Plot
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns.dropna())))
    sample_quantiles = np.sort(returns.dropna())
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                             mode='markers', name='Q-Q Plot',
                             marker=dict(color='#10b981', size=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles,
                             mode='lines', name='Normal',
                             line=dict(color='#ef4444', dash='dash')), row=1, col=2)
    
    # Skewness
    skew = calculate_skewness(returns, window=252)
    fig.add_trace(go.Scatter(x=skew.index, y=skew, name='Skewness',
                             line=dict(color='#f59e0b', width=2)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", row=2, col=1)
    
    # Kurtosis
    kurt = calculate_kurtosis(returns, window=252)
    fig.add_trace(go.Scatter(x=kurt.index, y=kurt, name='Kurtosis',
                             line=dict(color='#ec4899', width=2)), row=2, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", row=2, col=2)
    
    fig.update_layout(height=800, template='plotly_dark', showlegend=False)
    fig.update_xaxes(title_text="Retorno (%)", row=1, col=1)
    fig.update_xaxes(title_text="Quantiles Teóricos", row=1, col=2)
    fig.update_yaxes(title_text="Quantiles Observados", row=1, col=2)
    
    return fig

def plot_technical_indicators(prices):
    """Gráfico de indicadores técnicos"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Precio y Bollinger Bands', 'RSI', 'MACD'),
        vertical_spacing=0.1,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Bollinger Bands
    bb = calculate_bollinger_bands(prices)
    fig.add_trace(go.Scatter(x=prices.index, y=prices, name='Precio',
                             line=dict(color='#ffffff', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb.index, y=bb['upper'], name='Upper BB',
                             line=dict(color='#ef4444', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb.index, y=bb['middle'], name='SMA',
                             line=dict(color='#fbbf24', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb.index, y=bb['lower'], name='Lower BB',
                             line=dict(color='#10b981', width=1, dash='dash')), row=1, col=1)
    
    # RSI
    rsi = calculate_rsi(prices)
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name='RSI',
                             line=dict(color='#8b5cf6', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#10b981", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#666666", row=2, col=1)
    
    # MACD
    macd = calculate_macd(prices)
    fig.add_trace(go.Scatter(x=macd.index, y=macd['macd'], name='MACD',
                             line=dict(color='#3b82f6', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=macd.index, y=macd['signal'], name='Signal',
                             line=dict(color='#ef4444', width=1)), row=3, col=1)
    fig.add_trace(go.Bar(x=macd.index, y=macd['histogram'], name='Histogram',
                         marker_color='#10b981'), row=3, col=1)
    
    fig.update_layout(height=900, template='plotly_dark')
    fig.update_yaxes(title_text="Precio", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================

st.title("📊 Rolling Correlation Analyzer")
st.markdown("Analiza correlaciones dinámicas entre activos financieros en tiempo real")

# Sidebar - Configuración
st.sidebar.header("⚙️ Configuración")

# Selección de activos
st.sidebar.subheader("Activos a Analizar")
available_assets = list(ASSETS.keys())
default_assets = ['sp500', 'gold', 'btc', 'dxy', 'vix']

selected_assets = st.sidebar.multiselect(
    "Selecciona activos (mín. 2)",
    options=available_assets,
    default=[a for a in default_assets if a in available_assets],
    format_func=lambda x: ASSETS[x]['label']
)

if len(selected_assets) < 2:
    st.warning("⚠️ Selecciona al menos 2 activos para continuar")
    st.stop()

# Parámetros de correlación
st.sidebar.subheader("Parámetros")
window_size = st.sidebar.slider("Ventana de correlación (días)", 10, 90, 30, 5)
step_size = st.sidebar.slider("Paso de recálculo (días)", 1, 10, 5, 1)

# Rango de fechas
st.sidebar.subheader("Rango de Datos")
date_range = st.sidebar.selectbox(
    "Período",
    ['Todo', '5 años', '3 años', '2 años', '1 año', '6 meses']
)

# Botón de descarga
if st.sidebar.button("🔄 Actualizar Datos", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Descargar datos
with st.spinner("Descargando datos..."):
    asset_data = download_selected_assets(selected_assets, delay=1)

if not asset_data:
    st.error("No se pudieron descargar los datos. Intenta nuevamente.")
    st.stop()

# Combinar datos
df_prices = merge_asset_data(asset_data)

if df_prices.empty:
    st.error("No hay suficientes datos comunes entre los activos seleccionados.")
    st.stop()

# Filtrar por rango de fechas
if date_range != 'Todo':
    days_map = {'6 meses': 126, '1 año': 252, '2 años': 504, '3 años': 756, '5 años': 1260}
    days = days_map[date_range]
    df_prices = df_prices.iloc[-days:]

st.success(f"✅ Datos cargados: {len(df_prices)} días desde {df_prices.index[0].date()} hasta {df_prices.index[-1].date()}")

# =============================================================================
# TAB 1: ANÁLISIS DE PARES
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Análisis de Pares", 
    "🔥 Heatmap", 
    "📊 Estadísticas Básicas",
    "⚡ Métricas de Riesgo",
    "🎯 Pairs Trading",
    "📉 Análisis Técnico"
])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        asset1 = st.selectbox(
            "Activo 1",
            options=selected_assets,
            format_func=lambda x: ASSETS[x]['label'],
            key='asset1'
        )
    
    with col2:
        asset2 = st.selectbox(
            "Activo 2",
            options=[a for a in selected_assets if a != asset1],
            format_func=lambda x: ASSETS[x]['label'],
            key='asset2'
        )
    
    # Calcular correlación
    corr_df = calculate_rolling_correlation(df_prices, asset1, asset2, window_size, step_size)
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    
    current_corr = corr_df['correlation'].iloc[-1]
    mean_corr = corr_df['correlation'].mean()
    max_corr = corr_df['correlation'].max()
    min_corr = corr_df['correlation'].min()
    
    col1.metric("Correlación Actual", f"{current_corr:.4f}")
    col2.metric("Correlación Media", f"{mean_corr:.4f}")
    col3.metric("Máxima", f"{max_corr:.4f}")
    col4.metric("Mínima", f"{min_corr:.4f}")
    
    # Gráfico de correlación
    st.plotly_chart(
        plot_rolling_correlation(corr_df, ASSETS[asset1]['label'], 
                                ASSETS[asset2]['label'],
                                ASSETS[asset1]['color'], 
                                ASSETS[asset2]['color']),
        use_container_width=True
    )
    
    # Gráfico de precios
    st.plotly_chart(
        plot_price_comparison(df_prices, asset1, asset2, 
                            ASSETS[asset1]['label'], 
                            ASSETS[asset2]['label']),
        use_container_width=True
    )

with tab2:
    st.subheader("Matriz de Correlaciones entre Todos los Activos")
    st.plotly_chart(
        plot_correlation_heatmap(df_prices, selected_assets),
        use_container_width=True
    )
    
    # Mostrar tabla de correlaciones
    st.subheader("Tabla de Correlaciones")
    corr_matrix = df_prices[selected_assets].corr()
    
    # Formatear la tabla manualmente sin matplotlib
    def color_correlation(val):
        """Colorea las celdas según el valor de correlación"""
        if val >= 0.7:
            color = '#10b981'  # Verde fuerte
        elif val >= 0.3:
            color = '#84cc16'  # Verde claro
        elif val > -0.3:
            color = '#6b7280'  # Gris
        elif val > -0.7:
            color = '#f59e0b'  # Naranja
        else:
            color = '#ef4444'  # Rojo
        return f'background-color: {color}; color: white'
    
    # Aplicar estilo sin usar background_gradient
    styled_df = corr_matrix.style.applymap(color_correlation).format("{:.2f}")
    st.dataframe(styled_df, use_container_width=True)

with tab3:
    st.subheader("Estadísticas de Correlación")
    
    # Análisis de períodos
    positive = (corr_df['correlation'] > 0).sum()
    negative = (corr_df['correlation'] < 0).sum()
    strong_pos = (corr_df['correlation'] > 0.5).sum()
    strong_neg = (corr_df['correlation'] < -0.5).sum()
    total = len(corr_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("% Tiempo Correlación Positiva", f"{positive/total*100:.1f}%")
        st.metric("% Tiempo Correlación Negativa", f"{negative/total*100:.1f}%")
    
    with col2:
        st.metric("% Tiempo Fuerte Positiva (>0.5)", f"{strong_pos/total*100:.1f}%")
        st.metric("% Tiempo Fuerte Negativa (<-0.5)", f"{strong_neg/total*100:.1f}%")
    
    # Distribución de correlaciones
    st.subheader("Distribución de Correlaciones")
    fig_hist = go.Figure(data=[go.Histogram(
        x=corr_df['correlation'],
        nbinsx=50,
        marker_color='#3b82f6'
    )])
    
    fig_hist.update_layout(
        title='Distribución de Valores de Correlación',
        xaxis_title='Correlación',
        yaxis_title='Frecuencia',
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Descargar datos
    st.subheader("📥 Descargar Datos")
    csv = corr_df.to_csv(index=False)
    st.download_button(
        label="Descargar correlaciones como CSV",
        data=csv,
        file_name=f"correlacion_{asset1}_{asset2}.csv",
        mime="text/csv"
    )

with tab4:
    st.subheader("⚡ Métricas de Riesgo y Performance")
    
    # Selección de activo
    selected_asset = st.selectbox(
        "Selecciona activo para análisis de riesgo",
        options=selected_assets,
        format_func=lambda x: ASSETS[x]['label'],
        key='risk_asset'
    )
    
    returns = calculate_returns(df_prices[selected_asset])
    
    # Benchmark (para cálculo de Beta/Alpha)
    benchmark_asset = st.selectbox(
        "Benchmark (para Beta/Alpha)",
        options=[a for a in selected_assets if a != selected_asset],
        format_func=lambda x: ASSETS[x]['label'],
        key='benchmark',
        index=0 if 'sp500' not in selected_assets or selected_asset == 'sp500' else selected_assets.index('sp500') if 'sp500' in selected_assets else 0
    )
    
    returns_benchmark = calculate_returns(df_prices[benchmark_asset])
    
    # Métricas actuales
    st.markdown("### 📊 Métricas Actuales (últimos 252 días)")
    
    # Calcular métricas
    current_vol = calculate_volatility(returns, window=252).iloc[-1]
    current_sharpe = calculate_sharpe_ratio(returns, window=252).iloc[-1]
    current_sortino = calculate_sortino_ratio(returns, window=252).iloc[-1]
    max_dd = calculate_max_drawdown(df_prices[selected_asset]).min() * 100
    var_95 = calculate_var(returns, confidence=0.95, window=252).iloc[-1] * 100
    cvar_95 = calculate_cvar(returns, confidence=0.95, window=252).iloc[-1] * 100
    current_beta = calculate_beta(returns, returns_benchmark, window=252).iloc[-1]
    current_alpha = calculate_alpha(returns, returns_benchmark, window=252).iloc[-1] * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Volatilidad Anual", f"{current_vol:.2%}")
        st.metric("Sharpe Ratio", f"{current_sharpe:.3f}")
    
    with col2:
        st.metric("Sortino Ratio", f"{current_sortino:.3f}")
        st.metric("Max Drawdown", f"{max_dd:.2f}%")
    
    with col3:
        st.metric("VaR 95% (diario)", f"{var_95:.2f}%")
        st.metric("CVaR 95% (diario)", f"{cvar_95:.2f}%")
    
    with col4:
        st.metric("Beta", f"{current_beta:.3f}")
        st.metric("Alpha (anual)", f"{current_alpha:.2f}%")
    
    # Gráficos de riesgo
    st.markdown("### 📈 Evolución de Métricas de Riesgo")
    st.plotly_chart(plot_risk_metrics(df_prices, selected_asset, returns), use_container_width=True)
    
    # Gráficos de performance ratios
    st.markdown("### 🎯 Ratios de Performance")
    st.plotly_chart(plot_performance_ratios(returns, returns_benchmark), use_container_width=True)
    
    # Análisis de distribución
    st.markdown("### 📊 Análisis de Distribución de Retornos")
    
    # Calcular estadísticas de distribución
    returns_clean = returns.dropna()
    skew_current = returns_clean.skew()
    kurt_current = returns_clean.kurt()
    jb_test = jarque_bera_test(returns_clean)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Skewness", f"{skew_current:.3f}")
        st.caption("< 0: Asimetría negativa (más pérdidas extremas)")
    
    with col2:
        st.metric("Kurtosis (excess)", f"{kurt_current:.3f}")
        st.caption("> 0: Más eventos extremos que distribución normal")
    
    with col3:
        st.metric("Test Jarque-Bera", 
                 "✅ Normal" if jb_test['normal'] else "❌ No Normal",
                 f"p-value: {jb_test['pvalue']:.4f}")
        st.caption("p-value > 0.05: distribución normal")
    
    st.plotly_chart(plot_distribution_analysis(returns), use_container_width=True)
    
    # Métricas adicionales
    st.markdown("### 🔍 Métricas Adicionales")
    
    tail_ratio = calculate_tail_ratio(returns, window=252).iloc[-1]
    omega = calculate_omega_ratio(returns, window=252).iloc[-1]
    ulcer = calculate_ulcer_index(df_prices[selected_asset], window=14).iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tail Ratio", f"{tail_ratio:.3f}")
        st.caption("> 1: Más ganancias extremas que pérdidas")
    
    with col2:
        st.metric("Omega Ratio", f"{omega:.3f}")
        st.caption("> 1: Más probabilidad de ganancias")
    
    with col3:
        st.metric("Ulcer Index", f"{ulcer:.2f}")
        st.caption("< 5: Bajo estrés, > 10: Alto estrés")

with tab5:
    st.subheader("🎯 Pairs Trading & Mean Reversion Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pair_asset1 = st.selectbox(
            "Activo 1 (Pairs Trading)",
            options=selected_assets,
            format_func=lambda x: ASSETS[x]['label'],
            key='pair1'
        )
    
    with col2:
        pair_asset2 = st.selectbox(
            "Activo 2 (Pairs Trading)",
            options=[a for a in selected_assets if a != pair_asset1],
            format_func=lambda x: ASSETS[x]['label'],
            key='pair2'
        )
    
    prices1 = df_prices[pair_asset1]
    prices2 = df_prices[pair_asset2]
    
    # Test de cointegración
    st.markdown("### 🔬 Test de Cointegración")
    coint_test = test_cointegration(prices1, prices2)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cointegración", 
                 "✅ SÍ" if coint_test['cointegrated'] else "❌ NO")
        st.caption("Indica si el pair trading es viable")
    
    with col2:
        st.metric("P-Value", f"{coint_test['pvalue']:.4f}")
        st.caption("< 0.05 indica cointegración significativa")
    
    with col3:
        st.metric("Score", f"{coint_test['score']:.4f}")
        st.caption("Más negativo = más cointegrado")
    
    # Análisis del spread
    st.markdown("### 📊 Análisis del Spread")
    spread, hedge_ratio = calculate_spread(prices1, prices2)
    zscore = calculate_zscore(spread, window=30)
    half_life = calculate_half_life(spread)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Hedge Ratio", f"{hedge_ratio:.4f}")
        st.caption(f"1 unidad de {ASSETS[pair_asset1]['label']} = {hedge_ratio:.4f} unidades de {ASSETS[pair_asset2]['label']}")
    
    with col2:
        st.metric("Z-Score Actual", f"{zscore.iloc[-1]:.2f}")
        if abs(zscore.iloc[-1]) > 2:
            st.caption("🔴 Señal de trading (|Z| > 2)")
        else:
            st.caption("🟢 No hay señal")
    
    with col3:
        st.metric("Half-Life", f"{half_life:.1f} días" if not np.isnan(half_life) else "N/A")
        st.caption("Tiempo esperado para volver a la media")
    
    # Gráfico del spread
    st.plotly_chart(plot_spread_analysis(prices1, prices2, 
                                        ASSETS[pair_asset1]['label'],
                                        ASSETS[pair_asset2]['label']), 
                   use_container_width=True)
    
    # Test de estacionariedad
    st.markdown("### 📈 Test de Estacionariedad (ADF)")
    adf_spread = adf_test(spread)
    adf_asset1 = adf_test(prices1)
    adf_asset2 = adf_test(prices2)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Spread", "✅ Estacionario" if adf_spread['stationary'] else "❌ No estacionario")
        st.caption(f"ADF: {adf_spread['adf_stat']:.3f}, p: {adf_spread['pvalue']:.4f}")
    
    with col2:
        st.metric(ASSETS[pair_asset1]['label'], "✅ Estacionario" if adf_asset1['stationary'] else "❌ No estacionario")
        st.caption(f"ADF: {adf_asset1['adf_stat']:.3f}, p: {adf_asset1['pvalue']:.4f}")
    
    with col3:
        st.metric(ASSETS[pair_asset2]['label'], "✅ Estacionario" if adf_asset2['stationary'] else "❌ No estacionario")
        st.caption(f"ADF: {adf_asset2['adf_stat']:.3f}, p: {adf_asset2['pvalue']:.4f}")
    
    # Hurst Exponent
    st.markdown("### 🌊 Análisis de Mean Reversion (Hurst Exponent)")
    hurst_spread = calculate_hurst_exponent(spread.dropna())
    hurst_asset1 = calculate_hurst_exponent(prices1.dropna())
    hurst_asset2 = calculate_hurst_exponent(prices2.dropna())
    
    col1, col2, col3 = st.columns(3)
    
    def interpret_hurst(h):
        if h < 0.4:
            return "🔄 Mean Reverting", "#10b981"
        elif h < 0.6:
            return "🎲 Random Walk", "#f59e0b"
        else:
            return "📈 Trending", "#ef4444"
    
    with col1:
        interpretation, color = interpret_hurst(hurst_spread)
        st.metric("Spread Hurst", f"{hurst_spread:.3f}")
        st.markdown(f"<p style='color:{color}'>{interpretation}</p>", unsafe_allow_html=True)
    
    with col2:
        interpretation, color = interpret_hurst(hurst_asset1)
        st.metric(f"{ASSETS[pair_asset1]['label']} Hurst", f"{hurst_asset1:.3f}")
        st.markdown(f"<p style='color:{color}'>{interpretation}</p>", unsafe_allow_html=True)
    
    with col3:
        interpretation, color = interpret_hurst(hurst_asset2)
        st.metric(f"{ASSETS[pair_asset2]['label']} Hurst", f"{hurst_asset2:.3f}")
        st.markdown(f"<p style='color:{color}'>{interpretation}</p>", unsafe_allow_html=True)
    
    # Señales de Trading
    st.markdown("### 🎯 Señales de Trading Actuales")
    
    current_zscore = zscore.iloc[-1]
    
    if current_zscore > 2:
        st.error(f"🔴 VENDER {ASSETS[pair_asset1]['label']} / COMPRAR {ASSETS[pair_asset2]['label']}")
        st.caption(f"El spread está {current_zscore:.2f} desviaciones estándar por encima de la media")
    elif current_zscore < -2:
        st.success(f"🟢 COMPRAR {ASSETS[pair_asset1]['label']} / VENDER {ASSETS[pair_asset2]['label']}")
        st.caption(f"El spread está {abs(current_zscore):.2f} desviaciones estándar por debajo de la media")
    else:
        st.info("⚪ Sin señal - Spread dentro del rango normal")
        st.caption(f"Z-Score actual: {current_zscore:.2f}")
    
    # Descarga de datos del spread
    st.markdown("### 📥 Descargar Datos")
    spread_df = pd.DataFrame({
        'date': spread.index,
        'spread': spread.values,
        'zscore': zscore.values
    })
    csv_spread = spread_df.to_csv(index=False)
    st.download_button(
        label="Descargar análisis de spread como CSV",
        data=csv_spread,
        file_name=f"spread_{pair_asset1}_{pair_asset2}.csv",
        mime="text/csv"
    )

with tab6:
    st.subheader("📉 Análisis Técnico")
    
    tech_asset = st.selectbox(
        "Selecciona activo para análisis técnico",
        options=selected_assets,
        format_func=lambda x: ASSETS[x]['label'],
        key='tech_asset'
    )
    
    tech_prices = df_prices[tech_asset]
    tech_returns = calculate_returns(tech_prices)
    
    # Indicadores técnicos
    st.markdown("### 📊 Indicadores Técnicos")
    st.plotly_chart(plot_technical_indicators(tech_prices), use_container_width=True)
    
    # Métricas actuales de indicadores
    st.markdown("### 🎯 Valores Actuales")
    
    rsi_current = calculate_rsi(tech_prices).iloc[-1]
    bb = calculate_bollinger_bands(tech_prices)
    macd_data = calculate_macd(tech_prices)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("RSI (14)", f"{rsi_current:.2f}")
        if rsi_current > 70:
            st.caption("🔴 Sobrecomprado")
        elif rsi_current < 30:
            st.caption("🟢 Sobrevendido")
        else:
            st.caption("⚪ Neutral")
    
    with col2:
        bb_position = ((tech_prices.iloc[-1] - bb['lower'].iloc[-1]) / 
                      (bb['upper'].iloc[-1] - bb['lower'].iloc[-1])) * 100
        st.metric("Posición en BB", f"{bb_position:.1f}%")
        if bb_position > 80:
            st.caption("🔴 Cerca banda superior")
        elif bb_position < 20:
            st.caption("🟢 Cerca banda inferior")
        else:
            st.caption("⚪ Zona media")
    
    with col3:
        macd_signal = "🟢 Alcista" if macd_data['histogram'].iloc[-1] > 0 else "🔴 Bajista"
        st.metric("Señal MACD", macd_signal)
        st.caption(f"Histograma: {macd_data['histogram'].iloc[-1]:.4f}")
    
    # Análisis de momentum
    st.markdown("### 🚀 Análisis de Momentum")
    
    returns_1d = tech_returns.iloc[-1] * 100
    returns_5d = tech_returns.iloc[-5:].sum() * 100
    returns_20d = tech_returns.iloc[-20:].sum() * 100
    returns_60d = tech_returns.iloc[-60:].sum() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Retorno 1D", f"{returns_1d:.2f}%", 
                delta=f"{returns_1d:.2f}%")
    col2.metric("Retorno 5D", f"{returns_5d:.2f}%",
                delta=f"{returns_5d:.2f}%")
    col3.metric("Retorno 20D", f"{returns_20d:.2f}%",
                delta=f"{returns_20d:.2f}%")
    col4.metric("Retorno 60D", f"{returns_60d:.2f}%",
                delta=f"{returns_60d:.2f}%")
    
    # Señales combinadas
    st.markdown("### 🎯 Señales Combinadas")
    
    signals = []
    
    if rsi_current > 70:
        signals.append("⚠️ RSI sobrecomprado (señal de venta)")
    elif rsi_current < 30:
        signals.append("✅ RSI sobrevendido (señal de compra)")
    
    if bb_position > 90:
        signals.append("⚠️ Precio cerca de banda superior (posible retroceso)")
    elif bb_position < 10:
        signals.append("✅ Precio cerca de banda inferior (posible rebote)")
    
    if macd_data['histogram'].iloc[-1] > 0 and macd_data['histogram'].iloc[-2] < 0:
        signals.append("✅ MACD cruzó al alza (señal alcista)")
    elif macd_data['histogram'].iloc[-1] < 0 and macd_data['histogram'].iloc[-2] > 0:
        signals.append("⚠️ MACD cruzó a la baja (señal bajista)")
    
    if not signals:
        st.info("⚪ No hay señales claras en este momento")
    else:
        for signal in signals:
            st.markdown(f"- {signal}")
    
    # Niveles de soporte y resistencia
    st.markdown("### 📍 Niveles Clave (últimos 60 días)")
    
    recent_prices = tech_prices.iloc[-60:]
    resistance = recent_prices.max()
    support = recent_prices.min()
    current_price = tech_prices.iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Soporte", f"${support:.2f}")
    col2.metric("Precio Actual", f"${current_price:.2f}")
    col3.metric("Resistencia", f"${resistance:.2f}")
    
    distance_to_support = ((current_price - support) / current_price) * 100
    distance_to_resistance = ((resistance - current_price) / current_price) * 100
    
    st.caption(f"Distancia al soporte: {distance_to_support:.1f}% | Distancia a resistencia: {distance_to_resistance:.1f}%")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Interpretación")
st.sidebar.markdown("""
**Correlaciones:**
- **> 0.5**: Fuerte correlación positiva
- **< -0.5**: Fuerte correlación negativa
- **≈ 0**: Sin correlación

**Ratios de Performance:**
- **Sharpe > 1**: Bueno, **> 2**: Excelente
- **Sortino > 2**: Muy bueno
- **Omega > 1**: Más probabilidad de ganancias

**Hurst Exponent:**
- **< 0.5**: Mean reverting (bueno para pairs trading)
- **= 0.5**: Random walk
- **> 0.5**: Trending (bueno para momentum)

**RSI:**
- **> 70**: Sobrecomprado
- **< 30**: Sobrevendido

**Z-Score (Pairs Trading):**
- **> 2 o < -2**: Señal de trading
""")
st.sidebar.markdown("---")
st.sidebar.info("💡 Los datos se actualizan automáticamente cada hora")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Funciones Incluidas")
st.sidebar.markdown("""
✅ Correlaciones Rolling  
✅ Volatilidad (Histórica, EWMA)  
✅ Sharpe, Sortino, Calmar Ratios  
✅ VaR y CVaR  
✅ Beta y Alpha  
✅ Cointegración  
✅ Pairs Trading Signals  
✅ Hurst Exponent  
✅ RSI, MACD, Bollinger Bands  
✅ Análisis de Distribución  
✅ Mean Reversion Tests  
""")
