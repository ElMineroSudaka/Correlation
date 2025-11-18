import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Pairs Trading Analyzer",
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
    # Estados Unidos
    'sp500': {'label': 'S&P 500', 'symbol': '^GSPC', 'color': '#3b82f6', 'category': 'US Equity'},
    'nasdaq': {'label': 'NASDAQ', 'symbol': '^IXIC', 'color': '#8b5cf6', 'category': 'US Equity'},
    'dow': {'label': 'Dow Jones', 'symbol': '^DJI', 'color': '#10b981', 'category': 'US Equity'},
    'russell': {'label': 'Russell 2000', 'symbol': '^RUT', 'color': '#06b6d4', 'category': 'US Equity'},
    
    # Europa
    'ftse': {'label': 'FTSE 100', 'symbol': '^FTSE', 'color': '#f97316', 'category': 'Europe Equity'},
    'dax': {'label': 'DAX', 'symbol': '^GDAXI', 'color': '#eab308', 'category': 'Europe Equity'},
    'cac40': {'label': 'CAC 40', 'symbol': '^FCHI', 'color': '#84cc16', 'category': 'Europe Equity'},
    'stoxx50': {'label': 'Euro Stoxx 50', 'symbol': '^STOXX50E', 'color': '#06b6d4', 'category': 'Europe Equity'},
    
    # Asia
    'nikkei': {'label': 'Nikkei 225', 'symbol': '^N225', 'color': '#ec4899', 'category': 'Asia Equity'},
    'hang_seng': {'label': 'Hang Seng', 'symbol': '^HSI', 'color': '#d946ef', 'category': 'Asia Equity'},
    'shanghai': {'label': 'Shanghai', 'symbol': '000001.SS', 'color': '#c026d3', 'category': 'Asia Equity'},
    
    # ETFs Sectoriales
    'qqq': {'label': 'QQQ', 'symbol': 'QQQ', 'color': '#8b5cf6', 'category': 'US ETF'},
    'spy': {'label': 'SPY', 'symbol': 'SPY', 'color': '#3b82f6', 'category': 'US ETF'},
    'xlk': {'label': 'XLK Tech', 'symbol': 'XLK', 'color': '#8b5cf6', 'category': 'Sector ETF'},
    'xlf': {'label': 'XLF Finance', 'symbol': 'XLF', 'color': '#10b981', 'category': 'Sector ETF'},
    'xle': {'label': 'XLE Energy', 'symbol': 'XLE', 'color': '#000000', 'category': 'Sector ETF'},
    'xlv': {'label': 'XLV Health', 'symbol': 'XLV', 'color': '#dc2626', 'category': 'Sector ETF'},
    
    # Divisas
    'dxy': {'label': 'DXY', 'symbol': 'DX-Y.NYB', 'color': '#f59e0b', 'category': 'FX'},
    'eurusd': {'label': 'EUR/USD', 'symbol': 'EURUSD=X', 'color': '#3b82f6', 'category': 'FX'},
    'gbpusd': {'label': 'GBP/USD', 'symbol': 'GBPUSD=X', 'color': '#10b981', 'category': 'FX'},
    'usdjpy': {'label': 'USD/JPY', 'symbol': 'JPYUSD=X', 'color': '#ef4444', 'category': 'FX'},
    
    # Metales
    'gold': {'label': 'Gold', 'symbol': 'GC=F', 'color': '#fbbf24', 'category': 'Metals'},
    'silver': {'label': 'Silver', 'symbol': 'SI=F', 'color': '#d1d5db', 'category': 'Metals'},
    'gld': {'label': 'GLD ETF', 'symbol': 'GLD', 'color': '#fbbf24', 'category': 'Metals'},
    'slv': {'label': 'SLV ETF', 'symbol': 'SLV', 'color': '#d1d5db', 'category': 'Metals'},
    
    # Energía
    'oil': {'label': 'WTI Oil', 'symbol': 'CL=F', 'color': '#000000', 'category': 'Energy'},
    'uso': {'label': 'USO ETF', 'symbol': 'USO', 'color': '#000000', 'category': 'Energy'},
    
    # Bonos
    'us10y': {'label': 'US 10Y', 'symbol': '^TNX', 'color': '#ef4444', 'category': 'Bonds'},
    'tlt': {'label': 'TLT', 'symbol': 'TLT', 'color': '#b91c1c', 'category': 'Bonds'},
    
    # Volatilidad
    'vix': {'label': 'VIX', 'symbol': '^VIX', 'color': '#ec4899', 'category': 'Volatility'},
    
    # Crypto
    'btc': {'label': 'Bitcoin', 'symbol': 'BTC-USD', 'color': '#f7931a', 'category': 'Crypto'},
    'eth': {'label': 'Ethereum', 'symbol': 'ETH-USD', 'color': '#627eea', 'category': 'Crypto'},
}

@st.cache_data(ttl=3600)
def fetch_asset_data(symbol, start_date='2000-01-01', end_date=None):
    """Descarga datos históricos"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        return prices.dropna()
    except Exception as e:
        st.error(f"Error descargando {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def download_selected_assets(selected_keys, delay=10):
    """Descarga activos seleccionados"""
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
    """Combina datos en DataFrame"""
    if not data_dict:
        return pd.DataFrame()
    
    dfs = []
    for key, data in data_dict.items():
        if isinstance(data, pd.Series):
            df_temp = data.to_frame(name=key)
        else:
            df_temp = data.copy()
            df_temp.columns = [key]
        dfs.append(df_temp)
    
    if not dfs:
        return pd.DataFrame()
    
    df = dfs[0]
    for df_temp in dfs[1:]:
        df = df.join(df_temp, how='inner')
    
    return df

def calculate_returns(prices):
    """Calcula retornos logarítmicos"""
    return np.log(prices / prices.shift(1)).dropna()

def calculate_rolling_correlation(df, asset1, asset2, window=30):
    """Calcula correlación móvil"""
    returns1 = calculate_returns(df[asset1])
    returns2 = calculate_returns(df[asset2])
    rolling_corr = returns1.rolling(window).corr(returns2)
    return rolling_corr

def test_cointegration(prices1, prices2):
    """Test de cointegración"""
    try:
        score, pvalue, _ = coint(prices1, prices2)
        return {'score': score, 'pvalue': pvalue, 'cointegrated': pvalue < 0.05}
    except:
        return {'score': np.nan, 'pvalue': np.nan, 'cointegrated': False}

def calculate_spread(prices1, prices2):
    """Calcula spread para pairs trading"""
    prices1_clean = prices1.dropna()
    prices2_clean = prices2.dropna()
    
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

def calculate_hurst_exponent(series, max_lag=100):
    """Calcula Hurst Exponent"""
    lags = range(2, min(max_lag, len(series)//2))
    tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
    
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

def find_best_pairs_positive(df, min_correlation=0.7, min_cointegration_pvalue=0.05):
    """Encuentra mejores pares con correlación POSITIVA"""
    assets = df.columns
    candidates = []
    
    for i, asset1 in enumerate(assets):
        for asset2 in assets[i+1:]:
            try:
                prices1 = df[asset1].dropna()
                prices2 = df[asset2].dropna()
                
                common_idx = prices1.index.intersection(prices2.index)
                if len(common_idx) < 252:
                    continue
                
                p1 = prices1.loc[common_idx]
                p2 = prices2.loc[common_idx]
                
                # Correlación
                correlation = p1.corr(p2)
                if correlation < min_correlation:
                    continue
                
                # Cointegración
                coint_result = test_cointegration(p1, p2)
                if not coint_result['cointegrated']:
                    continue
                
                # Spread y Hurst
                spread, _ = calculate_spread(p1, p2)
                hurst = calculate_hurst_exponent(spread.dropna())
                
                # Score
                score = 0
                if coint_result['pvalue'] < min_cointegration_pvalue:
                    score += 40
                if correlation > 0.8:
                    score += 30
                elif correlation > 0.7:
                    score += 20
                if hurst < 0.5:
                    score += 30
                
                candidates.append({
                    'asset1': asset1,
                    'asset2': asset2,
                    'score': score,
                    'correlation': correlation,
                    'cointegration_pvalue': coint_result['pvalue'],
                    'hurst': hurst
                })
            except:
                continue
    
    # FIX: Verificar si hay candidatos antes de crear DataFrame
    if len(candidates) == 0:
        return pd.DataFrame(columns=['asset1', 'asset2', 'score', 'correlation', 
                                    'cointegration_pvalue', 'hurst'])
    
    return pd.DataFrame(candidates).sort_values('score', ascending=False)

def find_best_pairs_inverse(df, min_negative_correlation=-0.7, max_correlation=-0.3):
    """Encuentra mejores pares con correlación INVERSA"""
    assets = df.columns
    candidates = []
    
    for i, asset1 in enumerate(assets):
        for asset2 in assets[i+1:]:
            try:
                prices1 = df[asset1].dropna()
                prices2 = df[asset2].dropna()
                
                common_idx = prices1.index.intersection(prices2.index)
                if len(common_idx) < 252:
                    continue
                
                p1 = prices1.loc[common_idx]
                p2 = prices2.loc[common_idx]
                
                # Correlación
                correlation = p1.corr(p2)
                if correlation > max_correlation or correlation < min_negative_correlation:
                    continue
                
                # Volatilidad
                returns1 = calculate_returns(p1)
                returns2 = calculate_returns(p2)
                vol1 = returns1.std() * np.sqrt(252)
                vol2 = returns2.std() * np.sqrt(252)
                vol_ratio = min(vol1, vol2) / max(vol1, vol2)
                
                # Estabilidad
                rolling_corr = returns1.rolling(60).corr(returns2)
                corr_std = rolling_corr.std()
                
                # Score
                score = 0
                if correlation < -0.7:
                    score += 40
                elif correlation < -0.5:
                    score += 25
                
                if corr_std < 0.1:
                    score += 30
                elif corr_std < 0.2:
                    score += 20
                
                if vol_ratio > 0.7:
                    score += 20
                
                candidates.append({
                    'asset1': asset1,
                    'asset2': asset2,
                    'score': score,
                    'correlation': correlation,
                    'corr_stability': corr_std,
                    'vol_ratio': vol_ratio
                })
            except:
                continue
    
    # FIX: Verificar si hay candidatos
    if len(candidates) == 0:
        return pd.DataFrame(columns=['asset1', 'asset2', 'score', 'correlation', 
                                    'corr_stability', 'vol_ratio'])
    
    return pd.DataFrame(candidates).sort_values('score', ascending=False)

def plot_rolling_correlation(corr_series, asset1_name, asset2_name):
    """Gráfico de correlación rolling"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=corr_series.index,
        y=corr_series,
        mode='lines',
        name='Correlation',
        line=dict(color='#3b82f6', width=3)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#666666")
    fig.add_hline(y=0.5, line_dash="dot", line_color="#10b981", opacity=0.5)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="#ef4444", opacity=0.5)
    
    fig.update_layout(
        title=f'Rolling Correlation: {asset1_name} vs {asset2_name}',
        xaxis_title='Date',
        yaxis_title='Correlation',
        yaxis=dict(range=[-1, 1]),
        template='plotly_dark',
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_correlation_heatmap(df, selected_assets):
    """Heatmap de correlaciones"""
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
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Matrix',
        template='plotly_dark',
        height=600
    )
    
    return fig

def plot_price_comparison(df, asset1, asset2, asset1_name, asset2_name):
    """Gráfico comparativo de precios"""
    fig = go.Figure()
    
    norm1 = (df[asset1] / df[asset1].iloc[0]) * 100
    norm2 = (df[asset2] / df[asset2].iloc[0]) * 100
    
    fig.add_trace(go.Scatter(
        x=df.index, y=norm1, name=asset1_name,
        line=dict(color=ASSETS[asset1]['color'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=norm2, name=asset2_name,
        line=dict(color=ASSETS[asset2]['color'], width=2)
    ))
    
    fig.update_layout(
        title='Price Comparison (Base 100)',
        template='plotly_dark',
        hovermode='x unified',
        height=400
    )
    
    return fig

def plot_spread_analysis(prices1, prices2, asset1_name, asset2_name):
    """Análisis de spread"""
    spread, hedge_ratio = calculate_spread(prices1, prices2)
    zscore = calculate_zscore(spread, window=30)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Spread: {asset1_name} - {hedge_ratio:.4f} * {asset2_name}',
            'Z-Score'
        ),
        vertical_spacing=0.15
    )
    
    fig.add_trace(go.Scatter(
        x=spread.index, y=spread, name='Spread',
        line=dict(color='#3b82f6', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=zscore.index, y=zscore, name='Z-Score',
        line=dict(color='#8b5cf6', width=2)
    ), row=2, col=1)
    
    fig.add_hline(y=2, line_dash="dash", line_color="#ef4444", row=2, col=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="#10b981", row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#666666", row=2, col=1)
    
    fig.update_layout(height=700, template='plotly_dark')
    
    return fig

def plot_pairs_ranking(pairs_df, top_n=15, title="Best Pairs"):
    """Ranking de pares"""
    if len(pairs_df) == 0:
        return None
    
    top_pairs = pairs_df.head(top_n).copy()
    top_pairs['pair_label'] = top_pairs['asset1'].apply(lambda x: ASSETS[x]['label']) + ' / ' + \
                               top_pairs['asset2'].apply(lambda x: ASSETS[x]['label'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_pairs['pair_label'],
        x=top_pairs['score'],
        orientation='h',
        marker=dict(
            color=top_pairs['score'],
            colorscale='Viridis',
            showscale=True
        ),
        text=top_pairs['score'].round(1),
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Score',
        yaxis_title='Pair',
        template='plotly_dark',
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================

st.title("📊 Pairs Trading & Correlation Analyzer")
st.markdown("🔍 Búsqueda de Pares Correlacionados e Inversamente Correlacionados")

# Sidebar
st.sidebar.header("⚙️ Configuración")

# Categorías
categories = list(set([ASSETS[k]['category'] for k in ASSETS.keys()]))
categories.sort()
selected_categories = st.sidebar.multiselect(
    "Categorías",
    options=categories,
    default=['US Equity', 'FX', 'Metals', 'Crypto']
)

available_assets = [k for k in ASSETS.keys() if ASSETS[k]['category'] in selected_categories]

# Selección de activos
default_assets = ['sp500', 'nasdaq', 'gold', 'btc', 'dxy', 'vix']
selected_assets = st.sidebar.multiselect(
    "Activos (mín. 2)",
    options=available_assets,
    default=[a for a in default_assets if a in available_assets],
    format_func=lambda x: f"{ASSETS[x]['label']} ({ASSETS[x]['category']})"
)

if len(selected_assets) < 2:
    st.warning("⚠️ Selecciona al menos 2 activos")
    st.stop()

st.sidebar.info(f"✅ {len(selected_assets)} activos seleccionados")

# Parámetros
window_size = st.sidebar.slider("Ventana correlación (días)", 10, 90, 30, 5)
download_delay = st.sidebar.slider("Delay descargas (seg)", 1, 30, 10, 1)

# Período
date_range = st.sidebar.selectbox(
    "Período",
    ['Todo', '5 años', '3 años', '2 años', '1 año']
)

# Descarga de datos
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if st.sidebar.button("📥 Descargar Datos", type="primary", disabled=st.session_state.data_loaded):
    with st.spinner("Descargando datos..."):
        asset_data = download_selected_assets(selected_assets, delay=download_delay)
    
    if not asset_data:
        st.error("Error descargando datos")
        st.stop()
    
    df_prices = merge_asset_data(asset_data)
    
    if df_prices.empty:
        st.error("No hay datos suficientes")
        st.stop()
    
    if date_range != 'Todo':
        days_map = {'1 año': 252, '2 años': 504, '3 años': 756, '5 años': 1260}
        days = days_map[date_range]
        df_prices = df_prices.iloc[-days:]
    
    st.session_state.df_prices = df_prices
    st.session_state.data_loaded = True
    st.success(f"✅ Datos cargados: {len(df_prices)} días")
    st.rerun()

if st.sidebar.button("🔄 Limpiar"):
    st.session_state.data_loaded = False
    if 'df_prices' in st.session_state:
        del st.session_state.df_prices
    st.cache_data.clear()
    st.rerun()

if not st.session_state.data_loaded:
    st.info("""
    ### 👋 Bienvenido!
    
    **Para comenzar:**
    1. 📂 Selecciona categorías y activos
    2. ⚙️ Configura parámetros
    3. 📥 Presiona 'Descargar Datos'
    
    **Funcionalidades:**
    - 🔍 Búsqueda automática de pares correlacionados
    - 🛡️ Búsqueda de pares inversamente correlacionados (hedging)
    - 📊 Análisis de correlación detallado
    - 🎯 Pairs trading (spread, z-score, cointegración)
    """)
    st.stop()

df_prices = st.session_state.df_prices
st.success(f"✅ {len(df_prices)} días | {df_prices.index[0].date()} → {df_prices.index[-1].date()}")

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Análisis de Pares",
    "🔥 Matriz de Correlación",
    "🔍 Búsqueda Correlación Positiva",
    "🛡️ Búsqueda Correlación Inversa"
])

with tab1:
    st.subheader("📈 Análisis Detallado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        asset1 = st.selectbox(
            "Activo 1",
            options=selected_assets,
            format_func=lambda x: ASSETS[x]['label']
        )
    
    with col2:
        asset2 = st.selectbox(
            "Activo 2",
            options=[a for a in selected_assets if a != asset1],
            format_func=lambda x: ASSETS[x]['label']
        )
    
    # Correlación
    corr_series = calculate_rolling_correlation(df_prices, asset1, asset2, window_size)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Correlación Actual", f"{corr_series.iloc[-1]:.4f}")
    col2.metric("Correlación Media", f"{corr_series.mean():.4f}")
    col3.metric("Máxima", f"{corr_series.max():.4f}")
    
    st.plotly_chart(
        plot_rolling_correlation(corr_series, ASSETS[asset1]['label'], ASSETS[asset2]['label']),
        use_container_width=True
    )
    
    # Precios
    st.plotly_chart(
        plot_price_comparison(df_prices, asset1, asset2, ASSETS[asset1]['label'], ASSETS[asset2]['label']),
        use_container_width=True
    )
    
    # Tests
    st.subheader("📊 Tests Estadísticos")
    
    prices1 = df_prices[asset1]
    prices2 = df_prices[asset2]
    
    coint_test = test_cointegration(prices1, prices2)
    spread, hedge_ratio = calculate_spread(prices1, prices2)
    adf_result = adf_test(spread)
    hurst = calculate_hurst_exponent(spread.dropna())
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        "Cointegración",
        "✅ SÍ" if coint_test['cointegrated'] else "❌ NO",
        delta=f"p-value: {coint_test['pvalue']:.4f}"
    )
    
    col2.metric(
        "Estacionariedad",
        "✅ SÍ" if adf_result['stationary'] else "❌ NO",
        delta=f"p-value: {adf_result['pvalue']:.4f}"
    )
    
    col3.metric(
        "Hurst Exponent",
        f"{hurst:.3f}",
        delta="Mean Reverting" if hurst < 0.5 else "Trending"
    )
    
    # Spread
    if coint_test['cointegrated']:
        st.subheader("📊 Análisis de Spread")
        st.plotly_chart(
            plot_spread_analysis(prices1, prices2, ASSETS[asset1]['label'], ASSETS[asset2]['label']),
            use_container_width=True
        )

with tab2:
    st.subheader("🔥 Matriz de Correlaciones")
    st.plotly_chart(
        plot_correlation_heatmap(df_prices, selected_assets),
        use_container_width=True
    )
    
    # Tabla
    st.subheader("📋 Tabla de Correlaciones")
    corr_matrix = df_prices[selected_assets].corr()
    
    def color_correlation(val):
        if val >= 0.7:
            return 'background-color: #10b981; color: white'
        elif val >= 0.3:
            return 'background-color: #84cc16; color: white'
        elif val > -0.3:
            return 'background-color: #6b7280; color: white'
        elif val > -0.7:
            return 'background-color: #f59e0b; color: white'
        else:
            return 'background-color: #ef4444; color: white'
    
    styled_df = corr_matrix.style.applymap(color_correlation).format("{:.2f}")
    st.dataframe(styled_df, use_container_width=True)

with tab3:
    st.subheader("🔍 Búsqueda de Pares - Correlación Positiva")
    st.caption("Ideal para pairs trading y mean reversion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_corr = st.slider("Correlación mínima", 0.5, 0.95, 0.7, 0.05)
    
    with col2:
        max_pvalue = st.slider("P-value máx (coint.)", 0.01, 0.10, 0.05, 0.01)
    
    if st.button("🔎 Buscar Pares Positivos", type="primary"):
        with st.spinner("Analizando pares..."):
            best_pairs = find_best_pairs_positive(
                df_prices[selected_assets],
                min_correlation=min_corr,
                min_cointegration_pvalue=max_pvalue
            )
        
        if len(best_pairs) > 0:
            st.success(f"✅ Encontrados {len(best_pairs)} pares")
            
            # Gráfico
            fig = plot_pairs_ranking(best_pairs, top_n=15, title="Top Pares - Correlación Positiva")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Tabla
            st.markdown("### 📋 Resultados Detallados")
            
            display_df = best_pairs.head(20).copy()
            display_df['Activo 1'] = display_df['asset1'].apply(lambda x: ASSETS[x]['label'])
            display_df['Activo 2'] = display_df['asset2'].apply(lambda x: ASSETS[x]['label'])
            
            display_df = display_df[['Activo 1', 'Activo 2', 'score', 'correlation', 
                                     'cointegration_pvalue', 'hurst']]
            
            st.dataframe(
                display_df.style.format({
                    'score': '{:.1f}',
                    'correlation': '{:.3f}',
                    'cointegration_pvalue': '{:.4f}',
                    'hurst': '{:.3f}'
                }),
                use_container_width=True
            )
            
            # Mejor par
            if len(best_pairs) > 0:
                st.markdown("### 🏆 Mejor Par")
                best = best_pairs.iloc[0]
                
                st.info(f"**{ASSETS[best['asset1']]['label']} / {ASSETS[best['asset2']]['label']}** | Score: {best['score']:.1f}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Correlación", f"{best['correlation']:.3f}")
                col2.metric("P-value Coint.", f"{best['cointegration_pvalue']:.4f}")
                col3.metric("Hurst", f"{best['hurst']:.3f}")
            
            # Descarga
            csv = best_pairs.to_csv(index=False)
            st.download_button(
                "📥 Descargar CSV",
                csv,
                "pares_positivos.csv",
                "text/csv"
            )
        else:
            st.warning("⚠️ No se encontraron pares. Relaja los parámetros.")

with tab4:
    st.subheader("🛡️ Búsqueda de Pares - Correlación Inversa")
    st.caption("Ideal para hedging y diversificación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_neg_corr = st.slider("Corr. mín. (negativa)", -0.95, -0.3, -0.7, 0.05)
    
    with col2:
        max_neg_corr = st.slider("Corr. máx. (negativa)", -0.95, -0.3, -0.3, 0.05)
    
    if st.button("🔎 Buscar Pares Inversos", type="primary"):
        with st.spinner("Analizando pares inversos..."):
            inverse_pairs = find_best_pairs_inverse(
                df_prices[selected_assets],
                min_negative_correlation=min_neg_corr,
                max_correlation=max_neg_corr
            )
        
        if len(inverse_pairs) > 0:
            st.success(f"✅ Encontrados {len(inverse_pairs)} pares inversos")
            
            # Gráfico
            fig = plot_pairs_ranking(inverse_pairs, top_n=15, title="Top Pares - Correlación Inversa")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Tabla
            st.markdown("### 📋 Resultados Detallados")
            
            display_df = inverse_pairs.head(20).copy()
            display_df['Activo 1'] = display_df['asset1'].apply(lambda x: ASSETS[x]['label'])
            display_df['Activo 2'] = display_df['asset2'].apply(lambda x: ASSETS[x]['label'])
            
            display_df = display_df[['Activo 1', 'Activo 2', 'score', 'correlation', 
                                     'corr_stability', 'vol_ratio']]
            
            st.dataframe(
                display_df.style.format({
                    'score': '{:.1f}',
                    'correlation': '{:.3f}',
                    'corr_stability': '{:.3f}',
                    'vol_ratio': '{:.2f}'
                }),
                use_container_width=True
            )
            
            # Mejor par
            if len(inverse_pairs) > 0:
                st.markdown("### 🏆 Mejor Par Inverso")
                best = inverse_pairs.iloc[0]
                
                st.info(f"**{ASSETS[best['asset1']]['label']} / {ASSETS[best['asset2']]['label']}** | Score: {best['score']:.1f}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Correlación", f"{best['correlation']:.3f}")
                col2.metric("Estabilidad", f"{best['corr_stability']:.3f}")
                col3.metric("Vol Ratio", f"{best['vol_ratio']:.2f}")
                
                st.success(f"✅ Ideal para hedging: correlación negativa estable")
            
            # Descarga
            csv = inverse_pairs.to_csv(index=False)
            st.download_button(
                "📥 Descargar CSV",
                csv,
                "pares_inversos.csv",
                "text/csv"
            )
        else:
            st.warning("⚠️ No se encontraron pares inversos. Ajusta el rango.")

# Sidebar - Info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Guía")
st.sidebar.info("""
**Correlación Positiva (Tab 3):**
- > 0.7: Pairs trading
- Cointegración importante
- Hurst < 0.5: Mean reversion

**Correlación Inversa (Tab 4):**
- < -0.5: Hedging
- Reduce volatilidad
- Protección en caídas

**Z-Score Trading:**
- > 2: Vender spread
- < -2: Comprar spread
""")

st.sidebar.markdown("---")
st.sidebar.success("✨ Simplificado para búsqueda de pares")
