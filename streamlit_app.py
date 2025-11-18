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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Pro Correlation & Pairs Trader",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3346;
    }
    h1, h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONFIGURACIÓN DE ACTIVOS (CURADA - SIN DUPLICADOS)
# =============================================================================
ASSETS = {
    # --- ÍNDICES PRINCIPALES (Benchmark) ---
    'sp500': {'label': 'S&P 500', 'symbol': '^GSPC', 'color': '#3b82f6', 'category': 'Indices'},
    'nasdaq': {'label': 'NASDAQ Composite', 'symbol': '^IXIC', 'color': '#8b5cf6', 'category': 'Indices'},
    'dow': {'label': 'Dow Jones Ind.', 'symbol': '^DJI', 'color': '#10b981', 'category': 'Indices'},
    'russell': {'label': 'Russell 2000', 'symbol': '^RUT', 'color': '#06b6d4', 'category': 'Indices'},
    'vix': {'label': 'VIX (Volatilidad)', 'symbol': '^VIX', 'color': '#ec4899', 'category': 'Indices'},
    
    # --- EUROPA & ASIA ---
    'dax': {'label': 'DAX (Alemania)', 'symbol': '^GDAXI', 'color': '#eab308', 'category': 'Global'},
    'ftse': {'label': 'FTSE 100 (UK)', 'symbol': '^FTSE', 'color': '#f97316', 'category': 'Global'},
    'nikkei': {'label': 'Nikkei 225 (Japón)', 'symbol': '^N225', 'color': '#ec4899', 'category': 'Global'},
    'hang_seng': {'label': 'Hang Seng (HK)', 'symbol': '^HSI', 'color': '#d946ef', 'category': 'Global'},
    'shanghai': {'label': 'Shanghai Comp', 'symbol': '000001.SS', 'color': '#c026d3', 'category': 'Global'},
    'bovespa': {'label': 'Bovespa (Brasil)', 'symbol': '^BVSP', 'color': '#84cc16', 'category': 'Global'},

    # --- SECTORES (ETFs Selectos) ---
    'qqq': {'label': 'QQQ (Nasdaq 100)', 'symbol': 'QQQ', 'color': '#8b5cf6', 'category': 'US ETF'},
    'xlk': {'label': 'XLK (Tecnología)', 'symbol': 'XLK', 'color': '#3b82f6', 'category': 'Sectores'},
    'xlf': {'label': 'XLF (Financiero)', 'symbol': 'XLF', 'color': '#10b981', 'category': 'Sectores'},
    'xle': {'label': 'XLE (Energía)', 'symbol': 'XLE', 'color': '#000000', 'category': 'Sectores'},
    'xlv': {'label': 'XLV (Salud)', 'symbol': 'XLV', 'color': '#dc2626', 'category': 'Sectores'},
    'xly': {'label': 'XLY (Consumo Disc.)', 'symbol': 'XLY', 'color': '#ec4899', 'category': 'Sectores'},
    'xlp': {'label': 'XLP (Consumo Básico)', 'symbol': 'XLP', 'color': '#22c55e', 'category': 'Sectores'},
    'xlu': {'label': 'XLU (Utilities)', 'symbol': 'XLU', 'color': '#eab308', 'category': 'Sectores'},
    'xlre': {'label': 'XLRE (Real Estate)', 'symbol': 'XLRE', 'color': '#f59e0b', 'category': 'Sectores'},
    'arkk': {'label': 'ARKK (Innovación)', 'symbol': 'ARKK', 'color': '#a855f7', 'category': 'Sectores'},
    'smh': {'label': 'SMH (Semiconductores)', 'symbol': 'SMH', 'color': '#f97316', 'category': 'Sectores'},

    # --- MATERIAS PRIMAS (Futuros Principales) ---
    'gold': {'label': 'Oro (Futuros)', 'symbol': 'GC=F', 'color': '#fbbf24', 'category': 'Commodities'},
    'silver': {'label': 'Plata (Futuros)', 'symbol': 'SI=F', 'color': '#d1d5db', 'category': 'Commodities'},
    'copper': {'label': 'Cobre', 'symbol': 'HG=F', 'color': '#c2410c', 'category': 'Commodities'},
    'oil': {'label': 'Petróleo WTI', 'symbol': 'CL=F', 'color': '#000000', 'category': 'Commodities'},
    'brent': {'label': 'Petróleo Brent', 'symbol': 'BZ=F', 'color': '#171717', 'category': 'Commodities'},
    'natgas': {'label': 'Gas Natural', 'symbol': 'NG=F', 'color': '#059669', 'category': 'Commodities'},
    'corn': {'label': 'Maíz', 'symbol': 'ZC=F', 'color': '#fbbf24', 'category': 'Commodities'},
    'soybeans': {'label': 'Soja', 'symbol': 'ZS=F', 'color': '#84cc16', 'category': 'Commodities'},

    # --- FOREX ---
    'dxy': {'label': 'DXY (Dólar Index)', 'symbol': 'DX-Y.NYB', 'color': '#f59e0b', 'category': 'FX'},
    'eurusd': {'label': 'EUR/USD', 'symbol': 'EURUSD=X', 'color': '#3b82f6', 'category': 'FX'},
    'gbpusd': {'label': 'GBP/USD', 'symbol': 'GBPUSD=X', 'color': '#10b981', 'category': 'FX'},
    'usdjpy': {'label': 'USD/JPY', 'symbol': 'JPYUSD=X', 'color': '#ef4444', 'category': 'FX'},
    'audusd': {'label': 'AUD/USD', 'symbol': 'AUDUSD=X', 'color': '#059669', 'category': 'FX'},
    'usdmxn': {'label': 'USD/MXN', 'symbol': 'MXN=X', 'color': '#22c55e', 'category': 'FX'},
    'usdbrl': {'label': 'USD/BRL', 'symbol': 'BRL=X', 'color': '#84cc16', 'category': 'FX'},

    # --- CRYPTO ---
    'btc': {'label': 'Bitcoin', 'symbol': 'BTC-USD', 'color': '#f7931a', 'category': 'Crypto'},
    'eth': {'label': 'Ethereum', 'symbol': 'ETH-USD', 'color': '#627eea', 'category': 'Crypto'},
    'sol': {'label': 'Solana', 'symbol': 'SOL-USD', 'color': '#14f195', 'category': 'Crypto'},
    'bnb': {'label': 'Binance Coin', 'symbol': 'BNB-USD', 'color': '#f3ba2f', 'category': 'Crypto'},

    # --- BONOS ---
    'us10y': {'label': 'US 10Y Yield', 'symbol': '^TNX', 'color': '#ef4444', 'category': 'Bonos'},
    'tlt': {'label': 'TLT (Bonos 20Y+)', 'symbol': 'TLT', 'color': '#b91c1c', 'category': 'Bonos'},
    'lqd': {'label': 'LQD (Inv. Grade)', 'symbol': 'LQD', 'color': '#3b82f6', 'category': 'Bonos'},
    'hyg': {'label': 'HYG (High Yield)', 'symbol': 'HYG', 'color': '#f59e0b', 'category': 'Bonos'},
}

# =============================================================================
# FUNCIONES DATA & CACHE
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_asset_data(symbol, start_date='2010-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if data.empty: return None
        
        # Manejo flexible de columnas de Yahoo
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            return None
            
        if isinstance(prices, pd.DataFrame):
            if symbol in prices.columns:
                prices = prices[symbol]
            else:
                prices = prices.iloc[:, 0]
                
        return prices.dropna()
    except Exception:
        return None

@st.cache_data(ttl=3600)
def download_selected_assets(selected_keys):
    all_data = {}
    progress = st.progress(0)
    status = st.empty()
    total = len(selected_keys)
    
    for i, key in enumerate(selected_keys):
        sym = ASSETS[key]['symbol']
        status.caption(f"Descargando {ASSETS[key]['label']}...")
        data = fetch_asset_data(sym)
        if data is not None:
            all_data[key] = data
        progress.progress((i + 1) / total)
        time.sleep(0.05) # Breve pausa
        
    progress.empty()
    status.empty()
    return all_data

def merge_asset_data(data_dict):
    if not data_dict: return pd.DataFrame()
    series_list = []
    for k, v in data_dict.items():
        s = v.copy()
        s.name = k
        series_list.append(s)
    return pd.concat(series_list, axis=1, join='inner')

# =============================================================================
# FUNCIONES MATEMÁTICAS Y ESTADÍSTICAS
# =============================================================================

def calculate_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def calculate_rolling_correlation(df, asset1, asset2, window=30):
    return df[asset1].rolling(window).corr(df[asset2])

def calculate_spread(prices1, prices2):
    # OLS para Hedge Ratio
    X = sm.add_constant(prices2)
    model = sm.OLS(prices1, X).fit()
    hedge_ratio = model.params.iloc[1]
    spread = prices1 - hedge_ratio * prices2
    return spread, hedge_ratio

def calculate_zscore(series, window=30):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

def calculate_half_life(spread):
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    spread_lag2 = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret, spread_lag2)
    res = model.fit()
    lambda_param = res.params.iloc[1]
    if lambda_param >= 0: return np.inf
    return -np.log(2) / lambda_param

def calculate_hurst(series, max_lag=100):
    lags = range(2, min(max_lag, len(series)//2))
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    try:
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return np.nan

def calculate_conditional_correlation(returns1, returns2, condition='crisis'):
    """Calcula correlación en escenarios específicos"""
    if condition == 'positive':
        mask = (returns1 > 0) & (returns2 > 0)
    elif condition == 'negative':
        mask = (returns1 < 0) & (returns2 < 0)
    elif condition == 'crisis':
        # Alta volatilidad (más de 2 desviaciones estándar)
        vol_threshold = returns1.std() * 2
        mask = (abs(returns1) > vol_threshold) | (abs(returns2) > vol_threshold)
    else:
        mask = pd.Series(True, index=returns1.index)
    
    if mask.sum() < 5: return np.nan
    return returns1[mask].corr(returns2[mask])

def perform_pca_analysis(returns_df, n_components=3):
    """Análisis de Componentes Principales"""
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(returns_df.dropna())
        pca = PCA(n_components=n_components)
        pca.fit(scaled_data)
        
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=returns_df.columns
        )
        explained = pca.explained_variance_ratio_
        return loadings, explained
    except:
        return None, None

def detect_regimes(returns_df, n_regimes=3):
    """Clustering K-Means para detectar regímenes de mercado"""
    try:
        features = pd.DataFrame({
            'volatility': returns_df.rolling(20).std().mean(axis=1),
            'return': returns_df.rolling(20).mean().mean(axis=1)
        }).dropna()
        
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        clusters = kmeans.fit_predict(features)
        return pd.Series(clusters, index=features.index)
    except:
        return None

def run_backtest_strategy(spread, zscore, entry=2.0, exit=0.0):
    """Simulación simple de estrategia de reversión a la media"""
    signals = pd.Series(0, index=spread.index)
    position = 0 # 0:flat, 1:long, -1:short
    
    positions_hist = []
    for z in zscore:
        if position == 0:
            if z > entry: position = -1 # Vender spread caro
            elif z < -entry: position = 1 # Comprar spread barato
        elif position == -1:
            if z < exit: position = 0
        elif position == 1:
            if z > -exit: position = 0
        positions_hist.append(position)
        
    signals = pd.Series(positions_hist, index=spread.index)
    # PnL simplificado: cambio en el spread * posición ayer
    pnl = spread.diff() * signals.shift(1)
    return pnl.cumsum(), signals

# =============================================================================
# INTERFAZ DE USUARIO
# =============================================================================

st.sidebar.title("⚙️ Configuración")
st.sidebar.caption("v5")

# Filtros
all_cats = sorted(list(set([v['category'] for v in ASSETS.values()])))
sel_cats = st.sidebar.multiselect("Categorías", all_cats, default=['Indices', 'Commodities', 'Crypto', 'Sectores'])

avail_assets = [k for k, v in ASSETS.items() if v['category'] in sel_cats]
default_sel = ['sp500', 'gold', 'btc', 'oil', 'vix', 'nasdaq'] 
final_default = [x for x in default_sel if x in avail_assets]

selected_assets = st.sidebar.multiselect(
    "Activos (Selecciona varios)", 
    avail_assets, 
    default=final_default,
    format_func=lambda x: f"{ASSETS[x]['label']}"
)

if len(selected_assets) < 2:
    st.sidebar.error("Selecciona al menos 2 activos.")
    st.stop()

if st.sidebar.button("📥 Descargar Datos", type="primary"):
    data = download_selected_assets(selected_assets)
    if data:
        st.session_state['df'] = merge_asset_data(data)
        st.session_state['data_loaded'] = True
        st.rerun()

if 'df' not in st.session_state:
    st.info("👈 Por favor, selecciona tus activos y descarga los datos para comenzar.")
    st.stop()

df = st.session_state['df']
st.sidebar.success(f"Datos cargados: {len(df)} registros")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Análisis de Correlación",
    "🔥 Heatmap & Clusters",
    "🎯 Pairs Trading & Backtest",
    "🔍 Scanner de Oportunidades",
    "🛡️ Hedging (Inverso)"
])

# TAB 1: ANÁLISIS DETALLADO
with tab1:
    c1, c2 = st.columns(2)
    a1 = c1.selectbox("Activo A", df.columns, format_func=lambda x: ASSETS[x]['label'])
    a2 = c2.selectbox("Activo B", [x for x in df.columns if x != a1], format_func=lambda x: ASSETS[x]['label'])
    
    # Cálculos
    p1 = df[a1]
    p2 = df[a2]
    r1 = calculate_returns(p1)
    r2 = calculate_returns(p2)
    
    # Rolling Corr
    window = st.slider("Ventana (días)", 10, 200, 60)
    roll_corr = r1.rolling(window).corr(r2)
    
    # Gráfico Corr
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr, name='Correlación', line=dict(color='#3b82f6', width=2)))
    fig_corr.add_hline(y=0, line_dash='dash', line_color='gray')
    fig_corr.add_hline(y=0.7, line_dash='dot', line_color='green', opacity=0.5)
    fig_corr.add_hline(y=-0.7, line_dash='dot', line_color='red', opacity=0.5)
    fig_corr.update_layout(title=f"Correlación Histórica ({window} días)", template="plotly_dark", height=400)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Estadísticas Condicionales
    st.markdown("#### 📊 Comportamiento en Distintos Escenarios")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Corr. General", f"{r1.corr(r2):.2f}")
    k2.metric("En Mercado Alcista", f"{calculate_conditional_correlation(r1, r2, 'positive'):.2f}")
    k3.metric("En Mercado Bajista", f"{calculate_conditional_correlation(r1, r2, 'negative'):.2f}")
    k4.metric("En Crisis (Vol Alta)", f"{calculate_conditional_correlation(r1, r2, 'crisis'):.2f}")
    
    # Lead-Lag
    st.markdown("#### 🏁 Análisis de Liderazgo (Lead-Lag)")
    lags = range(-5, 6)
    lag_corrs = [r1.corr(r2.shift(l)) for l in lags]
    max_lag = lags[np.argmax(np.abs(lag_corrs))]
    
    fig_lag = go.Figure()
    fig_lag.add_trace(go.Bar(x=list(lags), y=lag_corrs, marker_color=np.where(np.array(lags)==0, '#f59e0b', '#3b82f6')))
    fig_lag.update_layout(title="Correlación con Desplazamiento (Días)", xaxis_title="Lag (Días)", template="plotly_dark", height=300)
    st.plotly_chart(fig_lag, use_container_width=True)
    
    if max_lag < 0:
        st.info(f"💡 **{ASSETS[a2]['label']}** tiende a moverse {abs(max_lag)} días ANTES que {ASSETS[a1]['label']}.")
    elif max_lag > 0:
        st.info(f"💡 **{ASSETS[a1]['label']}** tiende a moverse {max_lag} días ANTES que {ASSETS[a2]['label']}.")
    else:
        st.info("💡 El movimiento es simultáneo (sin lag claro).")

# TAB 2: HEATMAP & CLUSTERS
with tab2:
    st.subheader("🔥 Mapa de Calor del Mercado")
    corr_matrix = df.pct_change().corr()
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[ASSETS[c]['label'] for c in df.columns],
        y=[ASSETS[c]['label'] for c in df.columns],
        colorscale='RdBu', zmin=-1, zmax=1
    ))
    fig_heat.update_layout(height=700, template="plotly_dark")
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.subheader("🧠 Análisis de Componentes Principales (PCA)")
    st.caption("Identifica los factores ocultos que mueven tu portafolio.")
    loadings, explained = perform_pca_analysis(df.pct_change(), n_components=3)
    
    if loadings is not None:
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Factor 1 (Tendencia Gral)", f"{explained[0]*100:.1f}%")
        pc2.metric("Factor 2 (Rotación)", f"{explained[1]*100:.1f}%")
        pc3.metric("Factor 3 (Específico)", f"{explained[2]*100:.1f}%")
        
        # Mostrar loadings coloreados
        st.dataframe(loadings.style.background_gradient(cmap='RdYlGn'), use_container_width=True)

# TAB 3: PAIRS TRADING & BACKTEST
with tab3:
    st.subheader("🎯 Estrategia de Pairs Trading")
    
    c1, c2 = st.columns(2)
    pt_a1 = c1.selectbox("Activo Largo/Corto", df.columns, key='pt1', format_func=lambda x: ASSETS[x]['label'])
    pt_a2 = c2.selectbox("Activo Hedge", [x for x in df.columns if x != pt_a1], key='pt2', format_func=lambda x: ASSETS[x]['label'])
    
    # Cálculos
    spread, hedge_ratio = calculate_spread(df[pt_a1], df[pt_a2])
    zscore = calculate_zscore(spread)
    half_life = calculate_half_life(spread)
    coint_p = coint(df[pt_a1], df[pt_a2])[1]
    hurst = calculate_hurst(spread.dropna().values)
    
    # Métricas
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hedge Ratio", f"{hedge_ratio:.3f}")
    m2.metric("Cointegración (p-val)", f"{coint_p:.4f}", delta="✅" if coint_p < 0.05 else "❌")
    m3.metric("Half-Life", f"{half_life:.1f} días")
    m4.metric("Hurst Exp", f"{hurst:.2f}", delta="Mean Rev" if hurst < 0.5 else "Trend")
    
    # Gráficos
    fig_spread = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig_spread.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread", line=dict(color='#3b82f6')), row=1, col=1)
    fig_spread.add_hline(y=spread.mean(), line_dash="dash", row=1, col=1)
    
    fig_spread.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-Score", line=dict(color='#eab308')), row=2, col=1)
    fig_spread.add_hline(y=2, line_color="red", line_dash="dot", row=2, col=1)
    fig_spread.add_hline(y=-2, line_color="green", line_dash="dot", row=2, col=1)
    
    fig_spread.update_layout(height=600, template="plotly_dark", title="Monitor de Spread & Z-Score")
    st.plotly_chart(fig_spread, use_container_width=True)
    
    # --- BACKTEST ---
    st.markdown("---")
    st.subheader("💰 Simulación de Estrategia (Backtest)")
    
    b1, b2, b3 = st.columns(3)
    entry_z = b1.number_input("Entrada Z", 1.0, 4.0, 2.0, 0.1)
    exit_z = b2.number_input("Salida Z", 0.0, 2.0, 0.0, 0.1)
    
    if st.button("🚀 Correr Simulación"):
        cum_pnl, signals = run_backtest_strategy(spread, zscore, entry_z, exit_z)
        
        final_ret = cum_pnl.iloc[-1]
        trades = signals.diff().abs().sum() / 2
        
        bk1, bk2 = st.columns(2)
        bk1.metric("PnL Acumulado (Spread Units)", f"{final_ret:.2f}")
        bk2.metric("Total Operaciones", f"{int(trades)}")
        
        fig_bk = go.Figure()
        fig_bk.add_trace(go.Scatter(x=cum_pnl.index, y=cum_pnl, fill='tozeroy', mode='lines', name='Equity Curve'))
        fig_bk.update_layout(template="plotly_dark", height=300, title="Curva de Capital Teórica")
        st.plotly_chart(fig_bk, use_container_width=True)

# TAB 4: SCANNER
with tab4:
    st.subheader("📡 Radar de Oportunidades")
    st.caption("Busca pares cointegrados automáticamente.")
    
    if st.button("Escanear Pares"):
        results = []
        cols = df.columns
        prog = st.progress(0)
        pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i+1:]]
        
        for i, (a, b) in enumerate(pairs):
            try:
                score_coint = coint(df[a], df[b])[1]
                corr = df[a].corr(df[b])
                if score_coint < 0.1 and corr > 0.5:
                    spr, _ = calculate_spread(df[a], df[b])
                    hl = calculate_half_life(spr)
                    h = calculate_hurst(spr.dropna().values)
                    results.append({
                        'Par': f"{ASSETS[a]['label']} / {ASSETS[b]['label']}",
                        'P-Value': score_coint,
                        'Correlación': corr,
                        'Half-Life': hl,
                        'Hurst': h
                    })
            except: pass
            prog.progress((i+1)/len(pairs))
            
        if results:
            res_df = pd.DataFrame(results).sort_values('P-Value')
            st.dataframe(res_df.style.format({
                'P-Value': '{:.4f}', 'Correlación': '{:.2f}', 'Half-Life': '{:.1f}', 'Hurst': '{:.2f}'
            }).background_gradient(subset=['P-Value'], cmap='RdYlGn_r'), use_container_width=True)
        else:
            st.warning("No se encontraron pares significativos.")

# TAB 5: HEDGING
with tab5:
    st.subheader("🛡️ Coberturas (Correlación Inversa)")
    st.caption("Encuentra activos que se muevan en contra para proteger tu portafolio.")
    
    target_asset = st.selectbox("Activo a Proteger", df.columns, format_func=lambda x: ASSETS[x]['label'])
    
    corrs = df.corrwith(df[target_asset]).sort_values()
    neg_corrs = corrs[corrs < -0.3]
    
    if not neg_corrs.empty:
        st.success(f"Encontrados {len(neg_corrs)} posibles hedges.")
        for asset, corr in neg_corrs.items():
            hedge_ratio = -calculate_spread(df[target_asset], df[asset])[1]
            with st.expander(f"📉 {ASSETS[asset]['label']} (Corr: {corr:.2f})"):
                c1, c2 = st.columns(2)
                c1.metric("Correlación", f"{corr:.2f}")
                c2.metric("Ratio de Hedge Sugerido", f"{hedge_ratio:.3f}")
                st.caption(f"Por cada 1 unidad de {ASSETS[target_asset]['label']}, compra {hedge_ratio:.2f} de {ASSETS[asset]['label']}")
    else:
        st.warning("No se encontraron activos con correlación inversa significativa (-0.3) para este activo.")
