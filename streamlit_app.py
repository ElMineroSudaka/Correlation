import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA Y ESTILOS
# =============================================================================
st.set_page_config(
    page_title="Pairs Trading Master Class",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la estética
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
    .stAlert { border-radius: 8px; }
    /* Highlight para explicaciones */
    .explanation-box {
        background-color: #131720;
        border-left: 5px solid #3b82f6;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONFIGURACIÓN DE ACTIVOS (Mantenemos tu lista extendida)
# =============================================================================
ASSETS = {
    'sp500': {'label': 'S&P 500', 'symbol': '^GSPC', 'category': 'US Equity'},
    'nasdaq': {'label': 'NASDAQ', 'symbol': '^IXIC', 'category': 'US Equity'},
    'dow': {'label': 'Dow Jones', 'symbol': '^DJI', 'category': 'US Equity'},
    'russell': {'label': 'Russell 2000', 'symbol': '^RUT', 'category': 'US Equity'},
    'ftse': {'label': 'FTSE 100', 'symbol': '^FTSE', 'category': 'Europe Equity'},
    'dax': {'label': 'DAX', 'symbol': '^GDAXI', 'category': 'Europe Equity'},
    'cac40': {'label': 'CAC 40', 'symbol': '^FCHI', 'category': 'Europe Equity'},
    'stoxx50': {'label': 'Euro Stoxx 50', 'symbol': '^STOXX50E', 'category': 'Europe Equity'},
    'nikkei': {'label': 'Nikkei 225', 'symbol': '^N225', 'category': 'Asia Equity'},
    'hang_seng': {'label': 'Hang Seng', 'symbol': '^HSI', 'category': 'Asia Equity'},
    'qqq': {'label': 'QQQ', 'symbol': 'QQQ', 'category': 'US ETF'},
    'spy': {'label': 'SPY', 'symbol': 'SPY', 'category': 'US ETF'},
    'xlk': {'label': 'XLK Tech', 'symbol': 'XLK', 'category': 'Sector ETF'},
    'xlf': {'label': 'XLF Finance', 'symbol': 'XLF', 'category': 'Sector ETF'},
    'xle': {'label': 'XLE Energy', 'symbol': 'XLE', 'category': 'Sector ETF'},
    'xlv': {'label': 'XLV Health', 'symbol': 'XLV', 'category': 'Sector ETF'},
    'dxy': {'label': 'DXY Index', 'symbol': 'DX-Y.NYB', 'category': 'FX'},
    'eurusd': {'label': 'EUR/USD', 'symbol': 'EURUSD=X', 'category': 'FX'},
    'gbpusd': {'label': 'GBP/USD', 'symbol': 'GBPUSD=X', 'category': 'FX'},
    'usdjpy': {'label': 'USD/JPY', 'symbol': 'JPYUSD=X', 'category': 'FX'},
    'gold': {'label': 'Gold Futures', 'symbol': 'GC=F', 'category': 'Metals'},
    'silver': {'label': 'Silver Futures', 'symbol': 'SI=F', 'category': 'Metals'},
    'gld': {'label': 'GLD ETF', 'symbol': 'GLD', 'category': 'Metals'},
    'slv': {'label': 'SLV ETF', 'symbol': 'SLV', 'category': 'Metals'},
    'oil': {'label': 'WTI Oil', 'symbol': 'CL=F', 'category': 'Energy'},
    'uso': {'label': 'USO ETF', 'symbol': 'USO', 'category': 'Energy'},
    'us10y': {'label': 'US 10Y Yield', 'symbol': '^TNX', 'category': 'Bonds'},
    'tlt': {'label': 'TLT Bond ETF', 'symbol': 'TLT', 'category': 'Bonds'},
    'vix': {'label': 'VIX Index', 'symbol': '^VIX', 'category': 'Volatility'},
    'btc': {'label': 'Bitcoin', 'symbol': 'BTC-USD', 'category': 'Crypto'},
    'eth': {'label': 'Ethereum', 'symbol': 'ETH-USD', 'category': 'Crypto'},
    'nvda': {'label': 'NVIDIA', 'symbol': 'NVDA', 'category': 'Tech Stocks'},
    'amd': {'label': 'AMD', 'symbol': 'AMD', 'category': 'Tech Stocks'},
    'msft': {'label': 'Microsoft', 'symbol': 'MSFT', 'category': 'Tech Stocks'},
    'aapl': {'label': 'Apple', 'symbol': 'AAPL', 'category': 'Tech Stocks'},
    'ko': {'label': 'Coca-Cola', 'symbol': 'KO', 'category': 'Consumer'},
    'pep': {'label': 'PepsiCo', 'symbol': 'PEP', 'category': 'Consumer'},
}

# =============================================================================
# FUNCIONES DE LÓGICA Y CÁLCULO
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_asset_data(symbol, start_date='2015-01-01', end_date=None):
    """Descarga datos históricos con manejo de errores."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if data.empty: return None
        # Manejo flexible de columnas de Yahoo Finance
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            return None
            
        # Si es un DataFrame multi-index (caso reciente de yfinance), aplanar
        if isinstance(prices, pd.DataFrame):
            if symbol in prices.columns:
                prices = prices[symbol]
            else:
                prices = prices.iloc[:, 0]
                
        return prices.dropna()
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def download_selected_assets(selected_keys):
    """Descarga en lote."""
    all_data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(selected_keys)
    for idx, key in enumerate(selected_keys):
        symbol = ASSETS[key]['symbol']
        status_text.caption(f"Descargando {ASSETS[key]['label']} ({idx+1}/{total})...")
        data = fetch_asset_data(symbol)
        if data is not None:
            all_data[key] = data
        progress_bar.progress((idx + 1) / total)
        time.sleep(0.1) # Pequeño delay para no saturar API
    
    progress_bar.empty()
    status_text.empty()
    return all_data

def merge_asset_data(data_dict):
    """Combina series temporales."""
    if not data_dict: return pd.DataFrame()
    
    # Asegurar que todo sean Series con nombre
    series_list = []
    for k, v in data_dict.items():
        s = v.copy()
        s.name = k
        series_list.append(s)
        
    df = pd.concat(series_list, axis=1, join='inner')
    return df

def calculate_spread(prices1, prices2):
    """
    Calcula el spread usando OLS (Ordinary Least Squares).
    Spread = Precio1 - (Hedge_Ratio * Precio2)
    """
    # Alineación de datos
    common_idx = prices1.index.intersection(prices2.index)
    p1 = prices1.loc[common_idx]
    p2 = prices2.loc[common_idx]
    
    # Regresión Lineal (OLS)
    # p1 = alpha + beta * p2 + error
    X = sm.add_constant(p2)
    model = sm.OLS(p1, X).fit()
    
    # FIX: Usar .iloc para acceso posicional seguro
    hedge_ratio = model.params.iloc[1]
    alpha = model.params.iloc[0]
    
    # El spread son los residuos del modelo (error)
    spread = p1 - (hedge_ratio * p2) - alpha 
    
    return spread, hedge_ratio, alpha, model.rsquared

def calculate_zscore(series, window=30):
    """Calcula el Z-Score rolling (desviaciones estándar de la media)."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    zscore = (series - mean) / std
    return zscore

def calculate_half_life(spread):
    """
    Calcula la Vida Media (Half-Life) de reversión a la media.
    Indica cuánto tiempo tarda el spread en revertir la mitad de su desviación.
    Basado en proceso Ornstein-Uhlenbeck.
    """
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    
    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    
    spread_lag2 = sm.add_constant(spread_lag)
    
    model = sm.OLS(spread_ret, spread_lag2)
    res = model.fit()
    
    # FIX: Usar .iloc para evitar KeyError si el índice no es numérico
    lambda_param = res.params.iloc[1]
    
    if lambda_param >= 0:
        return np.inf # No revierte a la media
        
    half_life = -np.log(2) / lambda_param
    return half_life

def calculate_hurst_exponent(series, max_lag=100):
    """Calcula Exponente Hurst para medir persistencia vs reversión."""
    lags = range(2, min(max_lag, len(series)//2))
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    try:
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return np.nan

def run_simple_backtest(spread, zscore, entry_threshold=2.0, exit_threshold=0.0):
    """
    Simulación simple de estrategia:
    - Vender Spread cuando Z > entry_threshold
    - Comprar Spread cuando Z < -entry_threshold
    - Cerrar cuando Z cruza exit_threshold
    """
    signals = pd.Series(index=spread.index, data=0)
    position = 0 # 0: flat, 1: long, -1: short
    
    positions_history = []
    
    for z in zscore:
        if position == 0:
            if z > entry_threshold:
                position = -1 # Short Spread
            elif z < -entry_threshold:
                position = 1 # Long Spread
        elif position == -1:
            if z < exit_threshold:
                position = 0 # Exit Short
        elif position == 1:
            if z > -exit_threshold:
                position = 0 # Exit Long
        positions_history.append(position)
        
    signals = pd.Series(positions_history, index=spread.index)
    
    # Calcular PnL aproximado (Spread Diff * Posición de ayer)
    spread_diff = spread.diff()
    pnl_daily = spread_diff * signals.shift(1)
    cumulative_pnl = pnl_daily.cumsum()
    
    return cumulative_pnl, signals

# =============================================================================
# INTERFAZ DE USUARIO
# =============================================================================

st.title("🎓 Pairs Trading & Statistical Arbitrage Lab")
st.markdown("""
Esta aplicación no solo analiza pares, sino que te **enseña** cómo funciona el arbitraje estadístico.
Selecciona activos, analiza su relación matemática y simula estrategias.
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración de Estudio")
    
    # Filtro de Categorías
    all_cats = sorted(list(set([v['category'] for v in ASSETS.values()])))
    sel_cats = st.multiselect("Filtrar Categorías", all_cats, default=['US Equity', 'Metals', 'Tech Stocks', 'Consumer'])
    
    # Filtro de Activos
    avail_assets = [k for k, v in ASSETS.items() if v['category'] in sel_cats]
    
    # Selección Multi
    st.subheader("1. Selección de Datos")
    selected_assets = st.multiselect(
        "Selecciona Activos (mínimo 2)", 
        avail_assets, 
        default=['ko', 'pep'] if 'ko' in avail_assets else avail_assets[:2],
        format_func=lambda x: f"{ASSETS[x]['label']} ({x.upper()})"
    )
    
    if len(selected_assets) < 2:
        st.error("Necesitas al menos 2 activos.")
        st.stop()
        
    # Botón de Descarga
    if st.button("📥 Descargar/Actualizar Datos", type="primary"):
        with st.spinner("Obteniendo datos de mercado..."):
            raw_data = download_selected_assets(selected_assets)
            st.session_state['df_prices'] = merge_asset_data(raw_data)
            st.success("Datos cargados.")
    
    st.subheader("2. Parámetros del Modelo")
    window_size = st.slider("Ventana de Rolling (Días)", 10, 100, 30, help="Días para calcular la media móvil y desviación estándar del Z-Score.")
    
    st.info("💡 **Tip:** Comienza con Coca-Cola (KO) y Pepsi (PEP) para ver una correlación clásica.")

# Verificar datos
if 'df_prices' not in st.session_state or st.session_state['df_prices'].empty:
    st.warning("👈 Por favor, selecciona tus activos y presiona 'Descargar Datos' en la barra lateral.")
    st.stop()

df = st.session_state['df_prices']

# --- TABS PRINCIPALES ---
tab_analysis, tab_backtest, tab_educational, tab_scanner = st.tabs([
    "🔬 Laboratorio de Análisis", 
    "💰 Simulación (Backtest)", 
    "📚 Conceptos Teóricos",
    "📡 Scanner Automático"
])

# -----------------------------------------------------------------------------
# TAB 1: ANÁLISIS DETALLADO
# -----------------------------------------------------------------------------
with tab_analysis:
    col_a, col_b = st.columns(2)
    with col_a:
        asset1 = st.selectbox("Activo Y (Dependiente)", df.columns, format_func=lambda x: ASSETS[x]['label'])
    with col_b:
        asset2 = st.selectbox("Activo X (Independiente)", [c for c in df.columns if c != asset1], format_func=lambda x: ASSETS[x]['label'])

    # Cálculos principales
    p1 = df[asset1]
    p2 = df[asset2]
    
    # 1. Correlación
    corr = p1.corr(p2)
    
    # 2. Spread y Cointegración
    spread, hedge_ratio, alpha, r_squared = calculate_spread(p1, p2)
    coint_res = coint(p1, p2)
    is_coint = coint_res[1] < 0.05
    
    # 3. Estadísticas Avanzadas
    half_life = calculate_half_life(spread)
    hurst = calculate_hurst_exponent(spread.dropna().values)
    zscore = calculate_zscore(spread, window_size)
    
    # --- KPIs ---
    st.markdown("### 📊 Métricas Clave")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    kpi1.metric(
        "Correlación", 
        f"{corr:.2f}", 
        delta="Fuerte" if abs(corr)>0.8 else "Débil",
        help="Mide qué tan similarmente se mueven los precios (1 = idéntico, -1 = opuesto)."
    )
    
    kpi2.metric(
        "Cointegración (p-value)", 
        f"{coint_res[1]:.4f}", 
        delta="Cointegrado ✅" if is_coint else "No Cointegrado ❌",
        delta_color="normal" if is_coint else "inverse",
        help="Si p < 0.05, el spread es estacionario (la liga elástica no se rompe)."
    )
    
    kpi3.metric(
        "Vida Media (Half-Life)", 
        f"{half_life:.1f} días",
        help="Tiempo estimado para que el spread regrese a la mitad de su camino hacia la media. Importante para saber cuánto durará el trade."
    )
    
    hurst_label = "Reversión a la Media" if hurst < 0.5 else "Tendencia/Aleatorio"
    kpi4.metric(
        "Exponente Hurst", 
        f"{hurst:.2f}",
        delta=hurst_label,
        delta_color="normal" if hurst < 0.5 else "off",
        help="Hurst < 0.5 indica que si se aleja, tiende a volver."
    )

    # --- GRÁFICOS ---
    
    # 1. Comparación Normalizada
    st.subheader("1. Dinámica de Precios (Base 100)")
    fig_norm = go.Figure()
    fig_norm.add_trace(go.Scatter(x=p1.index, y=(p1/p1.iloc[0])*100, name=ASSETS[asset1]['label']))
    fig_norm.add_trace(go.Scatter(x=p2.index, y=(p2/p2.iloc[0])*100, name=ASSETS[asset2]['label']))
    fig_norm.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_norm, width="stretch")
    
    # 2. Scatter Plot (Regresión)
    st.subheader("2. Análisis de Regresión Lineal")
    with st.expander("¿Qué significa este gráfico?"):
        st.write(f"""
        Este gráfico muestra la relación precio a precio. La línea roja es el 'Valor Justo' predicho por el modelo.
        La ecuación es: **{ASSETS[asset1]['label']} = {alpha:.2f} + {hedge_ratio:.2f} * {ASSETS[asset2]['label']}**
        """)
    
    fig_scatter = px.scatter(x=p2, y=p1, labels={'x': ASSETS[asset2]['label'], 'y': ASSETS[asset1]['label']}, opacity=0.6)
    # Añadir línea de regresión manualmente para control
    line_x = np.linspace(p2.min(), p2.max(), 100)
    line_y = alpha + hedge_ratio * line_x
    fig_scatter.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='Regresión (Hedge Ratio)', line=dict(color='red')))
    fig_scatter.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig_scatter, width="stretch")

    # 3. Spread y Z-Score
    st.subheader("3. Señales de Trading (Spread & Z-Score)")
    
    fig_z = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.6, 0.4],
                          subplot_titles=("Spread Histórico (Residuos)", f"Z-Score (Normalizado a {window_size} días)"))
    
    # Spread
    fig_z.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread", line=dict(color='#3b82f6')), row=1, col=1)
    fig_z.add_hline(y=spread.mean(), line_dash="dash", row=1, col=1, annotation_text="Media")
    
    # Z-Score
    fig_z.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-Score", line=dict(color='#eab308')), row=2, col=1)
    fig_z.add_hline(y=2.0, line_dash="dot", line_color="red", row=2, col=1, annotation_text="Venta (+2)")
    fig_z.add_hline(y=-2.0, line_dash="dot", line_color="green", row=2, col=1, annotation_text="Compra (-2)")
    fig_z.add_hline(y=0, line_color="gray", row=2, col=1)
    
    # Marcar zonas de compra/venta
    buy_signals = zscore[zscore < -2]
    sell_signals = zscore[zscore > 2]
    
    fig_z.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals, mode='markers', marker=dict(color='green', size=8), name='Señal Compra'), row=2, col=1)
    fig_z.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals, mode='markers', marker=dict(color='red', size=8), name='Señal Venta'), row=2, col=1)

    fig_z.update_layout(template="plotly_dark", height=700)
    st.plotly_chart(fig_z, width="stretch")

# -----------------------------------------------------------------------------
# TAB 2: BACKTEST
# -----------------------------------------------------------------------------
with tab_backtest:
    st.header("💰 Simulador de Estrategia (Backtest)")
    st.markdown("""
    Aquí simulamos qué hubiera pasado si hubieras operado este par usando bandas de desviación estándar.
    *Nota: Esto es una simulación teórica simplificada.*
    """)
    
    col_b1, col_b2, col_b3 = st.columns(3)
    entry_z = col_b1.number_input("Entrar cuando Z-Score >", 1.0, 4.0, 2.0, 0.1)
    exit_z = col_b2.number_input("Salir cuando Z-Score cruza", 0.0, 1.0, 0.0, 0.1)
    
    if st.button("🚀 Ejecutar Simulación"):
        cum_pnl, signals = run_simple_backtest(spread, zscore, entry_z, exit_z)
        
        # Métricas Backtest
        total_ret = cum_pnl.iloc[-1]
        n_trades = signals.diff().abs().sum() / 2 # Aprox
        
        bk1, bk2 = st.columns(2)
        bk1.metric("PnL Acumulado (Unidades de Spread)", f"{total_ret:.4f}")
        bk2.metric("Operaciones Aprox.", f"{int(n_trades)}")
        
        # Gráfico PnL
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(x=cum_pnl.index, y=cum_pnl, fill='tozeroy', mode='lines', name='PnL Acumulado'))
        fig_pnl.update_layout(title="Curva de Equity (Teórica)", template="plotly_dark", height=400)
        st.plotly_chart(fig_pnl, width="stretch")
        
        # Gráfico Posiciones
        fig_pos = go.Figure()
        # FIX: mode='lines' y shape='hv' para efecto escalón, 'steps' no es válido
        fig_pos.add_trace(go.Scatter(x=signals.index, y=signals, mode='lines', line=dict(shape='hv'), name='Posición (-1, 0, 1)'))
        fig_pos.update_layout(title="Posiciones en el Mercado", yaxis=dict(tickvals=[-1, 0, 1], ticktext=['Short Spread', 'Flat', 'Long Spread']), template="plotly_dark", height=300)
        st.plotly_chart(fig_pos, width="stretch")

# -----------------------------------------------------------------------------
# TAB 3: EDUCACIÓN
# -----------------------------------------------------------------------------
with tab_educational:
    st.header("📚 Conceptos Fundamentales")
    
    st.markdown("""
    <div class="explanation-box">
    <h3>1. ¿Qué es el Pairs Trading?</h3>
    Es una estrategia neutral de mercado. No te importa si el mercado sube o baja, solo te importa la relación entre dos activos.
    <br>Ejemplo: Si Coca-Cola y Pepsi siempre se mueven igual, pero hoy Coca-Cola sube mucho y Pepsi no, vendes Coca-Cola y compras Pepsi, esperando que vuelvan a alinearse.
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🧩 Cointegración vs Correlación (La diferencia clave)"):
        st.write("""
        - **Correlación:** Mide si dos activos suben o bajan al mismo tiempo *diariamente*.
        - **Cointegración:** Mide si la *distancia* entre dos precios se mantiene estable a largo plazo.
        
        **Analogía del Borracho y el Perro:**
        Imagina un borracho paseando a su perro con una correa elástica. 
        - Caminan erráticamente (random walk).
        - A veces el perro se aleja, a veces se acerca.
        - Pero la correa (cointegración) asegura que nunca se separen demasiado.
        """)
    
    with st.expander("📉 ¿Qué es el Z-Score?"):
        st.write("""
        El Z-Score nos dice **cuántas desviaciones estándar** está el precio actual lejos de su media histórica.
        - **Z = 0:** El spread está en su precio justo promedio.
        - **Z = +2:** El spread está "caro" (estadísticamente improbable). Señal de Venta.
        - **Z = -2:** El spread está "barato". Señal de Compra.
        """)
        
    with st.expander("⏱️ Vida Media (Half-Life)"):
        st.write("""
        Es el tiempo esperado para que el spread corrija la mitad de su desviación.
        - Si Half-Life = 5 días, y entras en una operación, espera mantenerla al menos unas semanas.
        - Si Half-Life es muy alto (ej. > 60 días), la reversión es tan lenta que quizás no valga la pena operar (costos de swap/intereses).
        """)

# -----------------------------------------------------------------------------
# TAB 4: SCANNER AUTOMÁTICO
# -----------------------------------------------------------------------------
with tab_scanner:
    st.header("📡 Radar de Oportunidades")
    
    if st.button("Escanear todos los pares seleccionados"):
        pairs_list = []
        assets_list = df.columns
        
        progress_bar = st.progress(0)
        combinations = []
        # Generar combinaciones únicas
        for i, a1 in enumerate(assets_list):
            for a2 in assets_list[i+1:]:
                combinations.append((a1, a2))
        
        total_comb = len(combinations)
        
        with st.spinner(f"Analizando {total_comb} pares..."):
            for idx, (a1, a2) in enumerate(combinations):
                try:
                    p_a = df[a1]
                    p_b = df[a2]
                    
                    # Tests rápidos
                    score_corr = p_a.corr(p_b)
                    if score_corr < 0.5: continue # Filtro rápido
                    
                    c_res = coint(p_a, p_b)
                    if c_res[1] < 0.10: # Pre-filtro cointegración laxa
                        spr, hr, _, _ = calculate_spread(p_a, p_b)
                        hl = calculate_half_life(spr)
                        zs = calculate_zscore(spr).iloc[-1]
                        hurst_val = calculate_hurst_exponent(spr.dropna().values)
                        
                        pairs_list.append({
                            'Par': f"{ASSETS[a1]['label']} / {ASSETS[a2]['label']}",
                            'Correlación': score_corr,
                            'Cointegración (p-val)': c_res[1],
                            'Z-Score Actual': zs,
                            'Half-Life (Días)': hl,
                            'Hurst': hurst_val
                        })
                except:
                    pass
                progress_bar.progress((idx+1)/total_comb)
        
        if pairs_list:
            res_df = pd.DataFrame(pairs_list)
            # Ordenar por mejores oportunidades (Z-Score extremo y baja p-value)
            res_df['Abs Z'] = res_df['Z-Score Actual'].abs()
            res_df = res_df.sort_values(by=['Cointegración (p-val)', 'Abs Z'], ascending=[True, False])
            del res_df['Abs Z']
            
            st.success(f"Se encontraron {len(res_df)} pares interesantes.")
            
            # Formateo de color para la tabla
            def highlight_rows(val):
                if val < 0.05: return 'background-color: #1a472a' # Verde oscuro
                return ''

            st.dataframe(
                res_df.style.format({
                    'Correlación': '{:.2f}',
                    'Cointegración (p-val)': '{:.4f}',
                    'Z-Score Actual': '{:.2f}',
                    'Half-Life (Días)': '{:.1f}',
                    'Hurst': '{:.2f}'
                }).background_gradient(subset=['Z-Score Actual'], cmap='RdYlGn'),
                use_container_width=True
            )
        else:
            st.warning("No se encontraron pares con alta correlación en esta selección.")
