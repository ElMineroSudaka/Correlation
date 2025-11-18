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
    'sp500': {'label': 'S&P 500', 'symbol': '^GSPC', 'color': '#3b82f6'},
    'nasdaq': {'label': 'NASDAQ', 'symbol': '^IXIC', 'color': '#8b5cf6'},
    'dow': {'label': 'Dow Jones', 'symbol': '^DJI', 'color': '#10b981'},
    'russell': {'label': 'Russell 2000', 'symbol': '^RUT', 'color': '#06b6d4'},
    'dxy': {'label': 'DXY (Dólar)', 'symbol': 'DX-Y.NYB', 'color': '#f59e0b'},
    'gold': {'label': 'Oro', 'symbol': 'GC=F', 'color': '#fbbf24'},
    'silver': {'label': 'Plata', 'symbol': 'SI=F', 'color': '#d1d5db'},
    'oil': {'label': 'Petróleo WTI', 'symbol': 'CL=F', 'color': '#000000'},
    'natgas': {'label': 'Gas Natural', 'symbol': 'NG=F', 'color': '#059669'},
    'us10y': {'label': 'Treasury 10Y', 'symbol': '^TNX', 'color': '#ef4444'},
    'us2y': {'label': 'Treasury 2Y', 'symbol': '^IRX', 'color': '#dc2626'},
    'vix': {'label': 'VIX (Volatilidad)', 'symbol': '^VIX', 'color': '#ec4899'},
    'btc': {'label': 'Bitcoin', 'symbol': 'BTC-USD', 'color': '#f7931a'},
    'eth': {'label': 'Ethereum', 'symbol': 'ETH-USD', 'color': '#627eea'},
    'eur': {'label': 'EUR/USD', 'symbol': 'EURUSD=X', 'color': '#3b82f6'},
    'jpy': {'label': 'USD/JPY', 'symbol': 'JPY=X', 'color': '#ef4444'},
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

tab1, tab2, tab3 = st.tabs(["📈 Análisis de Pares", "🔥 Heatmap de Correlaciones", "📊 Estadísticas"])

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
        width='stretch'
    )
    
    # Gráfico de precios
    st.plotly_chart(
        plot_price_comparison(df_prices, asset1, asset2, 
                            ASSETS[asset1]['label'], 
                            ASSETS[asset2]['label']),
        width='stretch'
    )

with tab2:
    st.subheader("Matriz de Correlaciones entre Todos los Activos")
    st.plotly_chart(
        plot_correlation_heatmap(df_prices, selected_assets),
        width='stretch'
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
    st.dataframe(styled_df, width='stretch')

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
    
    st.plotly_chart(fig_hist, width='stretch')
    
    # Descargar datos
    st.subheader("📥 Descargar Datos")
    csv = corr_df.to_csv(index=False)
    st.download_button(
        label="Descargar correlaciones como CSV",
        data=csv,
        file_name=f"correlacion_{asset1}_{asset2}.csv",
        mime="text/csv"
    )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Interpretación")
st.sidebar.markdown("""
- **> 0.5**: Fuerte correlación positiva
- **< -0.5**: Fuerte correlación negativa
- **≈ 0**: Sin correlación
""")
st.sidebar.markdown("---")
st.sidebar.info("💡 Los datos se actualizan automáticamente cada hora")
