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
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Advanced Correlation & Trading Analyzer",
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

# Configuración EXPANDIDA de activos (120+ activos)
ASSETS = {
    # ============ ÍNDICES DE ACCIONES GLOBALES ============
    # Estados Unidos
    'sp500': {'label': 'S&P 500', 'symbol': '^GSPC', 'color': '#3b82f6', 'risk': 'Risk On', 'category': 'US Equity'},
    'nasdaq': {'label': 'NASDAQ', 'symbol': '^IXIC', 'color': '#8b5cf6', 'risk': 'Risk On', 'category': 'US Equity'},
    'dow': {'label': 'Dow Jones', 'symbol': '^DJI', 'color': '#10b981', 'risk': 'Risk On', 'category': 'US Equity'},
    'russell': {'label': 'Russell 2000', 'symbol': '^RUT', 'color': '#06b6d4', 'risk': 'Risk On', 'category': 'US Equity'},
    'sp400': {'label': 'S&P 400 MidCap', 'symbol': '^MID', 'color': '#14b8a6', 'risk': 'Risk On', 'category': 'US Equity'},
    'sp600': {'label': 'S&P 600 SmallCap', 'symbol': '^SML', 'color': '#0ea5e9', 'risk': 'Risk On', 'category': 'US Equity'},
    'nyse': {'label': 'NYSE Composite', 'symbol': '^NYA', 'color': '#06b6d4', 'risk': 'Risk On', 'category': 'US Equity'},
    
    # Europa
    'ftse': {'label': 'FTSE 100 (UK)', 'symbol': '^FTSE', 'color': '#f97316', 'risk': 'Risk On', 'category': 'Europe Equity'},
    'dax': {'label': 'DAX (Germany)', 'symbol': '^GDAXI', 'color': '#eab308', 'risk': 'Risk On', 'category': 'Europe Equity'},
    'cac40': {'label': 'CAC 40 (France)', 'symbol': '^FCHI', 'color': '#84cc16', 'risk': 'Risk On', 'category': 'Europe Equity'},
    'ibex': {'label': 'IBEX 35 (Spain)', 'symbol': '^IBEX', 'color': '#22c55e', 'risk': 'Risk On', 'category': 'Europe Equity'},
    'ftse_mib': {'label': 'FTSE MIB (Italy)', 'symbol': 'FTSEMIB.MI', 'color': '#10b981', 'risk': 'Risk On', 'category': 'Europe Equity'},
    'aex': {'label': 'AEX (Netherlands)', 'symbol': '^AEX', 'color': '#14b8a6', 'risk': 'Risk On', 'category': 'Europe Equity'},
    'stoxx50': {'label': 'Euro Stoxx 50', 'symbol': '^STOXX50E', 'color': '#06b6d4', 'risk': 'Risk On', 'category': 'Europe Equity'},
    'stoxx600': {'label': 'STOXX 600', 'symbol': '^STOXX', 'color': '#0ea5e9', 'risk': 'Risk On', 'category': 'Europe Equity'},
    
    # Asia-Pacífico
    'nikkei': {'label': 'Nikkei 225 (Japan)', 'symbol': '^N225', 'color': '#ec4899', 'risk': 'Risk On', 'category': 'Asia Equity'},
    'hang_seng': {'label': 'Hang Seng (Hong Kong)', 'symbol': '^HSI', 'color': '#d946ef', 'risk': 'Risk On', 'category': 'Asia Equity'},
    'shanghai': {'label': 'Shanghai Composite', 'symbol': '000001.SS', 'color': '#c026d3', 'risk': 'Risk On', 'category': 'Asia Equity'},
    'shenzhen': {'label': 'Shenzhen Component', 'symbol': '399001.SZ', 'color': '#a21caf', 'risk': 'Risk On', 'category': 'Asia Equity'},
    'kospi': {'label': 'KOSPI (South Korea)', 'symbol': '^KS11', 'color': '#86198f', 'risk': 'Risk On', 'category': 'Asia Equity'},
    'asx200': {'label': 'ASX 200 (Australia)', 'symbol': '^AXJO', 'color': '#f472b6', 'risk': 'Risk On', 'category': 'Asia Equity'},
    'nzx50': {'label': 'NZX 50 (New Zealand)', 'symbol': '^NZ50', 'color': '#ec4899', 'risk': 'Risk On', 'category': 'Asia Equity'},
    'sensex': {'label': 'BSE Sensex (India)', 'symbol': '^BSESN', 'color': '#db2777', 'risk': 'Risk On', 'category': 'Asia Equity'},
    'nifty50': {'label': 'Nifty 50 (India)', 'symbol': '^NSEI', 'color': '#be185d', 'risk': 'Risk On', 'category': 'Asia Equity'},
    'taiwan': {'label': 'Taiwan Weighted', 'symbol': '^TWII', 'color': '#9f1239', 'risk': 'Risk On', 'category': 'Asia Equity'},
    
    # Latinoamérica
    'bovespa': {'label': 'Bovespa (Brazil)', 'symbol': '^BVSP', 'color': '#84cc16', 'risk': 'Risk On', 'category': 'LatAm Equity'},
    'merval': {'label': 'Merval (Argentina)', 'symbol': '^MERV', 'color': '#65a30d', 'risk': 'Risk On', 'category': 'LatAm Equity'},
    'ipc': {'label': 'IPC (Mexico)', 'symbol': '^MXX', 'color': '#4d7c0f', 'risk': 'Risk On', 'category': 'LatAm Equity'},
    
    # Emergentes
    'emerging': {'label': 'MSCI Emerging Markets', 'symbol': 'EEM', 'color': '#ec4899', 'risk': 'Risk On', 'category': 'EM Equity'},
    'eafe': {'label': 'MSCI EAFE', 'symbol': 'EFA', 'color': '#d946ef', 'risk': 'Risk On', 'category': 'Intl Equity'},
    
    # ============ ETFs SECTORIALES US ============
    'qqq': {'label': 'QQQ (Nasdaq ETF)', 'symbol': 'QQQ', 'color': '#8b5cf6', 'risk': 'Risk On', 'category': 'US ETF'},
    'iwm': {'label': 'IWM (Russell 2000 ETF)', 'symbol': 'IWM', 'color': '#06b6d4', 'risk': 'Risk On', 'category': 'US ETF'},
    'dia': {'label': 'DIA (Dow Jones ETF)', 'symbol': 'DIA', 'color': '#10b981', 'risk': 'Risk On', 'category': 'US ETF'},
    'spy': {'label': 'SPY (S&P 500 ETF)', 'symbol': 'SPY', 'color': '#3b82f6', 'risk': 'Risk On', 'category': 'US ETF'},
    'xlk': {'label': 'XLK (Technology)', 'symbol': 'XLK', 'color': '#8b5cf6', 'risk': 'Risk On', 'category': 'Sector ETF'},
    'xlf': {'label': 'XLF (Financials)', 'symbol': 'XLF', 'color': '#10b981', 'risk': 'Risk On', 'category': 'Sector ETF'},
    'xle': {'label': 'XLE (Energy)', 'symbol': 'XLE', 'color': '#000000', 'risk': 'Risk On', 'category': 'Sector ETF'},
    'xlv': {'label': 'XLV (Healthcare)', 'symbol': 'XLV', 'color': '#dc2626', 'risk': 'Risk Off', 'category': 'Sector ETF'},
    'xly': {'label': 'XLY (Consumer Discretionary)', 'symbol': 'XLY', 'color': '#ec4899', 'risk': 'Risk On', 'category': 'Sector ETF'},
    'xlp': {'label': 'XLP (Consumer Staples)', 'symbol': 'XLP', 'color': '#22c55e', 'risk': 'Risk Off', 'category': 'Sector ETF'},
    'xlu': {'label': 'XLU (Utilities)', 'symbol': 'XLU', 'color': '#eab308', 'risk': 'Risk Off', 'category': 'Sector ETF'},
    'xlb': {'label': 'XLB (Materials)', 'symbol': 'XLB', 'color': '#c2410c', 'risk': 'Risk On', 'category': 'Sector ETF'},
    'xli': {'label': 'XLI (Industrials)', 'symbol': 'XLI', 'color': '#0ea5e9', 'risk': 'Risk On', 'category': 'Sector ETF'},
    'xlre': {'label': 'XLRE (Real Estate)', 'symbol': 'XLRE', 'color': '#f59e0b', 'risk': 'Risk On', 'category': 'Sector ETF'},
    'xlc': {'label': 'XLC (Communication)', 'symbol': 'XLC', 'color': '#8b5cf6', 'risk': 'Risk On', 'category': 'Sector ETF'},
    
    # ETFs Temáticos
    'ark': {'label': 'ARKK (Innovation)', 'symbol': 'ARKK', 'color': '#a855f7', 'risk': 'Risk On', 'category': 'Thematic ETF'},
    'arkw': {'label': 'ARKW (Next Gen Internet)', 'symbol': 'ARKW', 'color': '#9333ea', 'risk': 'Risk On', 'category': 'Thematic ETF'},
    'arkg': {'label': 'ARKG (Genomics)', 'symbol': 'ARKG', 'color': '#7c3aed', 'risk': 'Risk On', 'category': 'Thematic ETF'},
    'icln': {'label': 'ICLN (Clean Energy)', 'symbol': 'ICLN', 'color': '#22c55e', 'risk': 'Risk On', 'category': 'Thematic ETF'},
    'tan': {'label': 'TAN (Solar)', 'symbol': 'TAN', 'color': '#fbbf24', 'risk': 'Risk On', 'category': 'Thematic ETF'},
    'lit': {'label': 'LIT (Lithium)', 'symbol': 'LIT', 'color': '#d1d5db', 'risk': 'Risk On', 'category': 'Thematic ETF'},
    
    # ============ DIVISAS MAYORES ============
    'dxy': {'label': 'DXY (Dólar Index)', 'symbol': 'DX-Y.NYB', 'color': '#f59e0b', 'risk': 'Risk Off', 'category': 'FX Major'},
    'eurusd': {'label': 'EUR/USD', 'symbol': 'EURUSD=X', 'color': '#3b82f6', 'risk': 'Neutral', 'category': 'FX Major'},
    'gbpusd': {'label': 'GBP/USD', 'symbol': 'GBPUSD=X', 'color': '#10b981', 'risk': 'Risk On', 'category': 'FX Major'},
    'usdjpy': {'label': 'USD/JPY', 'symbol': 'JPYUSD=X', 'color': '#ef4444', 'risk': 'Risk Off', 'category': 'FX Major'},
    'usdchf': {'label': 'USD/CHF', 'symbol': 'CHFUSD=X', 'color': '#dc2626', 'risk': 'Risk Off', 'category': 'FX Major'},
    'audusd': {'label': 'AUD/USD', 'symbol': 'AUDUSD=X', 'color': '#10b981', 'risk': 'Risk On', 'category': 'FX Major'},
    'nzdusd': {'label': 'NZD/USD', 'symbol': 'NZDUSD=X', 'color': '#059669', 'risk': 'Risk On', 'category': 'FX Major'},
    'usdcad': {'label': 'USD/CAD', 'symbol': 'CADUSD=X', 'color': '#f97316', 'risk': 'Neutral', 'category': 'FX Major'},
    
    # Divisas Emergentes
    'usdbrl': {'label': 'USD/BRL', 'symbol': 'BRL=X', 'color': '#84cc16', 'risk': 'Risk On', 'category': 'FX EM'},
    'usdmxn': {'label': 'USD/MXN', 'symbol': 'MXN=X', 'color': '#22c55e', 'risk': 'Risk On', 'category': 'FX EM'},
    'usdcny': {'label': 'USD/CNY', 'symbol': 'CNY=X', 'color': '#dc2626', 'risk': 'Risk On', 'category': 'FX EM'},
    'usdinr': {'label': 'USD/INR', 'symbol': 'INR=X', 'color': '#f97316', 'risk': 'Risk On', 'category': 'FX EM'},
    'usdzar': {'label': 'USD/ZAR', 'symbol': 'ZAR=X', 'color': '#eab308', 'risk': 'Risk On', 'category': 'FX EM'},
    'usdtry': {'label': 'USD/TRY', 'symbol': 'TRY=X', 'color': '#ef4444', 'risk': 'Risk On', 'category': 'FX EM'},
    'usdrub': {'label': 'USD/RUB', 'symbol': 'RUB=X', 'color': '#b91c1c', 'risk': 'Risk On', 'category': 'FX EM'},
    
    # Cruces de Divisas
    'eurgbp': {'label': 'EUR/GBP', 'symbol': 'EURGBP=X', 'color': '#3b82f6', 'risk': 'Neutral', 'category': 'FX Cross'},
    'eurjpy': {'label': 'EUR/JPY', 'symbol': 'EURJPY=X', 'color': '#8b5cf6', 'risk': 'Risk On', 'category': 'FX Cross'},
    'eurchf': {'label': 'EUR/CHF', 'symbol': 'EURCHF=X', 'color': '#06b6d4', 'risk': 'Neutral', 'category': 'FX Cross'},
    'gbpjpy': {'label': 'GBP/JPY', 'symbol': 'GBPJPY=X', 'color': '#ec4899', 'risk': 'Risk On', 'category': 'FX Cross'},
    'audjpy': {'label': 'AUD/JPY', 'symbol': 'AUDJPY=X', 'color': '#f59e0b', 'risk': 'Risk On', 'category': 'FX Cross'},
    'nzdjpy': {'label': 'NZD/JPY', 'symbol': 'NZDJPY=X', 'color': '#10b981', 'risk': 'Risk On', 'category': 'FX Cross'},
    
    # ============ METALES PRECIOSOS ============
    'gold': {'label': 'Oro (GC)', 'symbol': 'GC=F', 'color': '#fbbf24', 'risk': 'Risk Off', 'category': 'Precious Metals'},
    'silver': {'label': 'Plata (SI)', 'symbol': 'SI=F', 'color': '#d1d5db', 'risk': 'Risk On', 'category': 'Precious Metals'},
    'platinum': {'label': 'Platino (PL)', 'symbol': 'PL=F', 'color': '#9ca3af', 'risk': 'Risk On', 'category': 'Precious Metals'},
    'palladium': {'label': 'Paladio (PA)', 'symbol': 'PA=F', 'color': '#6b7280', 'risk': 'Risk On', 'category': 'Precious Metals'},
    'gld': {'label': 'GLD (Gold ETF)', 'symbol': 'GLD', 'color': '#fbbf24', 'risk': 'Risk Off', 'category': 'Precious Metals'},
    'slv': {'label': 'SLV (Silver ETF)', 'symbol': 'SLV', 'color': '#d1d5db', 'risk': 'Risk On', 'category': 'Precious Metals'},
    
    # ============ COMMODITIES ENERGÍA ============
    'oil': {'label': 'Petróleo WTI', 'symbol': 'CL=F', 'color': '#000000', 'risk': 'Risk On', 'category': 'Energy'},
    'brent': {'label': 'Petróleo Brent', 'symbol': 'BZ=F', 'color': '#171717', 'risk': 'Risk On', 'category': 'Energy'},
    'natgas': {'label': 'Gas Natural', 'symbol': 'NG=F', 'color': '#059669', 'risk': 'Risk On', 'category': 'Energy'},
    'heating_oil': {'label': 'Heating Oil', 'symbol': 'HO=F', 'color': '#dc2626', 'risk': 'Risk On', 'category': 'Energy'},
    'gasoline': {'label': 'Gasoline', 'symbol': 'RB=F', 'color': '#f59e0b', 'risk': 'Risk On', 'category': 'Energy'},
    'uso': {'label': 'USO (Oil ETF)', 'symbol': 'USO', 'color': '#000000', 'risk': 'Risk On', 'category': 'Energy'},
    'ung': {'label': 'UNG (Nat Gas ETF)', 'symbol': 'UNG', 'color': '#059669', 'risk': 'Risk On', 'category': 'Energy'},
    
    # ============ COMMODITIES METALES INDUSTRIALES ============
    'copper': {'label': 'Cobre', 'symbol': 'HG=F', 'color': '#c2410c', 'risk': 'Risk On', 'category': 'Industrial Metals'},
    'aluminum': {'label': 'Aluminio', 'symbol': 'ALI=F', 'color': '#94a3b8', 'risk': 'Risk On', 'category': 'Industrial Metals'},
    'zinc': {'label': 'Zinc', 'symbol': 'ZNC=F', 'color': '#64748b', 'risk': 'Risk On', 'category': 'Industrial Metals'},
    'nickel': {'label': 'Níquel', 'symbol': 'NKL=F', 'color': '#475569', 'risk': 'Risk On', 'category': 'Industrial Metals'},
    
    # ============ COMMODITIES AGRÍCOLAS ============
    'corn': {'label': 'Maíz', 'symbol': 'ZC=F', 'color': '#fbbf24', 'risk': 'Risk On', 'category': 'Agriculture'},
    'wheat': {'label': 'Trigo', 'symbol': 'ZW=F', 'color': '#f59e0b', 'risk': 'Risk On', 'category': 'Agriculture'},
    'soybeans': {'label': 'Soja', 'symbol': 'ZS=F', 'color': '#84cc16', 'risk': 'Risk On', 'category': 'Agriculture'},
    'sugar': {'label': 'Azúcar', 'symbol': 'SB=F', 'color': '#ffffff', 'risk': 'Risk On', 'category': 'Agriculture'},
    'coffee': {'label': 'Café', 'symbol': 'KC=F', 'color': '#78350f', 'risk': 'Risk On', 'category': 'Agriculture'},
    'cocoa': {'label': 'Cacao', 'symbol': 'CC=F', 'color': '#451a03', 'risk': 'Risk On', 'category': 'Agriculture'},
    'cotton': {'label': 'Algodón', 'symbol': 'CT=F', 'color': '#e5e7eb', 'risk': 'Risk On', 'category': 'Agriculture'},
    'live_cattle': {'label': 'Ganado Vivo', 'symbol': 'LE=F', 'color': '#dc2626', 'risk': 'Risk On', 'category': 'Agriculture'},
    'lean_hogs': {'label': 'Cerdos', 'symbol': 'HE=F', 'color': '#f87171', 'risk': 'Risk On', 'category': 'Agriculture'},
    
    # ============ BONOS Y TASAS ============
    'us10y': {'label': 'Treasury 10Y Yield', 'symbol': '^TNX', 'color': '#ef4444', 'risk': 'Risk Off', 'category': 'Bonds'},
    'us2y': {'label': 'Treasury 2Y Yield', 'symbol': '^IRX', 'color': '#dc2626', 'risk': 'Risk Off', 'category': 'Bonds'},
    'us30y': {'label': 'Treasury 30Y Yield', 'symbol': '^TYX', 'color': '#b91c1c', 'risk': 'Risk Off', 'category': 'Bonds'},
    'us5y': {'label': 'Treasury 5Y Yield', 'symbol': '^FVX', 'color': '#f87171', 'risk': 'Risk Off', 'category': 'Bonds'},
    'tlt': {'label': 'TLT (20Y+ Treasury)', 'symbol': 'TLT', 'color': '#b91c1c', 'risk': 'Risk Off', 'category': 'Bonds'},
    'ief': {'label': 'IEF (7-10Y Treasury)', 'symbol': 'IEF', 'color': '#dc2626', 'risk': 'Risk Off', 'category': 'Bonds'},
    'shy': {'label': 'SHY (1-3Y Treasury)', 'symbol': 'SHY', 'color': '#f87171', 'risk': 'Risk Off', 'category': 'Bonds'},
    'agg': {'label': 'AGG (Aggregate Bond)', 'symbol': 'AGG', 'color': '#6b7280', 'risk': 'Risk Off', 'category': 'Bonds'},
    'hyg': {'label': 'HYG (High Yield)', 'symbol': 'HYG', 'color': '#f59e0b', 'risk': 'Risk On', 'category': 'Bonds'},
    'lqd': {'label': 'LQD (Investment Grade)', 'symbol': 'LQD', 'color': '#3b82f6', 'risk': 'Risk Off', 'category': 'Bonds'},
    'emb': {'label': 'EMB (Emerging Bonds)', 'symbol': 'EMB', 'color': '#ec4899', 'risk': 'Risk On', 'category': 'Bonds'},
    
    # ============ VOLATILIDAD ============
    'vix': {'label': 'VIX (S&P 500 Volatility)', 'symbol': '^VIX', 'color': '#ec4899', 'risk': 'Risk Off', 'category': 'Volatility'},
    'vxn': {'label': 'VXN (Nasdaq Volatility)', 'symbol': '^VXN', 'color': '#a855f7', 'risk': 'Risk Off', 'category': 'Volatility'},
    'rvx': {'label': 'RVX (Russell Volatility)', 'symbol': '^RVX', 'color': '#d946ef', 'risk': 'Risk Off', 'category': 'Volatility'},
    'vvix': {'label': 'VVIX (VIX Volatility)', 'symbol': '^VVIX', 'color': '#c026d3', 'risk': 'Risk Off', 'category': 'Volatility'},
    'move': {'label': 'MOVE (Bond Volatility)', 'symbol': '^MOVE', 'color': '#9333ea', 'risk': 'Risk Off', 'category': 'Volatility'},
    
    # ============ CRIPTOMONEDAS ============
    'btc': {'label': 'Bitcoin', 'symbol': 'BTC-USD', 'color': '#f7931a', 'risk': 'Risk On', 'category': 'Crypto'},
    'eth': {'label': 'Ethereum', 'symbol': 'ETH-USD', 'color': '#627eea', 'risk': 'Risk On', 'category': 'Crypto'},
    'bnb': {'label': 'Binance Coin', 'symbol': 'BNB-USD', 'color': '#f3ba2f', 'risk': 'Risk On', 'category': 'Crypto'},
    'xrp': {'label': 'Ripple', 'symbol': 'XRP-USD', 'color': '#00aae4', 'risk': 'Risk On', 'category': 'Crypto'},
    'ada': {'label': 'Cardano', 'symbol': 'ADA-USD', 'color': '#0033ad', 'risk': 'Risk On', 'category': 'Crypto'},
    'sol': {'label': 'Solana', 'symbol': 'SOL-USD', 'color': '#14f195', 'risk': 'Risk On', 'category': 'Crypto'},
    'doge': {'label': 'Dogecoin', 'symbol': 'DOGE-USD', 'color': '#c2a633', 'risk': 'Risk On', 'category': 'Crypto'},
    'dot': {'label': 'Polkadot', 'symbol': 'DOT-USD', 'color': '#e6007a', 'risk': 'Risk On', 'category': 'Crypto'},
    'matic': {'label': 'Polygon', 'symbol': 'MATIC-USD', 'color': '#8247e5', 'risk': 'Risk On', 'category': 'Crypto'},
    'link': {'label': 'Chainlink', 'symbol': 'LINK-USD', 'color': '#2a5ada', 'risk': 'Risk On', 'category': 'Crypto'},
    
    # ============ REITs ============
    'vnq': {'label': 'VNQ (REIT ETF)', 'symbol': 'VNQ', 'color': '#f59e0b', 'risk': 'Risk On', 'category': 'Real Estate'},
    'ita': {'label': 'ITA (Aerospace)', 'symbol': 'ITA', 'color': '#0ea5e9', 'risk': 'Risk On', 'category': 'Sector ETF'},
    'xhb': {'label': 'XHB (Homebuilders)', 'symbol': 'XHB', 'color': '#c2410c', 'risk': 'Risk On', 'category': 'Sector ETF'},
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
def download_selected_assets(selected_keys, delay=10):
    """Descarga solo los activos seleccionados con delay de 10 segundos"""
    all_data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, key in enumerate(selected_keys):
        asset_info = ASSETS[key]
        symbol = asset_info['symbol']
        
        status_text.text(f"Descargando {asset_info['label']} ({idx+1}/{len(selected_keys)})... ⏳ {delay}s delay")
        
        data = fetch_asset_data(symbol)
        
        if data is not None:
            all_data[key] = data
        else:
            st.warning(f"⚠️ No se pudo descargar {asset_info['label']}")
        
        progress_bar.progress((idx + 1) / len(selected_keys))
        
        # Delay de 10 segundos entre descargas para no saturar la API
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

def calculate_treynor_ratio(returns_asset, returns_market, rf_rate=0.02, window=252):
    """Calcula Treynor Ratio"""
    beta = calculate_beta(returns_asset, returns_market, window)
    asset_return = returns_asset.rolling(window).mean() * 252
    return (asset_return - rf_rate) / beta

def calculate_m2_measure(returns_asset, returns_market, rf_rate=0.02, window=252):
    """Calcula M2 (Modigliani-Modigliani measure)"""
    sharpe_asset = calculate_sharpe_ratio(returns_asset, rf_rate, window)
    market_vol = calculate_volatility(returns_market, window, 'historical')
    return sharpe_asset * market_vol + rf_rate

def calculate_upside_capture(returns_asset, returns_benchmark, window=252):
    """Calcula Upside Capture Ratio"""
    positive_benchmark = returns_benchmark[returns_benchmark > 0]
    positive_asset = returns_asset[returns_benchmark > 0]
    
    upside_asset = positive_asset.rolling(window).mean() * 252
    upside_benchmark = positive_benchmark.rolling(window).mean() * 252
    
    return (upside_asset / upside_benchmark) * 100

def calculate_downside_capture(returns_asset, returns_benchmark, window=252):
    """Calcula Downside Capture Ratio"""
    negative_benchmark = returns_benchmark[returns_benchmark < 0]
    negative_asset = returns_asset[returns_benchmark < 0]
    
    downside_asset = negative_asset.rolling(window).mean() * 252
    downside_benchmark = negative_benchmark.rolling(window).mean() * 252
    
    return (downside_asset / downside_benchmark) * 100

def test_cointegration(prices1, prices2):
    """Test de cointegración de Engle-Granger"""
    try:
        score, pvalue, _ = coint(prices1, prices2)
        return {'score': score, 'pvalue': pvalue, 'cointegrated': pvalue < 0.05}
    except:
        return {'score': np.nan, 'pvalue': np.nan, 'cointegrated': False}

def johansen_test(prices_df):
    """
    Test de cointegración de Johansen (para múltiples series)
    Más robusto que Engle-Granger para múltiples activos
    """
    try:
        result = coint_johansen(prices_df, det_order=0, k_ar_diff=1)
        # Retorna el número de vectores de cointegración
        trace_stats = result.lr1
        critical_values = result.cvt[:, 1]  # 95% confidence
        n_coint = sum(trace_stats > critical_values)
        return {
            'n_cointegrated': n_coint,
            'trace_stats': trace_stats,
            'critical_values': critical_values
        }
    except:
        return {'n_cointegrated': 0, 'trace_stats': [], 'critical_values': []}

def granger_causality_test(series1, series2, max_lag=5):
    """
    Test de causalidad de Granger
    Determina si una serie ayuda a predecir la otra
    """
    try:
        df_test = pd.DataFrame({'y': series1, 'x': series2}).dropna()
        result = grangercausalitytests(df_test, max_lag, verbose=False)
        
        # Extraer p-values para cada lag
        p_values = []
        for lag in range(1, max_lag + 1):
            p_value = result[lag][0]['ssr_ftest'][1]
            p_values.append(p_value)
        
        # Causalidad si algún p-value < 0.05
        causes = any(p < 0.05 for p in p_values)
        
        return {
            'causes': causes,
            'p_values': p_values,
            'best_lag': np.argmin(p_values) + 1 if causes else None
        }
    except:
        return {'causes': False, 'p_values': [], 'best_lag': None}

def calculate_spread(prices1, prices2):
    """Calcula el spread entre dos activos (para pairs trading)"""
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

def calculate_half_life(spread):
    """Calcula half-life del mean reversion"""
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    spread_lag = spread_lag.dropna()
    spread_diff = spread_diff.dropna()
    
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
# NUEVAS FUNCIONES ESTADÍSTICAS AVANZADAS
# =============================================================================

def calculate_kelly_criterion(returns, rf_rate=0.02):
    """
    Kelly Criterion: tamaño óptimo de posición
    f* = (p*b - q) / b
    donde p = prob ganar, q = prob perder, b = win/loss ratio
    """
    winning_returns = returns[returns > 0]
    losing_returns = returns[returns < 0]
    
    if len(winning_returns) == 0 or len(losing_returns) == 0:
        return np.nan
    
    win_rate = len(winning_returns) / len(returns)
    loss_rate = 1 - win_rate
    avg_win = winning_returns.mean()
    avg_loss = abs(losing_returns.mean())
    win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    
    kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio if win_loss_ratio != 0 else 0
    
    return {
        'kelly_pct': kelly * 100,
        'win_rate': win_rate * 100,
        'win_loss_ratio': win_loss_ratio,
        'avg_win': avg_win * 100,
        'avg_loss': avg_loss * 100
    }

def calculate_profit_factor(returns):
    """
    Profit Factor: suma de ganancias / suma de pérdidas
    >1: estrategia rentable
    """
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0
    
    return gross_profit / gross_loss

def calculate_win_rate(returns):
    """Calcula el porcentaje de operaciones ganadoras"""
    if len(returns) == 0:
        return 0
    return (returns > 0).sum() / len(returns) * 100

def calculate_expectancy(returns):
    """
    Expectancy: ganancia esperada por operación
    E = (Win% * AvgWin) - (Loss% * AvgLoss)
    """
    winning_returns = returns[returns > 0]
    losing_returns = returns[returns < 0]
    
    if len(returns) == 0:
        return 0
    
    win_rate = len(winning_returns) / len(returns)
    loss_rate = 1 - win_rate
    avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
    avg_loss = abs(losing_returns.mean()) if len(losing_returns) > 0 else 0
    
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    return expectancy * 100

def calculate_mae_mfe(prices, trades_df=None):
    """
    Maximum Adverse Excursion (MAE) y Maximum Favorable Excursion (MFE)
    Útil para optimizar stops y targets
    """
    returns = calculate_returns(prices)
    cumulative = (1 + returns).cumprod()
    
    # Calcular excursiones desde cada punto
    mae_series = []
    mfe_series = []
    
    for i in range(len(cumulative)):
        if i == 0:
            mae_series.append(0)
            mfe_series.append(0)
            continue
        
        future_prices = cumulative[i:]
        entry_price = cumulative.iloc[i]
        
        mae = ((future_prices.min() - entry_price) / entry_price) * 100
        mfe = ((future_prices.max() - entry_price) / entry_price) * 100
        
        mae_series.append(mae)
        mfe_series.append(mfe)
    
    return pd.DataFrame({
        'MAE': mae_series,
        'MFE': mfe_series
    }, index=prices.index)

def calculate_information_coefficient(predictions, actual_returns):
    """
    Information Coefficient: correlación entre predicciones y retornos reales
    Mide la habilidad predictiva
    """
    try:
        ic = predictions.corr(actual_returns)
        return ic
    except:
        return np.nan

def perform_pca_analysis(returns_df, n_components=3):
    """
    Principal Component Analysis para identificar factores de riesgo comunes
    """
    try:
        # Normalizar datos
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns_df.dropna())
        
        # PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(returns_scaled)
        
        # Crear DataFrame con componentes
        pca_df = pd.DataFrame(
            components,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=returns_df.dropna().index
        )
        
        return {
            'components': pca_df,
            'explained_variance': pca.explained_variance_ratio_,
            'loadings': pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=returns_df.columns
            )
        }
    except:
        return None

def detect_regime_clustering(returns_df, n_regimes=3):
    """
    Detecta regímenes de mercado usando K-Means clustering
    """
    try:
        # Características para clustering
        features = pd.DataFrame({
            'returns': returns_df.mean(axis=1),
            'volatility': returns_df.std(axis=1),
            'correlation': returns_df.corr().mean().mean()
        }).dropna()
        
        # K-Means
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regimes = kmeans.fit_predict(features)
        
        features['regime'] = regimes
        
        return features
    except:
        return None

def calculate_dynamic_hedge_ratio(prices1, prices2, window=60):
    """
    Calcula hedge ratio dinámico (rolling) para pairs trading
    """
    hedge_ratios = []
    dates = []
    
    for i in range(window, len(prices1)):
        p1_window = prices1.iloc[i-window:i]
        p2_window = prices2.iloc[i-window:i]
        
        # Regresión lineal rolling
        hedge_ratio = np.polyfit(p2_window, p1_window, 1)[0]
        hedge_ratios.append(hedge_ratio)
        dates.append(prices1.index[i])
    
    return pd.Series(hedge_ratios, index=dates)

def calculate_conditional_correlation(returns1, returns2, condition='positive'):
    """
    Correlación condicional: correlación solo durante mercados alcistas o bajistas
    """
    if condition == 'positive':
        # Correlación cuando ambos retornos son positivos
        mask = (returns1 > 0) & (returns2 > 0)
    elif condition == 'negative':
        # Correlación cuando ambos retornos son negativos
        mask = (returns1 < 0) & (returns2 < 0)
    elif condition == 'crisis':
        # Correlación durante alta volatilidad (crisis)
        vol_threshold = returns1.std() * 2
        mask = (abs(returns1) > vol_threshold) | (abs(returns2) > vol_threshold)
    else:
        mask = pd.Series(True, index=returns1.index)
    
    conditional_returns1 = returns1[mask]
    conditional_returns2 = returns2[mask]
    
    try:
        corr = conditional_returns1.corr(conditional_returns2)
        return corr
    except:
        return np.nan

def calculate_rolling_skew_kurt(returns, window=252):
    """Calcula rolling skewness y kurtosis juntos"""
    return pd.DataFrame({
        'skewness': calculate_skewness(returns, window),
        'kurtosis': calculate_kurtosis(returns, window)
    })

def calculate_garch_volatility(returns, p=1, q=1):
    """
    Simula modelo GARCH(p,q) para volatilidad
    Nota: implementación simplificada sin librería ARCH
    """
    # Implementación simplificada de GARCH(1,1)
    returns_clean = returns.dropna()
    
    # Parámetros iniciales
    omega = 0.01
    alpha = 0.1
    beta = 0.85
    
    # Inicializar volatilidad
    volatility = [returns_clean.var()]
    
    for i in range(1, len(returns_clean)):
        vol = omega + alpha * (returns_clean.iloc[i-1]**2) + beta * volatility[-1]
        volatility.append(vol)
    
    return pd.Series(np.sqrt(volatility) * np.sqrt(252), index=returns_clean.index)

def calculate_distance_correlation(prices1, prices2, method='euclidean'):
    """
    Calcula distancia entre dos series de precios
    Útil para encontrar pares similares
    """
    # Normalizar precios
    norm1 = (prices1 - prices1.mean()) / prices1.std()
    norm2 = (prices2 - prices2.mean()) / prices2.std()
    
    if method == 'euclidean':
        distance = np.sqrt(((norm1 - norm2)**2).sum())
    elif method == 'manhattan':
        distance = abs(norm1 - norm2).sum()
    elif method == 'cosine':
        dot_product = (norm1 * norm2).sum()
        magnitude = np.sqrt((norm1**2).sum() * (norm2**2).sum())
        distance = 1 - (dot_product / magnitude) if magnitude != 0 else 1
    else:
        distance = np.nan
    
    return distance

def calculate_appraisal_ratio(returns_asset, returns_benchmark, window=252):
    """
    Appraisal Ratio = Alpha / Tracking Error Residual
    Similar a Information Ratio pero usando alpha específico
    """
    alpha = calculate_alpha(returns_asset, returns_benchmark, window=window)
    active_return = returns_asset - returns_benchmark
    tracking_error = active_return.rolling(window).std() * np.sqrt(252)
    
    return alpha / tracking_error

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

st.title("📊 Advanced Correlation & Trading Analyzer Pro")
st.markdown("🚀 Análisis avanzado de correlaciones, trading stats y análisis multi-activo | 120+ Activos Globales")

# Sidebar - Configuración
st.sidebar.header("⚙️ Configuración")

# Filtrar activos por categoría
st.sidebar.subheader("Filtrar por Categoría")
categories = list(set([ASSETS[k]['category'] for k in ASSETS.keys()]))
categories.sort()
selected_categories = st.sidebar.multiselect(
    "Categorías",
    options=categories,
    default=['US Equity', 'FX Major', 'Precious Metals', 'Crypto']
)

# Filtrar activos disponibles
available_assets = [k for k in ASSETS.keys() if ASSETS[k]['category'] in selected_categories]

# Selección de activos
st.sidebar.subheader("Selección de Activos")
default_assets = ['sp500', 'gold', 'btc', 'dxy', 'vix'] if len(available_assets) >= 5 else available_assets[:5]

selected_assets = st.sidebar.multiselect(
    "Selecciona activos (mín. 2)",
    options=available_assets,
    default=[a for a in default_assets if a in available_assets],
    format_func=lambda x: f"{ASSETS[x]['label']} ({ASSETS[x]['category']})"
)

if len(selected_assets) < 2:
    st.warning("⚠️ Selecciona al menos 2 activos para continuar")
    st.stop()

# Mostrar número de activos seleccionados
st.sidebar.info(f"✅ {len(selected_assets)} activos seleccionados")

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

# Delay de descarga
download_delay = st.sidebar.slider("Delay entre descargas (segundos)", 1, 30, 10, 1)

# Botón de descarga
if st.sidebar.button("🔄 Actualizar Datos", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Descargar datos con delay configurable
with st.spinner(f"Descargando {len(selected_assets)} activos... (delay: {download_delay}s por activo)"):
    asset_data = download_selected_assets(selected_assets, delay=download_delay)

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

st.success(f"✅ Datos cargados: {len(df_prices)} días | {df_prices.index[0].date()} → {df_prices.index[-1].date()}")

# =============================================================================
# TABS PRINCIPALES
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Análisis de Pares", 
    "🔥 Heatmap & Clustering", 
    "📊 Estadísticas Básicas",
    "⚡ Métricas de Riesgo",
    "🎯 Pairs Trading Avanzado",
    "📉 Análisis Técnico",
    "🧮 Trading Stats Avanzadas"
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
    
    # Análisis condicional de correlación
    st.subheader("🔍 Correlación Condicional")
    
    returns1 = calculate_returns(df_prices[asset1])
    returns2 = calculate_returns(df_prices[asset2])
    
    col1, col2, col3 = st.columns(3)
    
    corr_positive = calculate_conditional_correlation(returns1, returns2, 'positive')
    corr_negative = calculate_conditional_correlation(returns1, returns2, 'negative')
    corr_crisis = calculate_conditional_correlation(returns1, returns2, 'crisis')
    
    col1.metric("Correlación (Alcista)", f"{corr_positive:.4f}")
    col2.metric("Correlación (Bajista)", f"{corr_negative:.4f}")
    col3.metric("Correlación (Crisis)", f"{corr_crisis:.4f}")

with tab2:
    st.subheader("🔥 Matriz de Correlaciones")
    st.plotly_chart(
        plot_correlation_heatmap(df_prices, selected_assets),
        use_container_width=True
    )
    
    # Tabla de correlaciones
    st.subheader("📋 Tabla de Correlaciones")
    corr_matrix = df_prices[selected_assets].corr()
    
    def color_correlation(val):
        if val >= 0.7:
            color = '#10b981'
        elif val >= 0.3:
            color = '#84cc16'
        elif val > -0.3:
            color = '#6b7280'
        elif val > -0.7:
            color = '#f59e0b'
        else:
            color = '#ef4444'
        return f'background-color: {color}; color: white'
    
    styled_df = corr_matrix.style.applymap(color_correlation).format("{:.2f}")
    st.dataframe(styled_df, use_container_width=True)
    
    # Detectar regímenes (si hay suficientes datos)
    if len(selected_assets) >= 3:
        st.subheader("🎯 Detección de Regímenes de Mercado")
        returns_df = df_prices[selected_assets].pct_change().dropna()
        regimes_df = detect_regime_clustering(returns_df, n_regimes=3)
        
        if regimes_df is not None:
            fig = go.Figure()
            
            for regime in regimes_df['regime'].unique():
                regime_data = regimes_df[regimes_df['regime'] == regime]
                fig.add_trace(go.Scatter(
                    x=regime_data.index,
                    y=regime_data['returns'],
                    mode='markers',
                    name=f'Régimen {regime}',
                    marker=dict(size=5)
                ))
            
            fig.update_layout(
                title='Regímenes de Mercado (K-Means Clustering)',
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # PCA Analysis
    if len(selected_assets) >= 3:
        st.subheader("📊 Principal Component Analysis (PCA)")
        returns_df = df_prices[selected_assets].pct_change().dropna()
        pca_result = perform_pca_analysis(returns_df, n_components=min(3, len(selected_assets)))
        
        if pca_result is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("PC1 Varianza Explicada", f"{pca_result['explained_variance'][0]*100:.1f}%")
            with col2:
                st.metric("PC2 Varianza Explicada", f"{pca_result['explained_variance'][1]*100:.1f}%")
            with col3:
                if len(pca_result['explained_variance']) > 2:
                    st.metric("PC3 Varianza Explicada", f"{pca_result['explained_variance'][2]*100:.1f}%")
            
            st.markdown("**Loadings (Contribución de cada activo a los componentes principales):**")
            st.dataframe(pca_result['loadings'].style.background_gradient(cmap='RdYlGn', axis=0))

with tab3:
    st.subheader("📊 Estadísticas de Correlación")
    
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
    st.subheader("📈 Distribución de Correlaciones")
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

with tab4:
    st.subheader("⚡ Métricas de Riesgo y Performance")
    
    selected_asset = st.selectbox(
        "Selecciona activo para análisis de riesgo",
        options=selected_assets,
        format_func=lambda x: ASSETS[x]['label'],
        key='risk_asset'
    )
    
    returns = calculate_returns(df_prices[selected_asset])
    
    benchmark_asset = st.selectbox(
        "Benchmark (para Beta/Alpha)",
        options=[a for a in selected_assets if a != selected_asset],
        format_func=lambda x: ASSETS[x]['label'],
        key='benchmark'
    )
    
    returns_benchmark = calculate_returns(df_prices[benchmark_asset])
    
    # Métricas actuales
    st.markdown("### 📊 Métricas Actuales (últimos 252 días)")
    
    current_vol = calculate_volatility(returns, window=252).iloc[-1]
    current_sharpe = calculate_sharpe_ratio(returns, window=252).iloc[-1]
    current_sortino = calculate_sortino_ratio(returns, window=252).iloc[-1]
    current_treynor = calculate_treynor_ratio(returns, returns_benchmark, window=252).iloc[-1]
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
        st.metric("Treynor Ratio", f"{current_treynor:.3f}")
    
    with col3:
        st.metric("Max Drawdown", f"{max_dd:.2f}%")
        st.metric("VaR 95% (diario)", f"{var_95:.2f}%")
    
    with col4:
        st.metric("Beta", f"{current_beta:.3f}")
        st.metric("Alpha (anual)", f"{current_alpha:.2f}%")
    
    # Capture Ratios
    st.markdown("### 📈 Capture Ratios")
    upside = calculate_upside_capture(returns, returns_benchmark, window=252).iloc[-1]
    downside = calculate_downside_capture(returns, returns_benchmark, window=252).iloc[-1]
    
    col1, col2 = st.columns(2)
    col1.metric("Upside Capture", f"{upside:.1f}%")
    col2.metric("Downside Capture", f"{downside:.1f}%")
    
    # Gráficos
    st.plotly_chart(plot_risk_metrics(df_prices, selected_asset, returns), use_container_width=True)
    st.plotly_chart(plot_performance_ratios(returns, returns_benchmark), use_container_width=True)
    st.plotly_chart(plot_distribution_analysis(returns), use_container_width=True)

with tab5:
    st.subheader("🎯 Pairs Trading Avanzado & Mean Reversion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pair_asset1 = st.selectbox(
            "Activo 1 (Pairs)",
            options=selected_assets,
            format_func=lambda x: ASSETS[x]['label'],
            key='pair1'
        )
    
    with col2:
        pair_asset2 = st.selectbox(
            "Activo 2 (Pairs)",
            options=[a for a in selected_assets if a != pair_asset1],
            format_func=lambda x: ASSETS[x]['label'],
            key='pair2'
        )
    
    prices1 = df_prices[pair_asset1]
    prices2 = df_prices[pair_asset2]
    
    # Cointegración
    st.markdown("### 🔬 Tests de Cointegración")
    coint_test = test_cointegration(prices1, prices2)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cointegración (Engle-Granger)", 
                 "✅ SÍ" if coint_test['cointegrated'] else "❌ NO")
        st.caption(f"p-value: {coint_test['pvalue']:.4f}")
    
    with col2:
        st.metric("Score", f"{coint_test['score']:.4f}")
    
    # Causalidad de Granger
    returns1 = calculate_returns(prices1)
    returns2 = calculate_returns(prices2)
    
    granger_1to2 = granger_causality_test(returns1, returns2, max_lag=5)
    granger_2to1 = granger_causality_test(returns2, returns1, max_lag=5)
    
    with col3:
        if granger_1to2['causes'] or granger_2to1['causes']:
            st.metric("Granger Causality", "✅ DETECTADA")
            if granger_1to2['causes']:
                st.caption(f"{ASSETS[pair_asset1]['label']} → {ASSETS[pair_asset2]['label']}")
            if granger_2to1['causes']:
                st.caption(f"{ASSETS[pair_asset2]['label']} → {ASSETS[pair_asset1]['label']}")
        else:
            st.metric("Granger Causality", "❌ NO DETECTADA")
    
    # Análisis del spread
    st.markdown("### 📊 Análisis del Spread")
    spread, hedge_ratio = calculate_spread(prices1, prices2)
    zscore = calculate_zscore(spread, window=30)
    half_life = calculate_half_life(spread)
    
    # Hedge ratio dinámico
    dynamic_hr = calculate_dynamic_hedge_ratio(prices1, prices2, window=60)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Hedge Ratio", f"{hedge_ratio:.4f}")
    
    with col2:
        st.metric("Z-Score Actual", f"{zscore.iloc[-1]:.2f}")
        if abs(zscore.iloc[-1]) > 2:
            st.caption("🔴 Señal de trading")
        else:
            st.caption("🟢 Sin señal")
    
    with col3:
        st.metric("Half-Life", f"{half_life:.1f} días" if not np.isnan(half_life) else "N/A")
    
    with col4:
        st.metric("HR Dinámico", f"{dynamic_hr.iloc[-1]:.4f}" if len(dynamic_hr) > 0 else "N/A")
    
    # Gráfico del spread
    st.plotly_chart(plot_spread_analysis(prices1, prices2, 
                                        ASSETS[pair_asset1]['label'],
                                        ASSETS[pair_asset2]['label']), 
                   use_container_width=True)
    
    # Tests estadísticos
    st.markdown("### 📈 Tests Estadísticos")
    adf_spread = adf_test(spread)
    hurst_spread = calculate_hurst_exponent(spread.dropna())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ADF Test", "✅ Estacionario" if adf_spread['stationary'] else "❌ No estacionario")
        st.caption(f"p-value: {adf_spread['pvalue']:.4f}")
    
    with col2:
        st.metric("Hurst Exponent", f"{hurst_spread:.3f}")
        if hurst_spread < 0.4:
            st.caption("🔄 Mean Reverting")
        elif hurst_spread < 0.6:
            st.caption("🎲 Random Walk")
        else:
            st.caption("📈 Trending")
    
    with col3:
        distance = calculate_distance_correlation(prices1, prices2, method='euclidean')
        st.metric("Distancia Euclidiana", f"{distance:.2f}")
        st.caption("Menor = Más similares")

with tab6:
    st.subheader("📉 Análisis Técnico")
    
    tech_asset = st.selectbox(
        "Selecciona activo",
        options=selected_assets,
        format_func=lambda x: ASSETS[x]['label'],
        key='tech_asset'
    )
    
    tech_prices = df_prices[tech_asset]
    
    st.plotly_chart(plot_technical_indicators(tech_prices), use_container_width=True)
    
    # Métricas actuales
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
    
    with col3:
        macd_signal = "🟢 Alcista" if macd_data['histogram'].iloc[-1] > 0 else "🔴 Bajista"
        st.metric("Señal MACD", macd_signal)

with tab7:
    st.subheader("🧮 Trading Statistics Avanzadas")
    
    stats_asset = st.selectbox(
        "Selecciona activo",
        options=selected_assets,
        format_func=lambda x: ASSETS[x]['label'],
        key='stats_asset'
    )
    
    stats_returns = calculate_returns(df_prices[stats_asset])
    
    # Kelly Criterion
    st.markdown("### 💰 Kelly Criterion & Trading Metrics")
    kelly = calculate_kelly_criterion(stats_returns)
    
    if isinstance(kelly, dict):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Kelly %", f"{kelly['kelly_pct']:.2f}%")
            st.caption("Tamaño óptimo de posición")
        
        with col2:
            st.metric("Win Rate", f"{kelly['win_rate']:.1f}%")
        
        with col3:
            st.metric("Win/Loss Ratio", f"{kelly['win_loss_ratio']:.2f}")
        
        with col4:
            st.metric("Avg Win", f"{kelly['avg_win']:.2f}%")
    
    # Profit Factor & Expectancy
    st.markdown("### 📊 Performance Metrics")
    
    profit_factor = calculate_profit_factor(stats_returns)
    win_rate = calculate_win_rate(stats_returns)
    expectancy = calculate_expectancy(stats_returns)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Profit Factor", f"{profit_factor:.2f}")
        st.caption(">1: Rentable")
    
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col3:
        st.metric("Expectancy", f"{expectancy:.3f}%")
        st.caption("Ganancia esperada por trade")
    
    # MAE/MFE Analysis
    st.markdown("### 📉 MAE/MFE Analysis")
    mae_mfe = calculate_mae_mfe(df_prices[stats_asset])
    
    if not mae_mfe.empty:
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Maximum Adverse Excursion', 'Maximum Favorable Excursion'))
        
        fig.add_trace(go.Scatter(x=mae_mfe.index, y=mae_mfe['MAE'], 
                                name='MAE', line=dict(color='#ef4444')), row=1, col=1)
        fig.add_trace(go.Scatter(x=mae_mfe.index, y=mae_mfe['MFE'], 
                                name='MFE', line=dict(color='#10b981')), row=2, col=1)
        
        fig.update_layout(height=600, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Guía de Métricas")
st.sidebar.markdown("""
**Correlaciones:**
- > 0.5: Fuerte positiva
- < -0.5: Fuerte negativa

**Performance Ratios:**
- Sharpe > 2: Excelente
- Sortino > 2: Muy bueno

**Kelly Criterion:**
- Tamaño óptimo de posición
- Usar 25-50% del Kelly

**Profit Factor:**
- > 1.5: Buena estrategia
- > 2.0: Excelente estrategia
""")

st.sidebar.markdown("---")
st.sidebar.info(f"💡 {len(ASSETS)} activos disponibles | Datos actualizados cada hora")
