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

def calculate_lead_lag_correlation(series1, series2, max_lag=10):
    """
    Calcula correlación con diferentes lags (retrasos)
    Útil para detectar relaciones de liderazgo entre activos
    """
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        if lag < 0:
            # series2 lidera a series1
            corr = series1.iloc[-lag:].corr(series2.iloc[:lag])
        elif lag > 0:
            # series1 lidera a series2
            corr = series1.iloc[:-lag].corr(series2.iloc[lag:])
        else:
            # Sin lag
            corr = series1.corr(series2)
        
        correlations.append(corr)
    
    return pd.DataFrame({
        'lag': lags,
        'correlation': correlations
    })

def calculate_rolling_correlation_multi_window(df, asset1, asset2, windows=[10, 30, 60, 120]):
    """
    Calcula correlación rolling con múltiples ventanas
    Útil para ver diferentes timeframes simultáneamente
    """
    returns1 = calculate_returns(df[asset1])
    returns2 = calculate_returns(df[asset2])
    
    result = pd.DataFrame(index=df.index)
    
    for window in windows:
        corr = returns1.rolling(window).corr(returns2)
        result[f'corr_{window}d'] = corr
    
    return result

def calculate_correlation_stability(corr_series, window=60):
    """
    Mide la estabilidad de la correlación
    Una baja desviación estándar indica correlación estable
    """
    rolling_std = corr_series.rolling(window).std()
    rolling_mean = corr_series.rolling(window).mean()
    
    # Coefficient of variation
    cv = (rolling_std / rolling_mean.abs()).replace([np.inf, -np.inf], np.nan)
    
    return pd.DataFrame({
        'corr_std': rolling_std,
        'corr_mean': rolling_mean,
        'stability_cv': cv
    })

def find_correlation_breakpoints(corr_series, threshold=0.3):
    """
    Detecta puntos donde la correlación cambia significativamente
    Identifica cambios de régimen
    """
    corr_diff = corr_series.diff().abs()
    breakpoints = corr_diff[corr_diff > threshold]
    
    return breakpoints

def calculate_optimal_hedge_ratio_methods(prices1, prices2):
    """
    Calcula hedge ratio con múltiples métodos:
    - OLS (Ordinary Least Squares)
    - TLS (Total Least Squares)
    - Variance Minimization
    """
    prices1_clean = prices1.dropna()
    prices2_clean = prices2.dropna()
    
    common_idx = prices1_clean.index.intersection(prices2_clean.index)
    p1 = prices1_clean.loc[common_idx]
    p2 = prices2_clean.loc[common_idx]
    
    # OLS
    ols_hr = np.polyfit(p2, p1, 1)[0]
    
    # Variance Minimization
    returns1 = np.diff(np.log(p1))
    returns2 = np.diff(np.log(p2))
    var_hr = np.cov(returns1, returns2)[0, 1] / np.var(returns2)
    
    # Correlation-based
    corr_hr = np.std(p1) / np.std(p2) * np.corrcoef(p1, p2)[0, 1]
    
    return {
        'ols': ols_hr,
        'variance_min': var_hr,
        'correlation': corr_hr
    }

def calculate_ou_parameters(spread):
    """
    Estima parámetros del proceso Ornstein-Uhlenbeck
    - theta: velocidad de reversión a la media
    - mu: nivel de equilibrio
    - sigma: volatilidad
    """
    spread_clean = spread.dropna()
    
    if len(spread_clean) < 2:
        return {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan, 'half_life': np.nan}
    
    spread_lag = spread_clean.shift(1).dropna()
    spread_diff = spread_clean.diff().dropna()
    
    # Alinear índices
    common_idx = spread_lag.index.intersection(spread_diff.index)
    spread_lag = spread_lag.loc[common_idx]
    spread_diff = spread_diff.loc[common_idx]
    
    # Regresión: ds = theta * (mu - s) * dt + sigma * dW
    # Simplificado: ds = a + b*s
    if len(spread_lag) < 2:
        return {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan, 'half_life': np.nan}
    
    coeffs = np.polyfit(spread_lag, spread_diff, 1)
    theta = -coeffs[0]
    mu = -coeffs[1] / coeffs[0] if coeffs[0] != 0 else np.nan
    sigma = np.std(spread_diff)
    half_life = np.log(2) / theta if theta > 0 else np.nan
    
    return {
        'theta': theta,
        'mu': mu,
        'sigma': sigma,
        'half_life': half_life
    }

def calculate_cointegration_strength(prices1, prices2, window=252):
    """
    Mide la fuerza de la cointegración en rolling window
    """
    coint_scores = []
    dates = []
    
    for i in range(window, len(prices1)):
        p1_window = prices1.iloc[i-window:i]
        p2_window = prices2.iloc[i-window:i]
        
        try:
            score, pvalue, _ = coint(p1_window, p2_window)
            coint_scores.append(-score)  # Más negativo = más cointegrado
            dates.append(prices1.index[i])
        except:
            coint_scores.append(np.nan)
            dates.append(prices1.index[i])
    
    return pd.Series(coint_scores, index=dates)

def detect_pairs_by_distance(df, method='euclidean', top_n=10):
    """
    Encuentra los mejores pares basándose en distancia
    Retorna los N pares más cercanos
    """
    assets = df.columns
    distances = []
    
    for i, asset1 in enumerate(assets):
        for asset2 in assets[i+1:]:
            dist = calculate_distance_correlation(df[asset1], df[asset2], method)
            distances.append({
                'asset1': asset1,
                'asset2': asset2,
                'distance': dist
            })
    
    distances_df = pd.DataFrame(distances).sort_values('distance')
    return distances_df.head(top_n)

def calculate_spread_quality_score(spread, zscore):
    """
    Calcula un score de calidad del spread para trading
    Combina: estacionariedad, mean reversion, volatilidad
    """
    adf_result = adf_test(spread)
    hurst = calculate_hurst_exponent(spread.dropna())
    
    # Crossings de cero del zscore (señal de mean reversion)
    zero_crossings = ((zscore.shift(1) * zscore) < 0).sum()
    crossing_rate = zero_crossings / len(zscore)
    
    # Volatilidad del spread
    spread_vol = spread.std()
    
    # Score compuesto (0-100)
    stationarity_score = 30 if adf_result['stationary'] else 0
    mean_reversion_score = max(0, 30 * (1 - abs(hurst - 0.5) * 2))  # Mejor cerca de 0.5 o menos
    crossing_score = min(30, crossing_rate * 1000)
    volatility_score = max(0, 10 - spread_vol * 10)  # Penaliza alta volatilidad
    
    total_score = stationarity_score + mean_reversion_score + crossing_score + volatility_score
    
    return {
        'total_score': total_score,
        'stationarity_score': stationarity_score,
        'mean_reversion_score': mean_reversion_score,
        'crossing_score': crossing_score,
        'volatility_score': volatility_score,
        'zero_crossings': zero_crossings,
        'crossing_rate': crossing_rate
    }

def calculate_rolling_beta_pairs(prices1, prices2, window=60):
    """
    Calcula beta rolling entre dos activos
    Útil para pairs trading dinámico
    """
    returns1 = calculate_returns(prices1)
    returns2 = calculate_returns(prices2)
    
    rolling_beta = returns1.rolling(window).cov(returns2) / returns2.rolling(window).var()
    
    return rolling_beta

def calculate_correlation_percentile(corr_series, percentiles=[10, 25, 50, 75, 90]):
    """
    Calcula percentiles de la distribución de correlación
    Útil para entender el rango histórico
    """
    result = {}
    for p in percentiles:
        result[f'p{p}'] = np.percentile(corr_series.dropna(), p)
    
    return result

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

def calculate_time_varying_correlation(returns1, returns2, method='ewm', span=30):
    """
    Correlación dinámica usando diferentes métodos
    - ewm: Exponentially Weighted Moving Average
    - dcc: Dynamic Conditional Correlation (simplificado)
    """
    if method == 'ewm':
        # Correlación EWMA
        mean1 = returns1.ewm(span=span).mean()
        mean2 = returns2.ewm(span=span).mean()
        
        cov = ((returns1 - mean1) * (returns2 - mean2)).ewm(span=span).mean()
        std1 = (returns1 - mean1).pow(2).ewm(span=span).mean().pow(0.5)
        std2 = (returns2 - mean2).pow(2).ewm(span=span).mean().pow(0.5)
        
        corr = cov / (std1 * std2)
        return corr
    else:
        # Rolling standard
        return returns1.rolling(span).corr(returns2)

def calculate_partial_correlation(df, asset1, asset2, control_assets):
    """
    Correlación parcial: correlación entre asset1 y asset2
    controlando por otros activos
    Útil para eliminar efectos espurios
    """
    from sklearn.linear_model import LinearRegression
    
    returns = df.pct_change().dropna()
    
    # Regresión de asset1 sobre control_assets
    X_control = returns[control_assets].values
    y1 = returns[asset1].values
    y2 = returns[asset2].values
    
    # Residuales
    model1 = LinearRegression().fit(X_control, y1)
    model2 = LinearRegression().fit(X_control, y2)
    
    residuals1 = y1 - model1.predict(X_control)
    residuals2 = y2 - model2.predict(X_control)
    
    # Correlación de residuales
    partial_corr = np.corrcoef(residuals1, residuals2)[0, 1]
    
    return partial_corr

def rank_correlation_spearman(prices1, prices2, window=60):
    """
    Correlación de Spearman (por rangos) rolling
    Más robusta a outliers que Pearson
    """
    corr_spearman = []
    dates = []
    
    for i in range(window, len(prices1)):
        p1_window = prices1.iloc[i-window:i]
        p2_window = prices2.iloc[i-window:i]
        
        rho, _ = stats.spearmanr(p1_window, p2_window)
        corr_spearman.append(rho)
        dates.append(prices1.index[i])
    
    return pd.Series(corr_spearman, index=dates)

def calculate_tail_correlation(returns1, returns2, quantile=0.05):
    """
    Correlación en las colas de la distribución
    Útil para entender comportamiento en eventos extremos
    """
    # Lower tail (pérdidas extremas)
    threshold_lower = returns1.quantile(quantile)
    mask_lower = (returns1 <= threshold_lower) | (returns2 <= returns2.quantile(quantile))
    
    # Upper tail (ganancias extremas)
    threshold_upper = returns1.quantile(1 - quantile)
    mask_upper = (returns1 >= threshold_upper) | (returns2 >= returns2.quantile(1 - quantile))
    
    corr_lower = returns1[mask_lower].corr(returns2[mask_lower])
    corr_upper = returns1[mask_upper].corr(returns2[mask_upper])
    corr_normal = returns1.corr(returns2)
    
    return {
        'lower_tail': corr_lower,
        'upper_tail': corr_upper,
        'normal': corr_normal
    }

def calculate_correlation_clustering_coefficient(corr_matrix, threshold=0.5):
    """
    Calcula coeficiente de clustering en red de correlaciones
    Identifica grupos de activos altamente correlacionados
    """
    # Crear matriz de adyacencia
    adj_matrix = (corr_matrix.abs() > threshold).astype(int)
    np.fill_diagonal(adj_matrix.values, 0)
    
    # Calcular clustering coefficient para cada nodo
    n = len(adj_matrix)
    clustering_coeffs = {}
    
    for i, asset in enumerate(adj_matrix.index):
        neighbors = adj_matrix.iloc[i][adj_matrix.iloc[i] == 1].index
        if len(neighbors) < 2:
            clustering_coeffs[asset] = 0
            continue
        
        # Contar enlaces entre vecinos
        links_between_neighbors = 0
        for n1 in neighbors:
            for n2 in neighbors:
                if n1 != n2 and adj_matrix.loc[n1, n2] == 1:
                    links_between_neighbors += 1
        
        # Clustering coefficient
        possible_links = len(neighbors) * (len(neighbors) - 1)
        clustering_coeffs[asset] = links_between_neighbors / possible_links if possible_links > 0 else 0
    
    return clustering_coeffs

def detect_correlation_regime_changes(corr_series, window=30, threshold=0.3):
    """
    Detecta cambios de régimen en correlación usando ventana móvil
    Retorna fechas donde ocurren cambios significativos
    """
    rolling_mean = corr_series.rolling(window).mean()
    rolling_std = corr_series.rolling(window).std()
    
    # Z-score de la correlación
    z_score = (corr_series - rolling_mean) / rolling_std
    
    # Detectar cambios significativos
    regime_changes = z_score[z_score.abs() > threshold]
    
    return regime_changes

def calculate_information_share(prices1, prices2, window=60):
    """
    Information Share: qué activo contribuye más al price discovery
    Basado en Hasbrouck (1995)
    """
    returns1 = calculate_returns(prices1)
    returns2 = calculate_returns(prices2)
    
    info_share = []
    dates = []
    
    for i in range(window, len(returns1)):
        r1_window = returns1.iloc[i-window:i]
        r2_window = returns2.iloc[i-window:i]
        
        # Varianzas y covarianza
        var1 = r1_window.var()
        var2 = r2_window.var()
        cov12 = r1_window.cov(r2_window)
        
        # Information share aproximado
        if var1 + var2 != 0:
            is1 = (var1 + cov12) / (var1 + var2 + 2*cov12)
            info_share.append(is1)
            dates.append(returns1.index[i])
        else:
            info_share.append(np.nan)
            dates.append(returns1.index[i])
    
    return pd.Series(info_share, index=dates)

def calculate_cophenetic_correlation(corr_matrix):
    """
    Correlación cofenética para evaluar calidad del clustering jerárquico
    Valores altos indican buena estructura de clustering
    """
    from scipy.cluster.hierarchy import linkage, cophenet
    from scipy.spatial.distance import pdist
    
    # Convertir correlación a distancia
    distance_matrix = 1 - corr_matrix.abs()
    
    # Clustering jerárquico
    linkage_matrix = linkage(pdist(distance_matrix), method='average')
    
    # Correlación cofenética
    c, _ = cophenet(linkage_matrix, pdist(distance_matrix))
    
    return c

def find_best_inverse_pairs(df, min_negative_correlation=-0.7, max_correlation=-0.3):
    """
    Encuentra pares con fuerte correlación NEGATIVA
    Útil para hedging y diversificación
    """
    assets = df.columns
    candidates = []
    
    for i, asset1 in enumerate(assets):
        for asset2 in assets[i+1:]:
            prices1 = df[asset1].dropna()
            prices2 = df[asset2].dropna()
            
            # Índice común
            common_idx = prices1.index.intersection(prices2.index)
            if len(common_idx) < 252:  # Mínimo 1 año de datos
                continue
            
            p1 = prices1.loc[common_idx]
            p2 = prices2.loc[common_idx]
            
            # Correlación
            correlation = p1.corr(p2)
            
            # Solo pares con correlación negativa
            if correlation > max_correlation or correlation < min_negative_correlation:
                continue
            
            # Tests adicionales
            returns1 = calculate_returns(p1)
            returns2 = calculate_returns(p2)
            
            # Volatilidad
            vol1 = returns1.std() * np.sqrt(252)
            vol2 = returns2.std() * np.sqrt(252)
            vol_ratio = min(vol1, vol2) / max(vol1, vol2)  # Ratio de volatilidades
            
            # Estabilidad de correlación
            rolling_corr = returns1.rolling(60).corr(returns2)
            corr_std = rolling_corr.std()
            
            # Lead-lag
            lead_lag = calculate_lead_lag_correlation(returns1, returns2, max_lag=5)
            max_lag_corr = lead_lag['correlation'].abs().max()
            
            # Score para correlación inversa
            score = 0
            
            # Correlación fuerte y estable
            if correlation < -0.7:
                score += 40
            elif correlation < -0.5:
                score += 25
            else:
                score += 10
            
            # Estabilidad (baja std de correlación)
            if corr_std < 0.1:
                score += 30
            elif corr_std < 0.2:
                score += 20
            else:
                score += 10
            
            # Volatilidades similares (mejor para hedging)
            if vol_ratio > 0.7:
                score += 20
            elif vol_ratio > 0.5:
                score += 10
            
            # Lead-lag fuerte
            if max_lag_corr > 0.5:
                score += 10
            
            candidates.append({
                'asset1': asset1,
                'asset2': asset2,
                'score': score,
                'correlation': correlation,
                'corr_stability': corr_std,
                'vol_ratio': vol_ratio,
                'vol1': vol1,
                'vol2': vol2,
                'max_lag_corr': max_lag_corr
            })
    
    # Manejar caso cuando no hay candidatos
    if len(candidates) == 0:
        return pd.DataFrame(columns=['asset1', 'asset2', 'score', 'correlation', 
                                    'corr_stability', 'vol_ratio', 'vol1', 'vol2', 'max_lag_corr'])
    
    return pd.DataFrame(candidates).sort_values('score', ascending=False)

def calculate_hedge_effectiveness(prices1, prices2, hedge_ratio=1.0):
    """
    Calcula la efectividad del hedge entre dos activos
    Retorna métricas de reducción de volatilidad y riesgo
    """
    returns1 = calculate_returns(prices1)
    returns2 = calculate_returns(prices2)
    
    # Portfolio hedgeado
    hedged_returns = returns1 - hedge_ratio * returns2
    
    # Métricas
    vol_original = returns1.std() * np.sqrt(252)
    vol_hedged = hedged_returns.std() * np.sqrt(252)
    vol_reduction = (1 - vol_hedged / vol_original) * 100
    
    # Drawdown
    dd_original = calculate_max_drawdown(prices1).min() * 100
    
    # Crear precios sintéticos del portfolio hedgeado
    hedged_prices = (1 + hedged_returns).cumprod()
    dd_hedged = calculate_max_drawdown(hedged_prices).min() * 100
    dd_reduction = (1 - abs(dd_hedged) / abs(dd_original)) * 100
    
    # Sharpe ratio
    sharpe_original = (returns1.mean() * 252) / (returns1.std() * np.sqrt(252))
    sharpe_hedged = (hedged_returns.mean() * 252) / (hedged_returns.std() * np.sqrt(252))
    
    return {
        'vol_original': vol_original,
        'vol_hedged': vol_hedged,
        'vol_reduction_pct': vol_reduction,
        'dd_original': dd_original,
        'dd_hedged': dd_hedged,
        'dd_reduction_pct': dd_reduction,
        'sharpe_original': sharpe_original,
        'sharpe_hedged': sharpe_hedged,
        'hedge_ratio': hedge_ratio
    }

def calculate_optimal_hedge_ratio_inverse(prices1, prices2):
    """
    Calcula el hedge ratio óptimo para correlación inversa
    Minimiza la volatilidad del portfolio
    """
    returns1 = calculate_returns(prices1)
    returns2 = calculate_returns(prices2)
    
    # Método de mínima varianza
    cov_matrix = np.cov(returns1, returns2)
    var2 = cov_matrix[1, 1]
    cov12 = cov_matrix[0, 1]
    
    # Hedge ratio óptimo
    optimal_hr = -cov12 / var2  # Negativo para correlación inversa
    
    return optimal_hr

def detect_correlation_regime_inverse(corr_series, threshold=-0.3):
    """
    Detecta cuando la correlación se vuelve suficientemente negativa
    para estrategias de hedging
    """
    inverse_regime = corr_series < threshold
    
    # Detectar inicio y fin de regímenes
    regime_changes = inverse_regime.astype(int).diff()
    regime_starts = regime_changes[regime_changes == 1].index
    regime_ends = regime_changes[regime_changes == -1].index
    
    return {
        'in_inverse_regime': inverse_regime.iloc[-1],
        'current_correlation': corr_series.iloc[-1],
        'regime_starts': regime_starts,
        'regime_ends': regime_ends,
        'pct_time_inverse': inverse_regime.sum() / len(inverse_regime) * 100
    }

def find_best_pairs_comprehensive(df, min_cointegration_pvalue=0.05, 
                                  max_distance=1.0, min_correlation=0.7):
    """
    Encuentra los mejores pares usando múltiples criterios:
    - Cointegración
    - Distancia
    - Correlación
    - Hurst exponent
    """
    assets = df.columns
    candidates = []
    
    for i, asset1 in enumerate(assets):
        for asset2 in assets[i+1:]:
            prices1 = df[asset1].dropna()
            prices2 = df[asset2].dropna()
            
            # Índice común
            common_idx = prices1.index.intersection(prices2.index)
            if len(common_idx) < 252:  # Mínimo 1 año de datos
                continue
            
            p1 = prices1.loc[common_idx]
            p2 = prices2.loc[common_idx]
            
            # Tests
            coint_result = test_cointegration(p1, p2)
            if not coint_result['cointegrated']:
                continue
            
            spread, _ = calculate_spread(p1, p2)
            hurst = calculate_hurst_exponent(spread.dropna())
            
            distance = calculate_distance_correlation(p1, p2, 'euclidean')
            correlation = p1.corr(p2)
            
            # Score compuesto
            score = 0
            if coint_result['pvalue'] < min_cointegration_pvalue:
                score += 30
            if abs(correlation) > min_correlation:
                score += 20
            if hurst < 0.5:  # Mean reverting
                score += 30
            if distance < max_distance:
                score += 20
            
            candidates.append({
                'asset1': asset1,
                'asset2': asset2,
                'score': score,
                'correlation': correlation,
                'cointegration_pvalue': coint_result['pvalue'],
                'hurst': hurst,
                'distance': distance
            })
    
    # Manejar caso cuando no hay candidatos
    if len(candidates) == 0:
        return pd.DataFrame(columns=['asset1', 'asset2', 'score', 'correlation', 
                                    'cointegration_pvalue', 'hurst', 'distance'])
    
    return pd.DataFrame(candidates).sort_values('score', ascending=False)

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

def plot_lead_lag_correlation(lead_lag_df, asset1_name, asset2_name):
    """Visualiza correlación con diferentes lags"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lead_lag_df['lag'],
        y=lead_lag_df['correlation'],
        mode='lines+markers',
        name='Lead-Lag Correlation',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=6)
    ))
    
    # Línea en lag=0
    fig.add_vline(x=0, line_dash="dash", line_color="#fbbf24", 
                  annotation_text="No Lag", annotation_position="top")
    
    # Línea de correlación cero
    fig.add_hline(y=0, line_dash="dot", line_color="#666666")
    
    # Encontrar máximo
    max_corr_idx = lead_lag_df['correlation'].abs().idxmax()
    max_lag = lead_lag_df.loc[max_corr_idx, 'lag']
    max_corr = lead_lag_df.loc[max_corr_idx, 'correlation']
    
    fig.add_annotation(
        x=max_lag, y=max_corr,
        text=f"Max: {max_corr:.3f}<br>Lag: {max_lag}",
        showarrow=True,
        arrowhead=2,
        bgcolor="#10b981" if max_corr > 0 else "#ef4444",
        opacity=0.8
    )
    
    fig.update_layout(
        title=f'Lead-Lag Correlation: {asset1_name} vs {asset2_name}',
        xaxis_title='Lag (días)',
        yaxis_title='Correlación',
        template='plotly_dark',
        height=500
    )
    
    return fig

def plot_multi_window_correlation(corr_multi_df, asset1_name, asset2_name):
    """Visualiza correlación con múltiples ventanas temporales"""
    fig = go.Figure()
    
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
    
    for i, col in enumerate(corr_multi_df.columns):
        window_size = col.replace('corr_', '').replace('d', '')
        fig.add_trace(go.Scatter(
            x=corr_multi_df.index,
            y=corr_multi_df[col],
            mode='lines',
            name=f'{window_size} días',
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#666666")
    
    fig.update_layout(
        title=f'Multi-Window Rolling Correlation: {asset1_name} vs {asset2_name}',
        xaxis_title='Fecha',
        yaxis_title='Correlación',
        template='plotly_dark',
        height=500,
        hovermode='x unified'
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

def plot_ou_process_analysis(spread, ou_params):
    """Visualiza análisis del proceso Ornstein-Uhlenbeck"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Spread (μ={ou_params["mu"]:.4f}, θ={ou_params["theta"]:.4f})',
            'Spread vs Nivel de Equilibrio'
        ),
        vertical_spacing=0.15
    )
    
    # Spread original
    fig.add_trace(go.Scatter(
        x=spread.index,
        y=spread,
        name='Spread',
        line=dict(color='#3b82f6', width=2)
    ), row=1, col=1)
    
    # Nivel de equilibrio
    if not np.isnan(ou_params['mu']):
        fig.add_hline(
            y=ou_params['mu'],
            line_dash="dash",
            line_color="#10b981",
            annotation_text=f"μ = {ou_params['mu']:.4f}",
            row=1, col=1
        )
        
        # Bandas basadas en sigma
        sigma = ou_params['sigma']
        fig.add_hline(y=ou_params['mu'] + sigma, line_dash="dot", 
                     line_color="#ef4444", opacity=0.5, row=1, col=1)
        fig.add_hline(y=ou_params['mu'] - sigma, line_dash="dot", 
                     line_color="#ef4444", opacity=0.5, row=1, col=1)
    
    # Desviación del equilibrio
    if not np.isnan(ou_params['mu']):
        deviation = spread - ou_params['mu']
        fig.add_trace(go.Scatter(
            x=spread.index,
            y=deviation,
            name='Desviación',
            line=dict(color='#ec4899', width=2),
            fill='tozeroy'
        ), row=2, col=1)
        
        fig.add_hline(y=0, line_dash="dash", line_color="#666666", row=2, col=1)
    
    fig.update_layout(height=700, template='plotly_dark')
    fig.update_yaxes(title_text="Spread", row=1, col=1)
    fig.update_yaxes(title_text="Desviación", row=2, col=1)
    
    return fig

def plot_cointegration_strength(coint_strength, asset1_name, asset2_name):
    """Visualiza fuerza de cointegración en el tiempo"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=coint_strength.index,
        y=coint_strength,
        mode='lines',
        name='Cointegration Score',
        line=dict(color='#8b5cf6', width=2),
        fill='tozeroy',
        fillcolor='rgba(139, 92, 246, 0.2)'
    ))
    
    # Línea de threshold (valores más altos = más cointegrado)
    fig.add_hline(y=0, line_dash="dash", line_color="#666666")
    
    fig.update_layout(
        title=f'Rolling Cointegration Strength: {asset1_name} vs {asset2_name}',
        xaxis_title='Fecha',
        yaxis_title='Cointegration Score (más alto = más cointegrado)',
        template='plotly_dark',
        height=400
    )
    
    return fig

def plot_best_pairs_ranking(pairs_df, top_n=15):
    """Visualiza ranking de mejores pares"""
    top_pairs = pairs_df.head(top_n).copy()
    top_pairs['pair_label'] = top_pairs['asset1'] + ' / ' + top_pairs['asset2']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_pairs['pair_label'],
        x=top_pairs['score'],
        orientation='h',
        marker=dict(
            color=top_pairs['score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Score")
        ),
        text=top_pairs['score'].round(1),
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Mejores Pares para Trading',
        xaxis_title='Score de Calidad',
        yaxis_title='Par',
        template='plotly_dark',
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def plot_inverse_pairs_ranking(pairs_df, top_n=15):
    """Visualiza ranking de mejores pares con correlación INVERSA"""
    top_pairs = pairs_df.head(top_n).copy()
    top_pairs['pair_label'] = top_pairs['asset1'] + ' / ' + top_pairs['asset2']
    
    fig = go.Figure()
    
    # Color basado en correlación (más rojo = más negativo)
    colors = ['#ef4444' if corr < -0.7 else '#f97316' if corr < -0.5 else '#fbbf24' 
              for corr in top_pairs['correlation']]
    
    fig.add_trace(go.Bar(
        y=top_pairs['pair_label'],
        x=top_pairs['score'],
        orientation='h',
        marker=dict(color=colors),
        text=top_pairs['score'].round(1),
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.1f}<br>Corr: %{customdata:.3f}<extra></extra>',
        customdata=top_pairs['correlation']
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Mejores Pares con Correlación INVERSA (Hedging)',
        xaxis_title='Score de Calidad',
        yaxis_title='Par',
        template='plotly_dark',
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def plot_hedge_effectiveness(prices1, prices2, hedge_ratio, asset1_name, asset2_name):
    """Visualiza la efectividad del hedge"""
    returns1 = calculate_returns(prices1)
    returns2 = calculate_returns(prices2)
    hedged_returns = returns1 - hedge_ratio * returns2
    
    # Crear precios acumulados
    cumul_original = (1 + returns1).cumprod()
    cumul_hedged = (1 + hedged_returns).cumprod()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Performance: Original vs Hedgeado (HR={hedge_ratio:.3f})',
            'Rolling Volatility (30 días)'
        ),
        vertical_spacing=0.15
    )
    
    # Performance acumulada
    fig.add_trace(go.Scatter(
        x=cumul_original.index,
        y=cumul_original,
        name=f'{asset1_name} Original',
        line=dict(color='#3b82f6', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=cumul_hedged.index,
        y=cumul_hedged,
        name=f'Portfolio Hedgeado',
        line=dict(color='#10b981', width=2)
    ), row=1, col=1)
    
    # Volatilidad rolling
    vol_original = returns1.rolling(30).std() * np.sqrt(252) * 100
    vol_hedged = hedged_returns.rolling(30).std() * np.sqrt(252) * 100
    
    fig.add_trace(go.Scatter(
        x=vol_original.index,
        y=vol_original,
        name='Vol Original',
        line=dict(color='#ef4444', width=2)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=vol_hedged.index,
        y=vol_hedged,
        name='Vol Hedgeada',
        line=dict(color='#10b981', width=2)
    ), row=2, col=1)
    
    fig.update_layout(
        height=700,
        template='plotly_dark',
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Performance (Base 1)", row=1, col=1)
    fig.update_yaxes(title_text="Volatilidad Anualizada (%)", row=2, col=1)
    
    return fig

def plot_correlation_regime_inverse(corr_series, threshold=-0.3):
    """Visualiza regímenes de correlación inversa"""
    fig = go.Figure()
    
    # Correlación
    fig.add_trace(go.Scatter(
        x=corr_series.index,
        y=corr_series,
        mode='lines',
        name='Correlación',
        line=dict(color='#3b82f6', width=2)
    ))
    
    # Threshold de régimen inverso
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"Threshold Inverso ({threshold})",
                  annotation_position="right")
    
    fig.add_hline(y=0, line_dash="dot", line_color="#666666")
    
    # Sombrear área de régimen inverso
    fig.add_hrect(y0=-1, y1=threshold, fillcolor="#ef4444", opacity=0.1, line_width=0,
                  annotation_text="Régimen Inverso", annotation_position="top left")
    
    fig.update_layout(
        title='Detección de Regímenes de Correlación Inversa',
        xaxis_title='Fecha',
        yaxis_title='Correlación',
        template='plotly_dark',
        height=400,
        yaxis=dict(range=[-1, 1])
    )
    
    return fig

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================

st.title("📊 Correlation & Pairs Trading Analyzer Pro")
st.markdown("🔍 **Búsqueda Automática** de Pares Correlacionados e Inversos | Pairs Trading & Hedging | 120+ Activos")

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

# Botón de descarga MANUAL
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if st.sidebar.button("📥 Descargar Datos", type="primary", disabled=st.session_state.data_loaded):
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
    
    # Guardar en session state
    st.session_state.df_prices = df_prices
    st.session_state.data_loaded = True
    st.success(f"✅ Datos cargados: {len(df_prices)} días | {df_prices.index[0].date()} → {df_prices.index[-1].date()}")
    st.rerun()

if st.sidebar.button("🔄 Limpiar y Recargar"):
    st.session_state.data_loaded = False
    if 'df_prices' in st.session_state:
        del st.session_state.df_prices
    st.cache_data.clear()
    st.rerun()

# Verificar si hay datos cargados
if not st.session_state.data_loaded:
    st.info("""
    ### 👋 Bienvenido al Correlation & Pairs Trading Analyzer!
    
    **Para comenzar:**
    1. 📂 **Selecciona categorías** de activos en el sidebar (US Equity, FX, Crypto, etc.)
    2. ✅ **Elige activos** que quieres analizar (mínimo 2, recomendado 10-20 para búsqueda)
    3. ⚙️ **Configura parámetros** (delay, período, ventana de correlación)
    4. 📥 **Presiona 'Descargar Datos'** en el sidebar
    
    **Pestañas disponibles:**
    - 📈 Análisis detallado de cualquier par
    - 🔥 Heatmap & clustering de correlaciones
    - 📊 Estadísticas avanzadas
    - 🎯 Pairs trading (cointegración, spread, z-score)
    - 🔍 **Búsqueda automática de mejores pares** ⭐
    - 🛡️ **Análisis de correlación inversa y hedging** ⭐
    
    💡 **Tip**: Usa categorías relacionadas (ej: US Equity + Sector ETFs) para mejores resultados
    """)
    st.stop()

# Obtener datos del session state
df_prices = st.session_state.df_prices
st.success(f"✅ Datos listos: {len(df_prices)} días | {df_prices.index[0].date()} → {df_prices.index[-1].date()}")

# =============================================================================
# TABS PRINCIPALES
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Análisis de Pares Detallado", 
    "🔥 Heatmap & Clustering", 
    "📊 Estadísticas de Correlación",
    "🎯 Pairs Trading Avanzado",
    "🔍 Búsqueda de Mejores Pares",
    "🛡️ Correlación Inversa & Hedging"
])

with tab1:
    st.subheader("📈 Análisis Detallado de Pares")
    
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
    
    # Calcular correlación básica
    corr_df = calculate_rolling_correlation(df_prices, asset1, asset2, window_size, step_size)
    
    # Métricas básicas
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
    
    # Gráfico de correlación rolling
    st.plotly_chart(
        plot_rolling_correlation(corr_df, ASSETS[asset1]['label'], 
                                ASSETS[asset2]['label'],
                                ASSETS[asset1]['color'], 
                                ASSETS[asset2]['color']),
        use_container_width=True
    )
    
    # Lead-Lag Correlation
    st.markdown("### 🔄 Lead-Lag Correlation Analysis")
    st.caption("Detecta si un activo lidera al otro (útil para predicción y timing)")
    
    returns1 = calculate_returns(df_prices[asset1])
    returns2 = calculate_returns(df_prices[asset2])
    
    lead_lag_df = calculate_lead_lag_correlation(returns1, returns2, max_lag=10)
    st.plotly_chart(plot_lead_lag_correlation(lead_lag_df, ASSETS[asset1]['label'], 
                                              ASSETS[asset2]['label']), use_container_width=True)
    
    # Interpretación del Lead-Lag
    max_corr_idx = lead_lag_df['correlation'].abs().idxmax()
    max_lag = lead_lag_df.loc[max_corr_idx, 'lag']
    max_corr_val = lead_lag_df.loc[max_corr_idx, 'correlation']
    
    if max_lag < 0:
        st.info(f"💡 **{ASSETS[asset2]['label']}** lidera a **{ASSETS[asset1]['label']}** por {abs(max_lag)} días (correlación: {max_corr_val:.3f})")
    elif max_lag > 0:
        st.info(f"💡 **{ASSETS[asset1]['label']}** lidera a **{ASSETS[asset2]['label']}** por {max_lag} días (correlación: {max_corr_val:.3f})")
    else:
        st.info(f"💡 No hay relación de liderazgo clara. Correlación simultánea: {max_corr_val:.3f}")
    
    # Multi-Window Correlation
    st.markdown("### 📊 Correlación Multi-Ventana")
    st.caption("Observa cómo cambia la correlación en diferentes timeframes (corto, medio, largo plazo)")
    
    multi_corr = calculate_rolling_correlation_multi_window(df_prices, asset1, asset2, 
                                                             windows=[10, 30, 60, 120])
    st.plotly_chart(plot_multi_window_correlation(multi_corr, ASSETS[asset1]['label'], 
                                                  ASSETS[asset2]['label']), use_container_width=True)
    
    # Métricas multi-ventana actuales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Corr 10d", f"{multi_corr['corr_10d'].iloc[-1]:.3f}" if 'corr_10d' in multi_corr else "N/A")
    col2.metric("Corr 30d", f"{multi_corr['corr_30d'].iloc[-1]:.3f}" if 'corr_30d' in multi_corr else "N/A")
    col3.metric("Corr 60d", f"{multi_corr['corr_60d'].iloc[-1]:.3f}" if 'corr_60d' in multi_corr else "N/A")
    col4.metric("Corr 120d", f"{multi_corr['corr_120d'].iloc[-1]:.3f}" if 'corr_120d' in multi_corr else "N/A")
    
    # Correlación condicional
    st.markdown("### 🔍 Correlación Condicional")
    st.caption("Cómo se correlacionan en diferentes condiciones de mercado (alcista/bajista/crisis)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    corr_positive = calculate_conditional_correlation(returns1, returns2, 'positive')
    corr_negative = calculate_conditional_correlation(returns1, returns2, 'negative')
    corr_crisis = calculate_conditional_correlation(returns1, returns2, 'crisis')
    tail_corr = calculate_tail_correlation(returns1, returns2, quantile=0.05)
    
    col1.metric("Mercado Alcista", f"{corr_positive:.4f}")
    col2.metric("Mercado Bajista", f"{corr_negative:.4f}")
    col3.metric("Alta Volatilidad", f"{corr_crisis:.4f}")
    col4.metric("Cola Inferior (5%)", f"{tail_corr['lower_tail']:.4f}")
    
    # Interpretación de correlación condicional
    if abs(corr_crisis) > abs(tail_corr['normal']):
        st.warning(f"⚠️ La correlación aumenta significativamente durante crisis ({corr_crisis:.3f} vs {tail_corr['normal']:.3f} normal)")
    
    # Comparación de precios
    st.markdown("### 📉 Comparación de Precios Normalizados")
    st.plotly_chart(
        plot_price_comparison(df_prices, asset1, asset2, 
                            ASSETS[asset1]['label'], 
                            ASSETS[asset2]['label']),
        use_container_width=True
    )

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
    st.subheader("📊 Estadísticas Avanzadas de Correlación")
    
    # Selección de par para análisis
    col1, col2 = st.columns(2)
    
    with col1:
        stats_asset1 = st.selectbox(
            "Activo 1",
            options=selected_assets,
            format_func=lambda x: ASSETS[x]['label'],
            key='stats_asset1'
        )
    
    with col2:
        stats_asset2 = st.selectbox(
            "Activo 2",
            options=[a for a in selected_assets if a != stats_asset1],
            format_func=lambda x: ASSETS[x]['label'],
            key='stats_asset2'
        )
    
    # Análisis de períodos
    corr_df_stats = calculate_rolling_correlation(df_prices, stats_asset1, stats_asset2, window_size, step_size)
    
    positive = (corr_df_stats['correlation'] > 0).sum()
    negative = (corr_df_stats['correlation'] < 0).sum()
    strong_pos = (corr_df_stats['correlation'] > 0.5).sum()
    strong_neg = (corr_df_stats['correlation'] < -0.5).sum()
    total = len(corr_df_stats)
    
    st.markdown("### 📈 Distribución Temporal")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("% Correlación Positiva", f"{positive/total*100:.1f}%")
    with col2:
        st.metric("% Correlación Negativa", f"{negative/total*100:.1f}%")
    with col3:
        st.metric("% Fuerte Positiva (>0.5)", f"{strong_pos/total*100:.1f}%")
    with col4:
        st.metric("% Fuerte Negativa (<-0.5)", f"{strong_neg/total*100:.1f}%")
    
    # Percentiles de correlación
    st.markdown("### 📊 Distribución Percentiles")
    percentiles = calculate_correlation_percentile(corr_df_stats['correlation'])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("P10", f"{percentiles['p10']:.3f}")
    col2.metric("P25", f"{percentiles['p25']:.3f}")
    col3.metric("P50 (Mediana)", f"{percentiles['p50']:.3f}")
    col4.metric("P75", f"{percentiles['p75']:.3f}")
    col5.metric("P90", f"{percentiles['p90']:.3f}")
    
    # Distribución de correlaciones
    st.markdown("### 📈 Distribución Histórica")
    fig_hist = go.Figure(data=[go.Histogram(
        x=corr_df_stats['correlation'],
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
    
    # Estabilidad de correlación
    st.markdown("### 🎯 Estabilidad de Correlación")
    stability_df = calculate_correlation_stability(corr_df_stats['correlation'], window=60)
    st.plotly_chart(plot_correlation_stability(stability_df), use_container_width=True)
    
    st.caption("💡 Un CV bajo indica una correlación más estable y predecible")
    
    # Breakpoints
    st.markdown("### ⚡ Puntos de Cambio de Régimen")
    breakpoints = find_correlation_breakpoints(corr_df_stats['correlation'], threshold=0.3)
    
    if len(breakpoints) > 0:
        st.warning(f"⚠️ Detectados {len(breakpoints)} cambios significativos de correlación")
        
        # Mostrar tabla de breakpoints
        bp_df = pd.DataFrame({
            'Fecha': breakpoints.index,
            'Cambio Absoluto': breakpoints.values
        }).sort_values('Cambio Absoluto', ascending=False).head(10)
        
        st.dataframe(bp_df, use_container_width=True)
    else:
        st.success("✅ Correlación relativamente estable (sin cambios abruptos)")
    
    # Correlación de Spearman vs Pearson
    st.markdown("### 📊 Correlación Spearman vs Pearson")
    st.caption("Spearman es más robusta a outliers que Pearson")
    
    prices1 = df_prices[stats_asset1]
    prices2 = df_prices[stats_asset2]
    
    spearman_corr = rank_correlation_spearman(prices1, prices2, window=60)
    pearson_corr = calculate_returns(prices1).rolling(60).corr(calculate_returns(prices2))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pearson_corr.index, y=pearson_corr, 
                            name='Pearson', line=dict(color='#3b82f6', width=2)))
    fig.add_trace(go.Scatter(x=spearman_corr.index, y=spearman_corr, 
                            name='Spearman', line=dict(color='#10b981', width=2)))
    
    fig.update_layout(
        title='Comparación Spearman vs Pearson',
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de correlación dinámica
    st.markdown("### 🔄 Correlación Dinámica (EWMA)")
    st.caption("Correlación con pesos exponenciales que da más importancia a datos recientes")
    
    returns1_stats = calculate_returns(prices1)
    returns2_stats = calculate_returns(prices2)
    
    ewma_corr = calculate_time_varying_correlation(returns1_stats, returns2_stats, method='ewm', span=30)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ewma_corr.index, y=ewma_corr,
                            name='EWMA Correlation',
                            line=dict(color='#8b5cf6', width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color="#666666")
    
    fig.update_layout(
        title='Correlación Dinámica (Exponentially Weighted)',
        xaxis_title='Fecha',
        yaxis_title='Correlación',
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
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

with tab5:
    st.subheader("🔍 Búsqueda Automática de Mejores Pares")
    st.caption("Identifica automáticamente los mejores pares para trading usando múltiples criterios")
    
    # Selector de tipo de correlación
    st.markdown("### 🎯 Tipo de Correlación")
    
    correlation_type = st.radio(
        "Selecciona el tipo de pares a buscar:",
        ["Correlación Positiva (Pairs Trading)", "Correlación Inversa (Hedging)", "Ambos"],
        horizontal=True
    )
    
    # Configuración de búsqueda
    st.markdown("### ⚙️ Configuración de Búsqueda")
    
    if correlation_type == "Correlación Positiva (Pairs Trading)":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_coint_pvalue = st.slider("P-value máximo (cointegración)", 0.01, 0.10, 0.05, 0.01)
        
        with col2:
            min_correlation = st.slider("Correlación mínima", 0.5, 0.95, 0.7, 0.05)
        
        with col3:
            max_distance = st.slider("Distancia máxima", 0.5, 2.0, 1.0, 0.1)
        
        if st.button("🔎 Buscar Mejores Pares", type="primary"):
            with st.spinner("Analizando pares... Esto puede tomar unos minutos..."):
                best_pairs = find_best_pairs_comprehensive(
                    df_prices[selected_assets],
                    min_cointegration_pvalue=min_coint_pvalue,
                    max_distance=max_distance,
                    min_correlation=min_correlation
                )
            
            if len(best_pairs) > 0:
                st.success(f"✅ Encontrados {len(best_pairs)} pares que cumplen los criterios")
                
                # Mostrar gráfico de ranking
                st.plotly_chart(plot_best_pairs_ranking(best_pairs, top_n=15), use_container_width=True)
                
                # Mostrar tabla detallada
                st.markdown("### 📋 Tabla Detallada de Pares")
                
                display_df = best_pairs.head(20).copy()
                display_df['asset1_name'] = display_df['asset1'].apply(lambda x: ASSETS[x]['label'])
                display_df['asset2_name'] = display_df['asset2'].apply(lambda x: ASSETS[x]['label'])
                
                display_columns = {
                    'asset1_name': 'Activo 1',
                    'asset2_name': 'Activo 2',
                    'score': 'Score',
                    'correlation': 'Correlación',
                    'cointegration_pvalue': 'P-value Coint.',
                    'hurst': 'Hurst Exp.',
                    'distance': 'Distancia'
                }
                
                display_df = display_df[list(display_columns.keys())].rename(columns=display_columns)
                
                # Formatear tabla
                def highlight_score(val):
                    if val > 80:
                        color = '#10b981'
                    elif val > 60:
                        color = '#84cc16'
                    elif val > 40:
                        color = '#f59e0b'
                    else:
                        color = '#ef4444'
                    return f'background-color: {color}; color: white'
                
                styled_table = display_df.style.applymap(
                    highlight_score, 
                    subset=['Score']
                ).format({
                    'Score': '{:.1f}',
                    'Correlación': '{:.3f}',
                    'P-value Coint.': '{:.4f}',
                    'Hurst Exp.': '{:.3f}',
                    'Distancia': '{:.2f}'
                })
                
                st.dataframe(styled_table, use_container_width=True)
                
                # Análisis del mejor par
                st.markdown("### 🏆 Análisis del Mejor Par")
                
                best_pair = best_pairs.iloc[0]
                best_asset1 = best_pair['asset1']
                best_asset2 = best_pair['asset2']
                
                st.info(f"**Mejor Par:** {ASSETS[best_asset1]['label']} / {ASSETS[best_asset2]['label']} | Score: {best_pair['score']:.1f}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Correlación", f"{best_pair['correlation']:.3f}")
                col2.metric("Cointegración p-val", f"{best_pair['cointegration_pvalue']:.4f}")
                col3.metric("Hurst Exponent", f"{best_pair['hurst']:.3f}")
                col4.metric("Distancia", f"{best_pair['distance']:.2f}")
                
                # Gráficos del mejor par
                prices1_best = df_prices[best_asset1]
                prices2_best = df_prices[best_asset2]
                
                st.plotly_chart(
                    plot_price_comparison(df_prices, best_asset1, best_asset2,
                                        ASSETS[best_asset1]['label'],
                                        ASSETS[best_asset2]['label']),
                    use_container_width=True
                )
                
                # Spread analysis
                spread_best, hr_best = calculate_spread(prices1_best, prices2_best)
                zscore_best = calculate_zscore(spread_best, window=30)
                
                st.plotly_chart(
                    plot_spread_analysis(prices1_best, prices2_best,
                                       ASSETS[best_asset1]['label'],
                                       ASSETS[best_asset2]['label']),
                    use_container_width=True
                )
                
                # Descargar resultados
                st.markdown("### 📥 Descargar Resultados")
                csv_pairs = best_pairs.to_csv(index=False)
                st.download_button(
                    label="Descargar tabla de pares como CSV",
                    data=csv_pairs,
                    file_name="mejores_pares_trading.csv",
                    mime="text/csv"
                )
                
            else:
                st.warning("⚠️ No se encontraron pares que cumplan todos los criterios. Intenta relajar los parámetros.")
    
    elif correlation_type == "Correlación Inversa (Hedging)":
        col1, col2 = st.columns(2)
        
        with col1:
            min_neg_corr = st.slider("Correlación mínima (negativa)", -0.95, -0.3, -0.7, 0.05)
        
        with col2:
            max_neg_corr = st.slider("Correlación máxima (negativa)", -0.95, -0.3, -0.3, 0.05)
        
        if st.button("🔎 Buscar Pares Inversos", type="primary"):
            with st.spinner("Buscando pares con correlación inversa..."):
                inverse_pairs = find_best_inverse_pairs(
                    df_prices[selected_assets],
                    min_negative_correlation=min_neg_corr,
                    max_correlation=max_neg_corr
                )
            
            if len(inverse_pairs) > 0:
                st.success(f"✅ Encontrados {len(inverse_pairs)} pares con correlación inversa")
                
                # Gráfico de ranking
                st.plotly_chart(plot_inverse_pairs_ranking(inverse_pairs, top_n=15), use_container_width=True)
                
                # Tabla detallada (similar al código en tab7)
                st.markdown("### 📋 Tabla Detallada de Pares Inversos")
                
                display_df = inverse_pairs.head(20).copy()
                display_df['asset1_name'] = display_df['asset1'].apply(lambda x: ASSETS[x]['label'])
                display_df['asset2_name'] = display_df['asset2'].apply(lambda x: ASSETS[x]['label'])
                
                st.dataframe(display_df[['asset1_name', 'asset2_name', 'score', 'correlation', 
                                        'corr_stability', 'vol_ratio']], use_container_width=True)
                
                # Descargar
                csv_inverse = inverse_pairs.to_csv(index=False)
                st.download_button(
                    label="Descargar pares inversos como CSV",
                    data=csv_inverse,
                    file_name="pares_correlacion_inversa.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ No se encontraron pares inversos en el rango especificado.")
    
    else:  # Ambos
        st.info("🔄 Buscando tanto pares con correlación positiva como inversa...")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Correlación Positiva**")
            min_corr_pos = st.slider("Mín. correlación positiva", 0.5, 0.95, 0.7, 0.05, key='pos_min')
        
        with col2:
            st.markdown("**Correlación Inversa**")
            max_corr_neg = st.slider("Máx. correlación negativa", -0.95, -0.3, -0.5, 0.05, key='neg_max')
        
        if st.button("🔎 Buscar Ambos Tipos de Pares", type="primary"):
            with st.spinner("Analizando todos los pares..."):
                # Buscar pares positivos
                best_pairs_pos = find_best_pairs_comprehensive(
                    df_prices[selected_assets],
                    min_cointegration_pvalue=0.05,
                    max_distance=1.0,
                    min_correlation=min_corr_pos
                )
                
                # Buscar pares inversos
                inverse_pairs = find_best_inverse_pairs(
                    df_prices[selected_assets],
                    min_negative_correlation=-0.95,
                    max_correlation=max_corr_neg
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Pares Correlación Positiva")
                if len(best_pairs_pos) > 0:
                    st.success(f"✅ {len(best_pairs_pos)} pares encontrados")
                    top_pos = best_pairs_pos.head(10).copy()
                    top_pos['Pair'] = top_pos['asset1'].apply(lambda x: ASSETS[x]['label']) + ' / ' + \
                                      top_pos['asset2'].apply(lambda x: ASSETS[x]['label'])
                    st.dataframe(top_pos[['Pair', 'score', 'correlation']], use_container_width=True)
                else:
                    st.warning("No encontrados")
            
            with col2:
                st.markdown("### 🛡️ Pares Correlación Inversa")
                if len(inverse_pairs) > 0:
                    st.success(f"✅ {len(inverse_pairs)} pares encontrados")
                    top_inv = inverse_pairs.head(10).copy()
                    top_inv['Pair'] = top_inv['asset1'].apply(lambda x: ASSETS[x]['label']) + ' / ' + \
                                      top_inv['asset2'].apply(lambda x: ASSETS[x]['label'])
                    st.dataframe(top_inv[['Pair', 'score', 'correlation']], use_container_width=True)
                else:
                    st.warning("No encontrados")
    
    # Análisis de distancias entre todos los pares
    st.markdown("---")
    st.markdown("### 📊 Análisis de Distancias")
    st.caption("Pares más similares por distancia euclidiana")
    
    if st.button("Calcular Distancias"):
        with st.spinner("Calculando distancias..."):
            distance_pairs = detect_pairs_by_distance(df_prices[selected_assets], 
                                                      method='euclidean', top_n=20)
        
        if len(distance_pairs) > 0:
            distance_pairs['asset1_name'] = distance_pairs['asset1'].apply(lambda x: ASSETS[x]['label'])
            distance_pairs['asset2_name'] = distance_pairs['asset2'].apply(lambda x: ASSETS[x]['label'])
            
            display_dist = distance_pairs[['asset1_name', 'asset2_name', 'distance']].rename(columns={
                'asset1_name': 'Activo 1',
                'asset2_name': 'Activo 2',
                'distance': 'Distancia'
            })
            
            st.dataframe(display_dist, use_container_width=True)

with tab6:
    st.subheader("🛡️ Análisis de Correlación Inversa & Hedging")
    st.caption("Identifica pares con correlación negativa fuerte para estrategias de cobertura y diversificación")
    
    # Búsqueda de pares inversos
    st.markdown("### 🔍 Búsqueda de Pares con Correlación Inversa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_neg_corr = st.slider("Correlación mínima (negativa)", -0.95, -0.3, -0.7, 0.05)
    
    with col2:
        max_neg_corr = st.slider("Correlación máxima (negativa)", -0.95, -0.3, -0.3, 0.05)
    
    if st.button("🔎 Buscar Pares Inversos", type="primary"):
        with st.spinner("Buscando pares con correlación inversa..."):
            inverse_pairs = find_best_inverse_pairs(
                df_prices[selected_assets],
                min_negative_correlation=min_neg_corr,
                max_correlation=max_neg_corr
            )
        
        if len(inverse_pairs) > 0:
            st.success(f"✅ Encontrados {len(inverse_pairs)} pares con correlación inversa")
            
            # Gráfico de ranking
            st.plotly_chart(plot_inverse_pairs_ranking(inverse_pairs, top_n=15), use_container_width=True)
            
            # Tabla detallada
            st.markdown("### 📋 Tabla Detallada de Pares Inversos")
            
            display_df = inverse_pairs.head(20).copy()
            display_df['asset1_name'] = display_df['asset1'].apply(lambda x: ASSETS[x]['label'])
            display_df['asset2_name'] = display_df['asset2'].apply(lambda x: ASSETS[x]['label'])
            
            display_columns = {
                'asset1_name': 'Activo 1',
                'asset2_name': 'Activo 2',
                'score': 'Score',
                'correlation': 'Correlación',
                'corr_stability': 'Estabilidad',
                'vol_ratio': 'Ratio Vol',
                'vol1': 'Vol 1',
                'vol2': 'Vol 2',
                'max_lag_corr': 'Max Lag Corr'
            }
            
            display_df = display_df[list(display_columns.keys())].rename(columns=display_columns)
            
            styled_table = display_df.style.format({
                'Score': '{:.1f}',
                'Correlación': '{:.3f}',
                'Estabilidad': '{:.3f}',
                'Ratio Vol': '{:.2f}',
                'Vol 1': '{:.2%}',
                'Vol 2': '{:.2%}',
                'Max Lag Corr': '{:.3f}'
            })
            
            st.dataframe(styled_table, use_container_width=True)
            
            # Análisis del mejor par inverso
            st.markdown("### 🏆 Análisis del Mejor Par Inverso")
            
            best_inverse = inverse_pairs.iloc[0]
            inv_asset1 = best_inverse['asset1']
            inv_asset2 = best_inverse['asset2']
            
            st.info(f"**Mejor Par Inverso:** {ASSETS[inv_asset1]['label']} / {ASSETS[inv_asset2]['label']} | Score: {best_inverse['score']:.1f}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Correlación", f"{best_inverse['correlation']:.3f}")
            col2.metric("Estabilidad", f"{best_inverse['corr_stability']:.3f}")
            col3.metric("Ratio Volatilidad", f"{best_inverse['vol_ratio']:.2f}")
            col4.metric("Max Lag Corr", f"{best_inverse['max_lag_corr']:.3f}")
            
            # Análisis de hedging
            st.markdown("### 🛡️ Análisis de Efectividad del Hedge")
            
            prices_inv1 = df_prices[inv_asset1]
            prices_inv2 = df_prices[inv_asset2]
            
            # Calcular hedge ratio óptimo
            optimal_hr = calculate_optimal_hedge_ratio_inverse(prices_inv1, prices_inv2)
            
            # Calcular efectividad con diferentes hedge ratios
            hr_test = [0.5, 0.75, 1.0, optimal_hr, 1.5]
            
            st.markdown(f"**Hedge Ratio Óptimo:** {optimal_hr:.3f}")
            st.caption("El hedge ratio óptimo minimiza la volatilidad del portfolio hedgeado")
            
            # Comparar diferentes hedge ratios
            st.markdown("#### Comparación de Hedge Ratios")
            
            hedge_results = []
            for hr in hr_test:
                result = calculate_hedge_effectiveness(prices_inv1, prices_inv2, hr)
                hedge_results.append({
                    'Hedge Ratio': hr,
                    'Vol Reduction %': result['vol_reduction_pct'],
                    'DD Reduction %': result['dd_reduction_pct'],
                    'Sharpe Hedged': result['sharpe_hedged']
                })
            
            hedge_df = pd.DataFrame(hedge_results)
            st.dataframe(hedge_df.style.format({
                'Hedge Ratio': '{:.3f}',
                'Vol Reduction %': '{:.1f}%',
                'DD Reduction %': '{:.1f}%',
                'Sharpe Hedged': '{:.3f}'
            }).background_gradient(subset=['Vol Reduction %', 'DD Reduction %'], cmap='RdYlGn'), 
            use_container_width=True)
            
            # Visualización de efectividad del hedge
            st.markdown("#### Visualización del Hedge")
            
            st.plotly_chart(
                plot_hedge_effectiveness(prices_inv1, prices_inv2, optimal_hr,
                                        ASSETS[inv_asset1]['label'],
                                        ASSETS[inv_asset2]['label']),
                use_container_width=True
            )
            
            # Métricas detalladas del hedge óptimo
            hedge_metrics = calculate_hedge_effectiveness(prices_inv1, prices_inv2, optimal_hr)
            
            st.markdown("#### 📊 Métricas Detalladas del Hedge Óptimo")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Volatilidad Original", f"{hedge_metrics['vol_original']:.2%}")
                st.metric("Volatilidad Hedgeada", f"{hedge_metrics['vol_hedged']:.2%}")
                st.metric("Reducción de Vol", f"{hedge_metrics['vol_reduction_pct']:.1f}%",
                         delta=f"{hedge_metrics['vol_reduction_pct']:.1f}%")
            
            with col2:
                st.metric("Drawdown Original", f"{hedge_metrics['dd_original']:.2f}%")
                st.metric("Drawdown Hedgeado", f"{hedge_metrics['dd_hedged']:.2f}%")
                st.metric("Reducción de DD", f"{hedge_metrics['dd_reduction_pct']:.1f}%",
                         delta=f"{hedge_metrics['dd_reduction_pct']:.1f}%")
            
            with col3:
                st.metric("Sharpe Original", f"{hedge_metrics['sharpe_original']:.3f}")
                st.metric("Sharpe Hedgeado", f"{hedge_metrics['sharpe_hedged']:.3f}")
                sharpe_change = hedge_metrics['sharpe_hedged'] - hedge_metrics['sharpe_original']
                st.metric("Cambio Sharpe", f"{sharpe_change:+.3f}",
                         delta=f"{sharpe_change:+.3f}")
            
            # Interpretación
            if hedge_metrics['vol_reduction_pct'] > 30:
                st.success(f"✅ Excelente hedge: reduce volatilidad en {hedge_metrics['vol_reduction_pct']:.1f}%")
            elif hedge_metrics['vol_reduction_pct'] > 15:
                st.info(f"✓ Buen hedge: reduce volatilidad en {hedge_metrics['vol_reduction_pct']:.1f}%")
            else:
                st.warning(f"⚠️ Hedge moderado: reduce volatilidad en {hedge_metrics['vol_reduction_pct']:.1f}%")
            
            # Detección de regímenes inversos
            st.markdown("### 📈 Detección de Regímenes de Correlación Inversa")
            
            returns_inv1 = calculate_returns(prices_inv1)
            returns_inv2 = calculate_returns(prices_inv2)
            rolling_corr_inv = returns_inv1.rolling(60).corr(returns_inv2)
            
            regime_info = detect_correlation_regime_inverse(rolling_corr_inv, threshold=-0.3)
            
            col1, col2 = st.columns(2)
            
            with col1:
                status = "✅ SÍ" if regime_info['in_inverse_regime'] else "❌ NO"
                st.metric("En Régimen Inverso Actual", status)
                st.caption("Correlación < -0.3")
            
            with col2:
                st.metric("% Tiempo en Régimen Inverso", f"{regime_info['pct_time_inverse']:.1f}%")
                st.metric("Correlación Actual", f"{regime_info['current_correlation']:.3f}")
            
            # Gráfico de regímenes
            st.plotly_chart(
                plot_correlation_regime_inverse(rolling_corr_inv, threshold=-0.3),
                use_container_width=True
            )
            
            # Estrategia de trading para correlación inversa
            st.markdown("### 💡 Estrategia Sugerida")
            
            if regime_info['in_inverse_regime']:
                st.success(f"""
                **✅ Estrategia de Hedging Activa**
                
                La correlación actual ({regime_info['current_correlation']:.3f}) indica un buen momento para implementar un hedge:
                
                1. **Posición Long**: {ASSETS[inv_asset1]['label']}
                2. **Posición Short**: {ASSETS[inv_asset2]['label']} con ratio {optimal_hr:.3f}
                3. **Objetivo**: Reducir volatilidad del portfolio en ~{hedge_metrics['vol_reduction_pct']:.0f}%
                4. **Beneficio**: Protección contra caídas con correlación inversa estable
                """)
            else:
                st.info(f"""
                **⚪ Monitorear Correlación**
                
                La correlación actual ({regime_info['current_correlation']:.3f}) no está en régimen inverso fuerte.
                
                - Esperar a que la correlación caiga por debajo de -0.3
                - Monitorear estabilidad de la correlación
                - Considerar otros pares con mejor correlación inversa
                """)
            
            # Descarga de resultados
            st.markdown("### 📥 Descargar Resultados")
            csv_inverse = inverse_pairs.to_csv(index=False)
            st.download_button(
                label="Descargar pares inversos como CSV",
                data=csv_inverse,
                file_name="pares_correlacion_inversa.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("⚠️ No se encontraron pares con correlación inversa en el rango especificado.")
    
    # Análisis individual de par inverso
    st.markdown("---")
    st.markdown("### 🔬 Análisis Individual de Par Inverso")
    
    col1, col2 = st.columns(2)
    
    with col1:
        inv_manual_asset1 = st.selectbox(
            "Activo 1 (para hedging)",
            options=selected_assets,
            format_func=lambda x: ASSETS[x]['label'],
            key='inv_manual_1'
        )
    
    with col2:
        inv_manual_asset2 = st.selectbox(
            "Activo 2 (hedge)",
            options=[a for a in selected_assets if a != inv_manual_asset1],
            format_func=lambda x: ASSETS[x]['label'],
            key='inv_manual_2'
        )
    
    # Análisis del par seleccionado manualmente
    prices_m1 = df_prices[inv_manual_asset1]
    prices_m2 = df_prices[inv_manual_asset2]
    
    corr_manual = prices_m1.corr(prices_m2)
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Correlación", f"{corr_manual:.3f}")
    
    if corr_manual < -0.5:
        col2.metric("Tipo", "🛡️ Hedge Fuerte")
        col3.metric("Recomendación", "✅ Bueno para hedging")
    elif corr_manual < -0.3:
        col2.metric("Tipo", "🛡️ Hedge Moderado")
        col3.metric("Recomendación", "⚠️ Considerar")
    else:
        col2.metric("Tipo", "❌ No Inverso")
        col3.metric("Recomendación", "❌ No recomendado")
    
    # Calcular y mostrar hedge ratio óptimo
    if corr_manual < -0.2:
        optimal_hr_manual = calculate_optimal_hedge_ratio_inverse(prices_m1, prices_m2)
        st.info(f"**Hedge Ratio Óptimo:** {optimal_hr_manual:.4f}")
        
        # Efectividad del hedge
        hedge_manual = calculate_hedge_effectiveness(prices_m1, prices_m2, optimal_hr_manual)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Reducción Volatilidad", f"{hedge_manual['vol_reduction_pct']:.1f}%")
        col2.metric("Reducción Drawdown", f"{hedge_manual['dd_reduction_pct']:.1f}%")
        col3.metric("Sharpe Hedgeado", f"{hedge_manual['sharpe_hedged']:.3f}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Guía Rápida")
st.sidebar.markdown("""
**Correlaciones:**
- **> 0.7**: Muy fuerte positiva ✅
- **0.5-0.7**: Fuerte positiva
- **< -0.5**: Fuerte INVERSA 🛡️
- **< -0.7**: Muy fuerte INVERSA ⚡

**Uso Principal:**
- **Tab 5**: Búsqueda automática de mejores pares
- **Tab 6**: Análisis de hedging (correlación inversa)
- **Tab 1**: Análisis detallado de cualquier par
- **Tab 4**: Pairs trading (cointegración, spread, z-score)

**Estrategias:**
1. **Pairs Trading**: Correlación positiva (0.7+)
   - Mean reversion
   - Z-score > 2 o < -2
   
2. **Hedging**: Correlación inversa (<-0.5)
   - Reduce volatilidad 30-60%
   - Protección en caídas
   
3. **Diversificación**: Correlación baja (-0.3 a 0.3)

**Cointegración:**
- P-value < 0.05: ✅ Cointegrados
- Spread estacionario

**Hurst Exponent:**
- < 0.5: Mean reverting ✅
- > 0.5: Trending

**Z-Score Pairs Trading:**
- > 2: Señal VENTA spread
- < -2: Señal COMPRA spread
- Entre -1 y 1: Sin señal
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.info(f"""
**{len(ASSETS)} activos disponibles**

- Delay recomendado: 10s
- Mín. activos: 2
- Recomendado: 10-20 activos para búsqueda
""")

st.sidebar.markdown("---")
st.sidebar.success("✨ Enfoque: Búsqueda de Pares & Hedging")
st.sidebar.markdown("""
**Funciones Clave:**
- 🔍 Búsqueda automática de pares
- 🛡️ Análisis de hedging inverso
- 📊 Lead-lag correlation
- 🎯 Multi-window analysis
- 📈 Régimen detection
- 💰 Hedge ratio óptimo
""")
