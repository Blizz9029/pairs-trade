import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from datetime import datetime, timedelta
import pytz
import json
import uuid
import requests
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Shashank Udupa Pairs Trading App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4b8bbe;
    }
    .dataframe {
        font-size: 12px;
    }
    .positive {
        color: #00ff00;
    }
    .negative {
        color: #ff0000;
    }
    .z-positive {
        color: #ff0000;
        font-weight: bold;
    }
    .z-negative {
        color: #00ff00;
        font-weight: bold;
    }
    .trade-form {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .position-card {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    div[data-testid="stSidebar"] {
        background-color: #0e1117;
    }
    hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
        border: 0;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Configure IST timezone
IST = pytz.timezone('Asia/Kolkata')

# Initialize session state
if 'nifty_stocks' not in st.session_state:
    st.session_state.nifty_stocks = {}
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'last_updated' not in st.session_state:
    st.session_state.last_updated = None
if 'pairs' not in st.session_state:
    st.session_state.pairs = []
if 'positions' not in st.session_state:
    st.session_state.positions = []
if 'selected_pair' not in st.session_state:
    st.session_state.selected_pair = None
if 'show_trade_form' not in st.session_state:
    st.session_state.show_trade_form = False
if 'stock_limit' not in st.session_state:
    st.session_state.stock_limit = 100  # Default number of stocks to analyze

# Fetch Nifty 500 stocks
def get_nifty500_list():
    """Get list of Nifty 500 stocks with their sectors"""
    try:
        # Try to fetch from NSE website or use CSV from GitHub
        url = "https://raw.githubusercontent.com/stock-analysis-project/stock-data/main/nifty500_stocks.csv"
        response = requests.get(url)
        
        if response.status_code == 200:
            # Parse CSV data
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            
            # Create dictionary with stock symbols and details
            stocks = {}
            for _, row in df.iterrows():
                symbol = row['Symbol'] + '.NS'
                stocks[symbol] = {
                    'name': row['Company Name'],
                    'sector': row['Sector']
                }
            
            return stocks
        else:
            # Fallback to a smaller set of stocks if unable to fetch
            st.warning("Could not fetch Nifty 500 stocks, using default stock list")
            return get_default_stocks()
            
    except Exception as e:
        st.error(f"Error fetching Nifty 500 stocks: {str(e)}")
        return get_default_stocks()

# Default stocks as fallback
def get_default_stocks():
    """Default list of major NSE stocks by sector"""
    stocks = {
        'ICICIBANK.NS': {'name': 'ICICI Bank Ltd', 'sector': 'Financial Services'},
        'HDFCBANK.NS': {'name': 'HDFC Bank Ltd', 'sector': 'Financial Services'},
        'RELIANCE.NS': {'name': 'Reliance Industries Ltd', 'sector': 'Energy'},
        'BPCL.NS': {'name': 'Bharat Petroleum Corp Ltd', 'sector': 'Energy'},
        'TCS.NS': {'name': 'Tata Consultancy Services Ltd', 'sector': 'IT'},
        'INFY.NS': {'name': 'Infosys Ltd', 'sector': 'IT'},
        'SBIN.NS': {'name': 'State Bank of India', 'sector': 'Financial Services'},
        'PNB.NS': {'name': 'Punjab National Bank', 'sector': 'Financial Services'},
        'SUNPHARMA.NS': {'name': 'Sun Pharmaceutical Industries Ltd', 'sector': 'Pharmaceuticals'},
        'DRREDDY.NS': {'name': 'Dr. Reddy\'s Laboratories Ltd', 'sector': 'Pharmaceuticals'},
        'TATAMOTORS.NS': {'name': 'Tata Motors Ltd', 'sector': 'Automobiles'},
        'M&M.NS': {'name': 'Mahindra & Mahindra Ltd', 'sector': 'Automobiles'},
        'HINDUNILVR.NS': {'name': 'Hindustan Unilever Ltd', 'sector': 'FMCG'},
        'ITC.NS': {'name': 'ITC Ltd', 'sector': 'FMCG'},
        'BHARTIARTL.NS': {'name': 'Bharti Airtel Ltd', 'sector': 'Telecom'},
        'IDEA.NS': {'name': 'Vodafone Idea Ltd', 'sector': 'Telecom'},
        'HCLTECH.NS': {'name': 'HCL Technologies Ltd', 'sector': 'IT'},
        'WIPRO.NS': {'name': 'Wipro Ltd', 'sector': 'IT'},
        'AXISBANK.NS': {'name': 'Axis Bank Ltd', 'sector': 'Financial Services'},
        'KOTAKBANK.NS': {'name': 'Kotak Mahindra Bank Ltd', 'sector': 'Financial Services'}
    }
    return stocks

# Initialize stocks list if empty
if not st.session_state.nifty_stocks:
    st.session_state.nifty_stocks = get_nifty500_list()

def fetch_stock_data(force_refresh=False):
    """Fetch historical price data for Nifty stocks with caching"""
    now = datetime.now(IST)
    
    # Return cached data if it's recent (within last 15 minutes) and not forced refresh
    if (not force_refresh and st.session_state.last_updated and 
        (now - st.session_state.last_updated).total_seconds() < 900 and 
        st.session_state.stock_data is not None):
        return st.session_state.stock_data
    
    # Get the list of stocks to analyze (limited to stock_limit)
    all_stocks = list(st.session_state.nifty_stocks.keys())
    
    # Group by sector and select top stocks from each sector
    sectors = {}
    for symbol, details in st.session_state.nifty_stocks.items():
        sector = details['sector']
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(symbol)
    
    # Balance stocks across sectors
    stocks_to_analyze = []
    stocks_per_sector = max(3, st.session_state.stock_limit // len(sectors))
    
    for sector, symbols in sectors.items():
        # Take top N stocks from each sector
        stocks_to_analyze.extend(symbols[:stocks_per_sector])
        
        # Stop if we've reached the limit
        if len(stocks_to_analyze) >= st.session_state.stock_limit:
            stocks_to_analyze = stocks_to_analyze[:st.session_state.stock_limit]
            break
    
    # Show a progress bar
    progress_bar = st.progress(0)
    total_stocks = len(stocks_to_analyze)
    
    with st.spinner(f'Fetching data for {total_stocks} stocks...'):
        try:
            # Fetch 180 days of historical data
            end_date = now
            start_date = end_date - timedelta(days=180)
            
            all_data = {}
            for i, symbol in enumerate(stocks_to_analyze):
                try:
                    stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                    if not stock_data.empty:
                        all_data[symbol] = stock_data['Adj Close']
                except Exception as e:
                    st.error(f"Error fetching data for {symbol}: {str(e)}")
                
                # Update progress
                progress_bar.progress((i + 1) / total_stocks)
            
            # Create DataFrame and update session state
            data_frame = pd.DataFrame(all_data)
            st.session_state.stock_data = data_frame
            st.session_state.last_updated = now
            
            # Remove progress bar
            progress_bar.empty()
            
            return data_frame
        
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            # Return old data if available
            if st.session_state.stock_data is not None:
                return st.session_state.stock_data
            return pd.DataFrame()  # Empty DataFrame as fallback

def calculate_stop_loss(stock_data, position_type, current_price, lookback=30):
    """Calculate stop loss based on volatility and technical levels"""
    if stock_data is None or len(stock_data) < lookback:
        # Default to 5% if insufficient data
        return round(current_price * (0.95 if position_type == 'BUY' else 1.05), 2)
    
    # Statistical stop loss based on volatility (2 standard deviations)
    recent_data = stock_data[-lookback:]
    daily_returns = recent_data.pct_change().dropna()
    volatility = daily_returns.std() * 2
    
    # Calculate stop loss based on position type
    if position_type == 'BUY':
        stop_loss = current_price * (1 - volatility)
    else:  # SELL
        stop_loss = current_price * (1 + volatility)
    
    return round(stop_loss, 2)

def analyze_pairs(min_correlation=0.7, min_z_score=2.0, selected_sector='All', require_cointegration=True):
    """Analyze all stock pairs to find trading opportunities"""
    # Get stock data
    df = st.session_state.stock_data
    if df is None or df.empty:
        df = fetch_stock_data()
        if df.empty:
            return []
    
    # Group stocks by sector
    sectors = {}
    for symbol in df.columns:
        if symbol in st.session_state.nifty_stocks:
            sector = st.session_state.nifty_stocks[symbol]['sector']
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(symbol)
    
    results = []
    
    # Create a progress bar
    progress_bar = st.progress(0)
    total_sectors = len(sectors)
    sector_idx = 0
    
    with st.spinner('Analyzing pairs...'):
        for sector, symbols in sectors.items():
            sector_idx += 1
            
            # Skip if filtering by sector
            if selected_sector != 'All' and sector != selected_sector:
                progress_bar.progress(sector_idx / total_sectors)
                continue
                
            if len(symbols) < 2:
                progress_bar.progress(sector_idx / total_sectors)
                continue
                
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    stock1 = symbols[i]
                    stock2 = symbols[j]
                    
                    if stock1 not in df.columns or stock2 not in df.columns:
                        continue
                    
                    # Calculate correlation
                    correlation = df[stock1].corr(df[stock2])
                    
                    if correlation < min_correlation:
                        continue
                    
                    # Test for cointegration
                    try:
                        stock1_data = df[stock1].dropna()
                        stock2_data = df[stock2].dropna()
                        
                        if len(stock1_data) != len(stock2_data):
                            # Align the indexes
                            common_idx = stock1_data.index.intersection(stock2_data.index)
                            stock1_data = stock1_data.loc[common_idx]
                            stock2_data = stock2_data.loc[common_idx]
                        
                        _, pvalue, _ = coint(stock1_data, stock2_data)
                        is_cointegrated = pvalue < 0.05
                        
                        if require_cointegration and not is_cointegrated:
                            continue
                        
                        # Calculate hedge ratio using OLS
                        X = sm.add_constant(stock2_data)
                        model = sm.OLS(stock1_data, X).fit()
                        hedge_ratio = model.params[1]
                        
                        # Calculate spread and z-score
                        spread = stock1_data - hedge_ratio * stock2_data
                        mean = spread.mean()
                        std = spread.std()
                        z_score = (spread.iloc[-1] - mean) / std
                        
                        # Current prices
                        stock1_price = df[stock1].iloc[-1]
                        stock2_price = df[stock2].iloc[-1]
                        
                        # Determine trading signal
                        signal = "NO SIGNAL"
                        position_type = ""
                        
                        if z_score > min_z_score:
                            signal = f"Sell {st.session_state.nifty_stocks[stock1]['name']} / Buy {st.session_state.nifty_stocks[stock2]['name']}"
                            position_type = "SELL_BUY"
                        elif z_score < -min_z_score:
                            signal = f"Buy {st.session_state.nifty_stocks[stock1]['name']} / Sell {st.session_state.nifty_stocks[stock2]['name']}"
                            position_type = "BUY_SELL"
                        
                        # Calculate stop losses
                        if position_type == "SELL_BUY":
                            stock1_sl = calculate_stop_loss(df[stock1], "SELL", stock1_price)
                            stock2_sl = calculate_stop_loss(df[stock2], "BUY", stock2_price)
                        elif position_type == "BUY_SELL":
                            stock1_sl = calculate_stop_loss(df[stock1], "BUY", stock1_price)
                            stock2_sl = calculate_stop_loss(df[stock2], "SELL", stock2_price)
                        else:
                            stock1_sl = 0
                            stock2_sl = 0
                        
                        # Calculate target exit z-score (50% reversion to mean)
                        target_z = z_score * 0.5
                        
                        pair_data = {
                            'pair_id': f"{stock1.split('.')[0]}_{stock2.split('.')[0]}",
                            'sector': sector,
                            'stock1': st.session_state.nifty_stocks[stock1]['name'],
                            'stock2': st.session_state.nifty_stocks[stock2]['name'],
                            'stock1_symbol': stock1,
                            'stock2_symbol': stock2,
                            'correlation': round(correlation, 2),
                            'is_cointegrated': is_cointegrated,
                            'cointegration_strength': 'Yes' if is_cointegrated else 'Weak',
                            'z_score': round(z_score, 2),
                            'hedge_ratio': round(hedge_ratio, 4),
                            'signal': signal,
                            'position_type': position_type,
                            'stock1_price': round(stock1_price, 2),
                            'stock2_price': round(stock2_price, 2),
                            'stock1_sl': stock1_sl,
                            'stock2_sl': stock2_sl,
                            'target_exit_z': round(target_z, 2)
                        }
                        
                        results.append(pair_data)
                    except Exception as e:
                        pass  # Silently skip pairs that fail analysis
            
            # Update progress
            progress_bar.progress(sector_idx / total_sectors)
        
        # Sort results by absolute z-score (descending)
        results.sort(key=lambda x: abs(x['z_score']), reverse=True)
        
        # Complete progress
        progress_bar.progress(1.0)
        # Remove progress bar
        progress_bar.empty()
    
    return results

def display_pair_analysis(pair):
    """Display analysis for a selected pair"""
    stock1_symbol = pair['stock1_symbol']
    stock2_symbol = pair['stock2_symbol']
    
    try:
        # Get stock data
        df = st.session_state.stock_data
        if df is None or df.empty or stock1_symbol not in df.columns or stock2_symbol not in df.columns:
            st.error(f"Data not available for {stock1_symbol} and {stock2_symbol}")
            return
        
        stock1_data = df[stock1_symbol].dropna()
        stock2_data = df[stock2_symbol].dropna()
        
        # Align the indexes
        common_idx = stock1_data.index.intersection(stock2_data.index)
        stock1_data = stock1_data.loc[common_idx]
        stock2_data = stock2_data.loc[common_idx]
        
        # Calculate hedge ratio using OLS
        X = sm.add_constant(stock2_data)
        model = sm.OLS(stock1_data, X).fit()
        hedge_ratio = model.params[1]
        
        # Calculate spread and z-score
        spread = stock1_data - hedge_ratio * stock2_data
        mean = spread.mean()
        std = spread.std()
        z_scores = (spread - mean) / std
        
        # Calculate rolling correlation (30 days)
        rolling_corr = stock1_data.rolling(window=30).corr(stock2_data)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Normalized Price Comparison', 'Z-Score', 'Rolling 30-Day Correlation'),
            vertical_spacing=0.1,
            specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "scatter"}]]
        )
        
        # Normalize data for price comparison
        stock1_norm = stock1_data / stock1_data.iloc[0]
        stock2_norm = stock2_data / stock2_data.iloc[0]
        
        # Add price traces
        fig.add_trace(
            go.Scatter(
                x=common_idx, 
                y=stock1_norm, 
                name=f"{pair['stock1']} (Normalized)",
                line=dict(color='#4b8bbe')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=common_idx, 
                y=stock2_norm, 
                name=f"{pair['stock2']} (Normalized)",
                line=dict(color='#ffbb00')
            ),
            row=1, col=1
        )
        
        # Add z-score trace
        fig.add_trace(
            go.Scatter(
                x=common_idx, 
                y=z_scores, 
                name='Z-Score',
                line=dict(color='#ff5050')
            ),
            row=2, col=1
        )
        
        # Add threshold lines
        fig.add_trace(
            go.Scatter(
                x=common_idx, 
                y=[2] * len(common_idx), 
                name='Upper Threshold (+2)',
                line=dict(color='red', width=1, dash='dash')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=common_idx, 
                y=[-2] * len(common_idx), 
                name='Lower Threshold (-2)',
                line=dict(color='green', width=1, dash='dash')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=common_idx, 
                y=[0] * len(common_idx), 
                name='Mean (0)',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=2, col=1
        )
        
        # Add correlation trace
        fig.add_trace(
            go.Scatter(
                x=rolling_corr.dropna().index, 
                y=rolling_corr.dropna(), 
                name='Rolling Correlation',
                line=dict(color='#5d9eeb')
            ),
            row=3, col=1
        )
        
        # Add correlation threshold
        fig.add_trace(
            go.Scatter(
                x=rolling_corr.dropna().index, 
                y=[0.7] * len(rolling_corr.dropna()), 
                name='Correlation Threshold (0.7)',
                line=dict(color='yellow', width=1, dash='dash')
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800, 
            title_text=f"Analysis: {pair['stock1']} vs {pair['stock2']}",
            template='plotly_dark',
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Update y-axis ranges
        fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
        fig.update_yaxes(title_text="Z-Score", row=2, col=1)
        fig.update_yaxes(title_text="Correlation", range=[-1, 1], row=3, col=1)
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display statistics and trade form
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pair Statistics")
            stats_df = pd.DataFrame({
                "Metric": [
                    "Hedge Ratio",
                    "Correlation",
                    "Cointegration",
                    "Current Z-Score",
                    "Stock 1 Current Price",
                    "Stock 2 Current Price"
                ],
                "Value": [
                    f"{hedge_ratio:.4f}",
                    f"{pair['correlation']:.2f}",
                    pair['cointegration_strength'],
                    f"{pair['z_score']:.2f}",
                    f"₹{pair['stock1_price']:.2f}",
                    f"₹{pair['stock2_price']:.2f}"
                ]
            })
            st.table(stats_df)
        
        with col2:
            st.subheader("Trading Signal")
            
            signal_color = "gray"
            if pair['z_score'] > 2:
                signal_color = "red"
            elif pair['z_score'] < -2:
                signal_color = "green"
            
            st.markdown(f"<h3 style='color:{signal_color};'>{pair['signal']}</h3>", unsafe_allow_html=True)
            
            if pair['position_type']:
                st.subheader("Stop Losses")
                sl_df = pd.DataFrame({
                    "Stock": [pair['stock1'], pair['stock2']],
                    "Stop Loss": [f"₹{pair['stock1_sl']:.2f}", f"₹{pair['stock2_sl']:.2f}"]
                })
                st.table(sl_df)
                
                # Trade button
                if st.button("Trade This Pair"):
                    st.session_state.selected_pair = pair
                    st.session_state.show_trade_form = True
                    st.experimental_rerun()
        
    except Exception as e:
        st.error(f"Error creating analysis: {str(e)}")

def show_trade_form(pair):
    """Display trade entry form for selected pair"""
    st.subheader("Enter Trade Details")
    
    with st.form(key="trade_form", clear_on_submit=False):
        st.markdown(f"**Trading {pair['stock1']} / {pair['stock2']}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if pair['position_type'] == 'BUY_SELL':
                st.markdown(f"**BUY {pair['stock1']}**")
                action1 = "BUY"
            else:
                st.markdown(f"**SELL {pair['stock1']}**")
                action1 = "SELL"
            
            price1 = st.number_input(f"{pair['stock1']} Price", 
                                     value=float(pair['stock1_price']), 
                                     step=0.05,
                                     format="%.2f")
            
            qty1 = st.number_input(f"{pair['stock1']} Quantity", 
                                   value=int(100000/price1), 
                                   step=1)
            
            if action1 == "BUY":
                sl1 = st.number_input(f"{pair['stock1']} Stop Loss", 
                                     value=float(pair['stock1_sl']), 
                                     step=0.05,
                                     format="%.2f")
            else:
                sl1 = st.number_input(f"{pair['stock1']} Stop Loss", 
                                     value=float(pair['stock1_sl']), 
                                     step=0.05,
                                     format="%.2f")
        
        with col2:
            if pair['position_type'] == 'BUY_SELL':
                st.markdown(f"**SELL {pair['stock2']}**")
                action2 = "SELL"
            else:
                st.markdown(f"**BUY {pair['stock2']}**")
                action2 = "BUY"
            
            price2 = st.number_input(f"{pair['stock2']} Price", 
                                     value=float(pair['stock2_price']), 
                                     step=0.05,
                                     format="%.2f")
            
            # Calculate qty2 based on hedge ratio for equal weighting
            suggested_qty2 = int((qty1 * price1) / (price2 * pair['hedge_ratio']))
            qty2 = st.number_input(f"{pair['stock2']} Quantity", 
                                   value=suggested_qty2, 
                                   step=1)
            
            if action2 == "BUY":
                sl2 = st.number_input(f"{pair['stock2']} Stop Loss", 
                                     value=float(pair['stock2_sl']), 
                                     step=0.05,
                                     format="%.2f")
            else:
                sl2 = st.number_input(f"{pair['stock2']} Stop Loss", 
                                     value=float(pair['stock2_sl']), 
                                     step=0.05,
                                     format="%.2f")
        
        # Calculate notional values and display
        notional1 = qty1 * price1
        notional2 = qty2 * price2
        
        st.markdown(f"**Notional Values**: {pair['stock1']}: ₹{notional1:,.2f} | {pair['stock2']}: ₹{notional2:,.2f}")
        st.markdown(f"**Ratio**: {notional1/notional2:.2f}:1")
        
        submitted = st.form_submit_button("Submit Trade")
        cancel = st.form_submit_button("Cancel")
        
        if submitted:
            # Create position object
            position = {
                'id': str(uuid.uuid4()),
                'entry_time': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
                'stock1_symbol': pair['stock1_symbol'],
                'stock2_symbol': pair['stock2_symbol'],
                'stock1': pair['stock1'],
                'stock2': pair['stock2'],
                'position_type': pair['position_type'],
                'entry_price1': price1,
                'entry_price2': price2,
                'quantity1': qty1,
                'quantity2': qty2,
                'stop_loss1': sl1,
                'stop_loss2': sl2,
                'hedge_ratio': pair['hedge_ratio'],
                'target_exit_z': 0.5,  # Target 50% reversion to mean
                'sector': pair['sector']
            }
            
            # Add position to session state
            st.session_state.positions.append(position)
            
            # Reset selected pair and hide form
            st.session_state.selected_pair = None
            st.session_state.show_trade_form = False
            
            st.success(f"Position added successfully!")
            st.experimental_rerun()
        
        if cancel:
            st.session_state.selected_pair = None
            st.session_state.show_trade_form = False
            st.experimental_rerun()

update_positions():
    """Update positions with current prices and status"""
    if not st.session_state.positions:
        return []
    
    updated_positions = []
    
    with st.spinner('Updating positions...'):
        # Get current data
        df = st.session_state.stock_data
        now = datetime.now(IST)
        
        for position in st.session_state.positions:
            updated = position.copy()
            
            stock1_symbol = position['stock1_symbol']
            stock2_symbol = position['stock2_symbol']
            
            if df is not None and not df.empty and stock1_symbol in df.columns and stock2_symbol in df.columns:
                # Get current prices
                current_price1 = df[stock1_symbol].iloc[-1]
                current_price2 = df[stock2_symbol].iloc[-1]
                
                updated['current_price1'] = round(current_price1, 2)
                updated['current_price2'] = round(current_price2, 2)
                
                # Calculate P&L
                if position['position_type'] == 'BUY_SELL':
                    pnl1 = (current_price1 - position['entry_price1']) / position['entry_price1'] * 100
                    pnl2 = (position['entry_price2'] - current_price2) / position['entry_price2'] * 100
                else:  # SELL_BUY
                    pnl1 = (position['entry_price1'] - current_price1) / position['entry_price1'] * 100
                    pnl2 = (current_price2 - position['entry_price2']) / position['entry_price2'] * 100
                
                updated['pnl1'] = round(pnl1, 2)
                updated['pnl2'] = round(pnl2, 2)
                updated['total_pnl'] = round(pnl1 + pnl2, 2)
                
                # Calculate current z-score
                stock1_data = df[stock1_symbol].dropna()
                stock2_data = df[stock2_symbol].dropna()
                
                # Align the indexes
                common_idx = stock1_data.index.intersection(stock2_data.index)
                stock1_data = stock1_data.loc[common_idx]
                stock2_data = stock2_data.loc[common_idx]
                
                # Calculate spread and z-score
                spread = stock1_data - position['hedge_ratio'] * stock2_data
                mean = spread.mean()
                std = spread.std()
                current_z_score = (spread.iloc[-1] - mean) / std
                
                updated['current_z_score'] = round(current_z_score, 2)
                
                # Determine exit signal
                updated['stop_loss_triggered'] = False
                updated['exit_signal'] = 'HOLD'
                
                # Check for stop loss hit
                if position['position_type'] == 'BUY_SELL':
                    if current_price1 < position['stop_loss1'] or current_price2 > position['stop_loss2']:
                        updated['stop_loss_triggered'] = True
                        updated['exit_signal'] = 'CLOSE (Stop Loss Hit)'
                    elif abs(current_z_score) < 0.5:
                        updated['exit_signal'] = 'CLOSE (Target Reached)'
                else:  # SELL_BUY
                    if current_price1 > position['stop_loss1'] or current_price2 < position['stop_loss2']:
                        updated['stop_loss_triggered'] = True
                        updated['exit_signal'] = 'CLOSE (Stop Loss Hit)'
                    elif abs(current_z_score) < 0.5:
                        updated['exit_signal'] = 'CLOSE (Target Reached)'
                
                # Calculate days in trade
                entry_date = datetime.strptime(position['entry_time'].split()[0], '%Y-%m-%d')
                days_in_trade = (now.date() - entry_date.date()).days
                updated['days_in_trade'] = days_in_trade
            
            updated_positions.append(updated)
    
    return updated_positions


def close_position(position_id):
    """Close a position"""
    st.session_state.positions = [p for p in st.session_state.positions if p['id'] != position_id]
    st.success("Position closed successfully")

def display_positions(positions):
    """Display active positions in a formatted way"""
    if not positions:
        st.info("No active positions")
        return
    
    for position in positions:
        with st.container():
            cols = st.columns([3, 3, 2, 2])
            
            with cols[0]:
                st.markdown(f"### {position['stock1']} / {position['stock2']}")
                st.markdown(f"**Sector:** {position['sector']}")
                st.markdown(f"**Entry Date:** {position['entry_time'].split()[0]}")
                if 'days_in_trade' in position:
                    st.markdown(f"**Days in Trade:** {position['days_in_trade']}")
            
            with cols[1]:
                if position['position_type'] == 'BUY_SELL':
                    leg1 = f"🟢 **BUY** {position['stock1']}"
                    leg2 = f"🔴 **SELL** {position['stock2']}"
                else:
                    leg1 = f"🔴 **SELL** {position['stock1']}"
                    leg2 = f"🟢 **BUY** {position['stock2']}"
                
                st.markdown(leg1)
                st.markdown(f"Entry: ₹{position['entry_price1']:.2f} | Current: ₹{position.get('current_price1', 0):.2f}")
                st.markdown(f"Stop Loss: ₹{position['stop_loss1']:.2f} | Qty: {position['quantity1']}")
                
                st.markdown(leg2)
                st.markdown(f"Entry: ₹{position['entry_price2']:.2f} | Current: ₹{position.get('current_price2', 0):.2f}")
                st.markdown(f"Stop Loss: ₹{position['stop_loss2']:.2f} | Qty: {position['quantity2']}")
            
            with cols[2]:
                st.markdown("### Performance")
                
                # PnL display
                if 'total_pnl' in position:
                    pnl_color = "green" if position['total_pnl'] >= 0 else "red"
                    st.markdown(f"**Total P&L:** <span style='color:{pnl_color};'>{position['total_pnl']:.2f}%</span>", unsafe_allow_html=True)
                    
                    st.markdown(f"**{position['stock1']} P&L:** <span style='color:{'green' if position.get('pnl1', 0) >= 0 else 'red'};'>{position.get('pnl1', 0):.2f}%</span>", unsafe_allow_html=True)
                    st.markdown(f"**{position['stock2']} P&L:** <span style='color:{'green' if position.get('pnl2', 0) >= 0 else 'red'};'>{position.get('pnl2', 0):.2f}%</span>", unsafe_allow_html=True)
                
                # Z-score display
                if 'current_z_score' in position:
                    z_color = "red" if position['current_z_score'] > 1 else ("green" if position['current_z_score'] < -1 else "white")
                    st.markdown(f"**Current Z-Score:** <span style='color:{z_color};'>{position['current_z_score']:.2f}</span>", unsafe_allow_html=True)
            
            with cols[3]:
                st.markdown("### Action")
                
                # Exit signal
                if 'exit_signal' in position:
                    signal_color = "red" if position['exit_signal'].startswith('CLOSE') else "yellow"
                    st.markdown(f"**Signal:** <span style='color:{signal_color};'>{position['exit_signal']}</span>", unsafe_allow_html=True)
                
                # Close button
                if st.button("Close Position", key=f"close_{position['id']}"):
                    close_position(position['id'])
                    st.experimental_rerun()
                
                # Analyze button
                if st.button("Analyze Pair", key=f"analyze_{position['id']}"):
                    # Find pair data
                    pair = None
                    for p in st.session_state.pairs:
                        if p['stock1_symbol'] == position['stock1_symbol'] and p['stock2_symbol'] == position['stock2_symbol']:
                            pair = p
                            break
                    
                    if pair:
                        st.session_state.selected_pair = pair
                        st.experimental_rerun()
                    else:
                        # Create a simple pair structure if not found in current pairs
                        simple_pair = {
                            'stock1_symbol': position['stock1_symbol'],
                            'stock2_symbol': position['stock2_symbol'],
                            'stock1': position['stock1'],
                            'stock2': position['stock2'],
                            'sector': position['sector'],
                            'position_type': position['position_type'],
                            'hedge_ratio': position['hedge_ratio']
                        }
                        st.session_state.selected_pair = simple_pair
                        st.experimental_rerun()
            
            st.markdown("---")

def export_positions():
    """Export positions to CSV"""
    if not st.session_state.positions:
        st.warning("No positions to export")
        return
    
    updated_positions = update_positions()
    
    # Convert to DataFrame
    df = pd.DataFrame(updated_positions)
    
    # Select columns to export
    if len(df.columns) > 0:
        cols = ['id', 'entry_time', 'stock1', 'stock2', 'position_type', 
                'entry_price1', 'entry_price2', 'current_price1', 'current_price2',
                'quantity1', 'quantity2', 'stop_loss1', 'stop_loss2', 
                'pnl1', 'pnl2', 'total_pnl', 'current_z_score', 'exit_signal']
        
        # Keep only columns that exist
        export_cols = [col for col in cols if col in df.columns]
        df = df[export_cols]
    
    # Convert to CSV
    csv = df.to_csv(index=False)
    
    # Create download button
    st.download_button(
        label="Download Positions CSV",
        data=csv,
        file_name=f"pairs_trading_positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def export_pairs():
    """Export pairs analysis to CSV"""
    if not st.session_state.pairs:
        st.warning("No pairs to export")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.pairs)
    
    # Convert to CSV
    csv = df.to_csv(index=False)
    
    # Create download button
    st.download_button(
        label="Download Pairs Analysis CSV",
        data=csv,
        file_name=f"pairs_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Main app layout
def main():
    st.title("Shashank Udupa Pairs Trading App")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Stock limit slider
        st.session_state.stock_limit = st.slider(
            "Number of stocks to analyze", 
            min_value=20, 
            max_value=500, 
            value=st.session_state.stock_limit,
            step=20,
            help="Higher values will analyze more stocks but take longer"
        )
        
        # Data refresh
        if st.button("🔄 Force Data Refresh"):
            with st.spinner("Refreshing data..."):
                fetch_stock_data(force_refresh=True)
                st.success("Data refreshed successfully")
        
        # Export options
        st.header("Export Data")
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("Export Positions"):
                export_positions()
        
        with export_col2:
            if st.button("Export Pairs"):
                export_pairs()
        
        # About section
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This app identifies statistical pairs trading opportunities in the NSE 500 index.")
        st.markdown("Created by Shashank Udupa")
        
        # Last updated time
        if st.session_state.last_updated:
            st.markdown(f"Data last updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if we should show trade form for a selected pair
    if st.session_state.show_trade_form and st.session_state.selected_pair:
        show_trade_form(st.session_state.selected_pair)
        return
    
    # Tabs for different sections
    tab1, tab2 = st.tabs(["Pairs Screener", "Active Positions"])
    
    with tab1:
        # Header section with refresh button and last updated time
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.subheader("NSE Pairs Screener")
        with col2:
            market_hours = datetime.now(IST).hour >= 9 and datetime.now(IST).hour < 16
            market_day = datetime.now(IST).weekday() < 5  # Monday to Friday
            market_status = "OPEN" if market_hours and market_day else "CLOSED"
            st.markdown(f"Market: <span style='color:{'green' if market_status == 'OPEN' else 'red'};'>{market_status}</span>", unsafe_allow_html=True)
        with col3:
            if st.button("🔄 Refresh Pairs"):
                with st.spinner("Analyzing pairs..."):
                    st.session_state.pairs = analyze_pairs()
                    st.success("Pairs analysis updated")
        
        # Filter section
        st.subheader("Filter Pairs")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # Get unique sectors
            sectors = set()
            for _, details in st.session_state.nifty_stocks.items():
                sectors.add(details['sector'])
            sector_options = ['All'] + sorted(list(sectors))
            selected_sector = st.selectbox("Sector", sector_options)
        
        with col2:
            min_correlation = st.slider("Min Correlation", 0.0, 1.0, 0.7, 0.05)
        
        with col3:
            min_z_score = st.slider("Min Z-Score", 0.0, 4.0, 2.0, 0.1)
        
        with col4:
            require_cointegration = st.checkbox("Require Cointegration", value=True)
        
        if st.button("Apply Filters"):
            with st.spinner("Analyzing pairs..."):
                st.session_state.pairs = analyze_pairs(
                    min_correlation=min_correlation,
                    min_z_score=min_z_score,
                    selected_sector=selected_sector,
                    require_cointegration=require_cointegration
                )
        
        # Pairs table
        if not st.session_state.pairs:
            if st.session_state.stock_data is None:
                with st.spinner("Fetching initial data..."):
                    fetch_stock_data()
                    
            st.session_state.pairs = analyze_pairs(
                min_correlation=min_correlation,
                min_z_score=min_z_score,
                selected_sector=selected_sector,
                require_cointegration=require_cointegration
            )
        
        if st.session_state.pairs:
            # Filter pairs based on current criteria
            filtered_pairs = []
            for pair in st.session_state.pairs:
                if abs(pair['z_score']) < min_z_score:
                    continue
                if selected_sector != 'All' and pair['sector'] != selected_sector:
                    continue
                if require_cointegration and not pair['is_cointegrated']:
                    continue
                filtered_pairs.append(pair)
            
            if filtered_pairs:
                st.subheader(f"Showing {len(filtered_pairs)} pairs")
                
                # Create DataFrame for display
                pairs_df = pd.DataFrame([
                    {
                        'Pair': f"{p['stock1']} / {p['stock2']}",
                        'Sector': p['sector'],
                        'Correlation': p['correlation'],
                        'Cointegrated': '✅' if p['is_cointegrated'] else '❌',
                        'Z-Score': p['z_score'],
                        'Signal': p['signal'],
                        'Action': f"<button class='btn-analyze' onclick=\"alert('Use Streamlit buttons below')\">Analyze</button>"
                    }
                    for p in filtered_pairs
                ])
                
                # Apply styler for Z-score coloring
                def color_z_score(val):
                    if val > 2:
                        return 'color: red; font-weight: bold'
                    elif val < -2:
                        return 'color: green; font-weight: bold'
                    return ''
                
                # Display table
                st.dataframe(
                    pairs_df.style.applymap(color_z_score, subset=['Z-Score']),
                    hide_index=True,
                    column_config={
                        'Action': st.column_config.Column(
                            'Action',
                            width='small',
                            help='Click to analyze'
                        )
                    }
                )
                
                # Add selection
                st.subheader("Select Pair to Analyze")
                
                # Create selectbox for pairs
                pair_options = [f"{p['stock1']} / {p['stock2']} (Z-Score: {p['z_score']})" for p in filtered_pairs]
                selected_pair_idx = st.selectbox("Select a pair", range(len(pair_options)), format_func=lambda x: pair_options[x])
                
                if st.button("Analyze Selected Pair"):
                    st.session_state.selected_pair = filtered_pairs[selected_pair_idx]
                    st.experimental_rerun()
            else:
                st.info("No pairs match the current filter criteria")
        else:
            st.info("No pairs found. Try adjusting filter criteria or increasing the number of stocks to analyze.")
    
    with tab2:
        st.subheader("Active Positions")
        
        # Add position button
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("➕ Add Position"):
                if st.session_state.pairs:
                    # Select first pair by default
                    st.session_state.selected_pair = st.session_state.pairs[0]
                    st.session_state.show_trade_form = True
                    st.experimental_rerun()
                else:
                    st.error("No pairs available. Please run pairs analysis first.")
        
        # Update positions
        positions = update_positions()
        
        # Display positions
        display_positions(positions)
    
    # If a pair is selected for analysis, show analysis
    if st.session_state.selected_pair and not st.session_state.show_trade_form:
        st.subheader(f"Analysis: {st.session_state.selected_pair['stock1']} / {st.session_state.selected_pair['stock2']}")
        display_pair_analysis(st.session_state.selected_pair)

if __name__ == "__main__":
    main()

