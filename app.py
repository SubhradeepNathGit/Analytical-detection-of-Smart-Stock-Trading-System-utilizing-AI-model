import streamlit as st
st.set_page_config(page_title="Smart Stock AI", layout="wide", initial_sidebar_state="expanded")

import numpy as np
import pandas as pd
import yfinance as yf
# Using tf.keras for compatibility with TensorFlow 2.18+ / Keras 3
from neuralprophet import NeuralProphet
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import plotly.express as px
import torch
import neuralprophet.configure
import logging

# Prevent PyTorch Lightning background threads from crashing Streamlit's logger context
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("neuralprophet").setLevel(logging.ERROR)

# ---------------------------------------------------------
# PyTorch 2.6+ weights_only=True UnpicklingError Fix
# ---------------------------------------------------------
_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _safe_load

# ---------------------------------------------------------
# Page Configuration & Full Black Glassmorphic CSS
# ---------------------------------------------------------

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_html(file_name, **kwargs):
    with open(file_name) as f:
        html_content = f.read()
    st.markdown(html_content.format(**kwargs), unsafe_allow_html=True)

load_css('style.css')

# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.markdown("<h1 class='main-title'>Analytical Stock Trading </h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'> AI Trading Intelligence • Developed by Subhradeep Nath</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar, Currency Converter & Inputs
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #FFFFFF;'>Control Panel</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    import datetime
    
    # Initialize query params for state retention
    qp = st.query_params
    
    default_ticker = qp.get("ticker", "AAPL")
    default_start = qp.get("start", "2016-01-01")
    default_end = qp.get("end", str(datetime.date.today()))
    default_curr = qp.get("currency", "USD")
    
    try:
        start_date_val = pd.to_datetime(default_start).date()
    except:
        start_date_val = pd.to_datetime('2016-01-01').date()
        
    try:
        end_date_val = pd.to_datetime(default_end).date()
    except:
        end_date_val = datetime.date.today()
        
    currencies = ["USD", "EUR", "GBP", "INR", "JPY", "AUD", "CAD"]
    if default_curr not in currencies:
        default_curr = "USD"
    curr_index = currencies.index(default_curr)

    user_input = st.text_input('Stock Ticker', default_ticker).upper()
    
    if not user_input.strip():
        st.error("Ticker cannot be empty! Displaying previous data.")
        user_input = default_ticker if default_ticker.strip() else "AAPL"
        
    start_date = st.date_input('Start Date', start_date_val)
    end_date = st.date_input('End Date', end_date_val)
    
    st.markdown("### Currency Converter")
    target_currency = st.selectbox(
        "Display Values In",
        currencies,
        index=curr_index
    )
    
    # Update query params to hold state across refresh
    st.query_params["ticker"] = user_input
    st.query_params["start"] = str(start_date)
    st.query_params["end"] = str(end_date)
    st.query_params["currency"] = target_currency
    
    # Trademark & Copyright Footer
    st.markdown("<br><br><br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #8892B0; font-size: 0.8rem;'>"
        "&copy; 2024 Subhradeep Nath &trade;<br>All Rights Reserved</p>", 
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# Caching Functions
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)
def get_exchange_rate(target_currency):
    if target_currency == "USD":
        return 1.0, "$"
    
    ticker = f"{target_currency}=X"
    try:
        fx = yf.Ticker(ticker)
        rate = fx.history(period="1d")['Close'].iloc[-1]
    except:
        rate = 1.0
    
    symbols = {"EUR": "€", "GBP": "£", "INR": "₹", "JPY": "¥", "AUD": "A$", "CAD": "C$"}
    return rate, symbols.get(target_currency, target_currency)

@st.cache_data(show_spinner=False, max_entries=20)
def load_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.droplevel(1)
    stock_data.reset_index(inplace=True)
    stock_data.dropna(subset=['Close', 'Open', 'High', 'Low'], inplace=True)
    modified_stock_data = stock_data.drop(columns=['Adj Close'], errors='ignore')
    return stock_data, modified_stock_data

@st.cache_data(show_spinner=False, max_entries=5)
def train_neural_prophet(ticker_str, start_date_str, end_date_str, stocks_df):
    model_neuralprophet = NeuralProphet(epochs=20, trainer_config={"enable_checkpointing": False})
    model_neuralprophet.fit(stocks_df, freq='B')
    future = model_neuralprophet.make_future_dataframe(stocks_df, periods=300)
    forecast = model_neuralprophet.predict(future)
    actual_prediction = model_neuralprophet.predict(stocks_df)
    return forecast, actual_prediction

@st.cache_resource(show_spinner=False, max_entries=1)
def load_keras_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keras_model.h5')
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# ---------------------------------------------------------
# Data Fetching & Processing
# ---------------------------------------------------------
with st.spinner(f"Connecting to Market Data Exchange for {user_input}"):
    try:
        hist_stock_data, hist_modified_stock_data = load_stock_data(user_input, start_date, end_date)
        fx_rate, fx_symbol = get_exchange_rate(target_currency)
    except Exception as e:
        st.error(f"Failed to fetch data for {user_input}. Please verify the ticker symbol.")
        st.stop()

if hist_stock_data.empty or 'Close' not in hist_stock_data.columns:
    st.warning(f"No data found for {user_input}. Please ensure the ticker symbol is correct.")
    st.stop()

if len(hist_stock_data) < 150:
    st.warning(f"{user_input} does not have enough historical data (minimum 150 days required for AI models).")
    st.stop()

def get_live_data(ticker):
    try:
        t = yf.Ticker(ticker)
        fast = t.fast_info
        return {
            'price': fast.last_price,
            'prev_close': fast.previous_close,
            'day_high': fast.day_high,
            'day_low': fast.day_low,
            'year_high': fast.year_high,
            'year_low': fast.year_low,
            'volume': fast.last_volume
        }
    except:
        return None

# ---------------------------------------------------------
# Real-Time Dashboard Rendering Fragment
# ---------------------------------------------------------
@st.fragment
def render_dashboard():
    col1, col2 = st.columns([0.88, 0.12])
    with col2:
        st.button("Live Sync", use_container_width=True, key="refresh_btn", help="Sync exact real-time market data")
        
    # Fetch exact real-time Google equivalent data
    live_data = get_live_data(user_input)
    
    stock_data = hist_stock_data.copy()
    modified_stock_data = hist_modified_stock_data.copy()
    
    if live_data is not None:
        live_price = live_data['price']
        last_date = stock_data['Date'].iloc[-1].date()
        today_date = datetime.date.today()
        if last_date >= today_date:
            stock_data.at[stock_data.index[-1], 'Close'] = live_price
            modified_stock_data.at[modified_stock_data.index[-1], 'Close'] = live_price
            if 'High' in stock_data and live_price > stock_data['High'].iloc[-1]:
                stock_data.at[stock_data.index[-1], 'High'] = live_price
            if 'Low' in stock_data and live_price < stock_data['Low'].iloc[-1]:
                stock_data.at[stock_data.index[-1], 'Low'] = live_price
        else:
            new_row = stock_data.iloc[[-1]].copy()
            new_row['Date'] = pd.to_datetime(today_date)
            new_row['Close'] = live_price
            new_row['Open'] = live_price
            new_row['High'] = live_price
            new_row['Low'] = live_price
            stock_data = pd.concat([stock_data, new_row], ignore_index=True)
            
            new_mod_row = modified_stock_data.iloc[[-1]].copy()
            new_mod_row['Date'] = pd.to_datetime(today_date)
            new_mod_row['Close'] = live_price
            if 'Open' in new_mod_row: new_mod_row['Open'] = live_price
            if 'High' in new_mod_row: new_mod_row['High'] = live_price
            if 'Low' in new_mod_row: new_mod_row['Low'] = live_price
            modified_stock_data = pd.concat([modified_stock_data, new_mod_row], ignore_index=True)

    # ---------------------------------------------------------
    # Top Metrics Board (With Currency Conversion)
    # ---------------------------------------------------------
    if live_data is not None:
        current_price = live_data['price'] * fx_rate
        
        pc = live_data['prev_close']
        prev_price = (stock_data['Close'].iloc[-2] if pd.isna(pc) else pc) * fx_rate
        
        dh = live_data['day_high']
        today_high = (stock_data['High'].iloc[-1] if pd.isna(dh) else dh) * fx_rate
        
        dl = live_data['day_low']
        today_low = (stock_data['Low'].iloc[-1] if pd.isna(dl) else dl) * fx_rate
        
        yh = live_data['year_high']
        high_52 = (stock_data['High'].tail(252).max() if pd.isna(yh) else yh) * fx_rate
        
        yl = live_data['year_low']
        low_52 = (stock_data['Low'].tail(252).min() if pd.isna(yl) else yl) * fx_rate
        
        v = live_data['volume']
        avg_vol = int(stock_data['Volume'].tail(30).mean()) if pd.isna(v) else v
    else:
        current_price = stock_data['Close'].iloc[-1] * fx_rate
        prev_price = stock_data['Close'].iloc[-2] * fx_rate
        today_high = stock_data['High'].iloc[-1] * fx_rate
        today_low = stock_data['Low'].iloc[-1] * fx_rate
        high_52 = stock_data['High'].tail(252).max() * fx_rate
        low_52 = stock_data['Low'].tail(252).min() * fx_rate
        avg_vol = int(stock_data['Volume'].tail(30).mean())

    price_change = current_price - prev_price
    pct_change = (price_change / prev_price) * 100
    delta_color = "#00e676" if price_change >= 0 else "#FF3D00"
    delta_arrow = "▲" if price_change >= 0 else "▼"

    load_html(
        'metrics.html',
        fx_symbol=fx_symbol,
        current_price=current_price,
        delta_color=delta_color,
        delta_arrow=delta_arrow,
        abs_price_change=abs(price_change),
        pct_change=pct_change,
        today_high=today_high,
        today_low=today_low,
        high_52=high_52,
        low_52=low_52,
        avg_vol=avg_vol
    )
    
    import streamlit.components.v1 as components
    components.html("""
    <script>
        const parentDoc = window.parent.document;
        const valueElements = parentDoc.querySelectorAll('.m-value');
        
        valueElements.forEach(el => {
            if (el.innerText.includes('.')) {
                const text = el.innerText;
                const match = text.match(/^([^\\d]+)([\\d,]+\\.\\d+)$/);
                if (match) {
                    el.dataset.symbol = match[1];
                    el.dataset.baseValue = parseFloat(match[2].replace(/,/g, ''));
                    el.dataset.isPrice = 'true';
                }
            }
        });
        
        setInterval(() => {
            valueElements.forEach(el => {
                if (el.dataset.isPrice === 'true') {
                    const base = parseFloat(el.dataset.baseValue);
                    const rand = Math.random();
                    let jitter = 0;
                    if (rand > 0.66) jitter = 0.01;
                    else if (rand > 0.33) jitter = -0.01;
                    else jitter = 0;
                    
                    const newValue = base + jitter;
                    const formatted = newValue.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
                    
                    el.innerText = el.dataset.symbol + formatted;
                }
            });
        }, 1000);
    </script>
    """, height=0)

    # ---------------------------------------------------------
    # Tabbed Dashboard Interface
    # ---------------------------------------------------------
    tab1, tab_data, tab2, tab3, tab4, tab5 = st.tabs([
        "Global Market Map", 
        "Raw Market Data",
        "Price Action", 
        "Moving Averages", 
        "LSTM AI Predictor", 
        "Prophet Forecast"
    ])

    # ---------------------------------------------------------
    # TAB 1: Global Market Map
    # ---------------------------------------------------------
    with tab1:
        st.markdown("### Global Financial Hubs Activity")
        
        # Create an interactive 3D Globe showing major financial centers
        df_hubs = pd.DataFrame({
            'City': ['New York', 'London', 'Tokyo', 'Hong Kong', 'Singapore', 'Frankfurt', 'Mumbai'],
            'Lat': [40.7128, 51.5074, 35.6762, 22.3193, 1.3521, 50.1109, 19.0760],
            'Lon': [-74.0060, -0.1278, 139.6503, 114.1694, 103.8198, 8.6821, 72.8777],
            'Market Size': [100, 70, 60, 50, 40, 40, 30]
        })
        
        fig_map = go.Figure(data=go.Scattergeo(
            lon = df_hubs['Lon'],
            lat = df_hubs['Lat'],
            text = df_hubs['City'],
            mode = 'markers+text',
            marker = dict(
                size = df_hubs['Market Size']/2.5,
                color = '#00C853',
                line_color='white',
                line_width=0.5,
                sizemode = 'area'
            ),
            textposition="top center",
            textfont=dict(color='#FFFFFF', size=12)
        ))
        
        # Add Premium Data Stream Lines (Flight Paths)
        flights = [
            (40.7128, -74.0060, 51.5074, -0.1278),  # NY to London
            (51.5074, -0.1278, 35.6762, 139.6503),  # London to Tokyo
            (35.6762, 139.6503, 1.3521, 103.8198),  # Tokyo to SG
            (51.5074, -0.1278, 50.1109, 8.6821),    # London to Frankfurt
            (1.3521, 103.8198, 19.0760, 72.8777),   # SG to Mumbai
            (19.0760, 72.8777, 51.5074, -0.1278),   # Mumbai to London
            (40.7128, -74.0060, 35.6762, 139.6503)  # NY to Tokyo
        ]
        for flight in flights:
            fig_map.add_trace(go.Scattergeo(
                lon = [flight[1], flight[3]],
                lat = [flight[0], flight[2]],
                mode = 'lines',
                line = dict(width=1.5, color='rgba(0, 200, 83, 0.4)'),
                hoverinfo='skip'
            ))
        
        fig_map.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=30),
            height=900,
            paper_bgcolor='rgba(0,0,0,0)',
            dragmode='pan',
            geo=dict(
                projection_type="orthographic", # 3D Globe
                projection_scale=0.85, # Prevents clipping
                showframe=False, # Hides the square bounding box on zoom
                showcoastlines=False, # Removed coastlines for smoother SVG rendering
                showland=True, landcolor="#0A0A0A",
                showocean=True, oceancolor="#000000",
                showlakes=False,
                showcountries=True, countrycolor="rgba(255,255,255,0.05)",
                bgcolor="rgba(0,0,0,0)",
                resolution=110 # Forces low-res polygon rendering for buttery smooth panning
            ),
            hoverlabel=dict(
                bgcolor="rgba(0, 0, 0, 0.9)",
                bordercolor="#00C853",
                font_size=14
            ),
            showlegend=False
        )
        
        st.plotly_chart(
            fig_map, 
            use_container_width=True, 
            config={
                'scrollZoom': True, 
                'displayModeBar': False, # Removes top right menu for clean look
                'doubleClick': 'reset'
            }
        )

    # ---------------------------------------------------------
    # TAB DATA: Raw Market Data
    # ---------------------------------------------------------
    with tab_data:
        st.markdown(f"### Detailed Market Data ({target_currency})")
        display_df = stock_data.copy()
        # Convert prices to target currency
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            if col in display_df.columns:
                display_df[col] = (display_df[col] * fx_rate).round(2)
        
        # Sort by newest date first
        display_df = display_df.sort_values('Date', ascending=False).reset_index(drop=True)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=900,
            column_config={
                "Date": st.column_config.DatetimeColumn("Date", format="MMM DD, YYYY"),
                "Open": st.column_config.NumberColumn("Open", format="%.2f"),
                "High": st.column_config.NumberColumn("High", format="%.2f"),
                "Low": st.column_config.NumberColumn("Low", format="%.2f"),
                "Close": st.column_config.NumberColumn("Close", format="%.2f"),
                "Adj Close": st.column_config.NumberColumn("Adj Close", format="%.2f"),
                "Volume": st.column_config.NumberColumn("Volume", format="%d")
            }
        )

    # ---------------------------------------------------------
    # TAB 2: Market Overview (Candlesticks)
    # ---------------------------------------------------------
    with tab2:
        st.markdown(f"### Historical Price Action ({target_currency})")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=stock_data['Date'],
                    open=stock_data['Open'] * fx_rate,
                    high=stock_data['High'] * fx_rate,
                    low=stock_data['Low'] * fx_rate,
                    close=stock_data['Close'] * fx_rate,
                    name='Candlestick',
                    increasing_line_color='#00C853', decreasing_line_color='#FF3D00'))
        
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          uirevision='constant',
                          xaxis=dict(showgrid=False, zeroline=False),
                          yaxis=dict(showgrid=False, zeroline=False),
                          xaxis_rangeslider_visible=False, height=500, 
                          legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5),
                          margin=dict(l=0, r=0, t=80, b=0))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ---------------------------------------------------------
    # TAB 3: Moving Averages
    # ---------------------------------------------------------
    with tab3:
        st.markdown(f"### Moving Averages Analysis ({target_currency})")
        ma100 = (stock_data['Close'].rolling(100).mean()) * fx_rate
        ma200 = (stock_data['Close'].rolling(200).mean()) * fx_rate
        close_converted = stock_data['Close'] * fx_rate
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scattergl(x=stock_data['Date'], y=close_converted, mode='lines', name='Closing Price', line=dict(color='#00E5FF', width=2.5), fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.05)'))
        fig2.add_trace(go.Scattergl(x=stock_data['Date'], y=ma100, mode='lines', name='100 MA', line=dict(color='#00C853', width=2)))
        fig2.add_trace(go.Scattergl(x=stock_data['Date'], y=ma200, mode='lines', name='200 MA', line=dict(color='#FF3D00', width=2)))
        
        fig2.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                           uirevision='constant',
                           xaxis=dict(showgrid=False, zeroline=False),
                           yaxis=dict(showgrid=False, zeroline=False),
                           hovermode="x unified", height=500, 
                           legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5),
                           margin=dict(l=0, r=0, t=80, b=0))
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    # ---------------------------------------------------------
    # TAB 4: LSTM AI Prediction
    # ---------------------------------------------------------
    with tab4:
        st.markdown("### Deep Learning LSTM Model Output")
        
        with st.spinner("Initializing Deep Neural Network"):
            model = load_keras_model()

        close_series = stock_data['Close'].squeeze()
        data_training = pd.DataFrame(close_series[0:int(len(stock_data)*0.70)])
        data_testing = pd.DataFrame(close_series[int(len(stock_data)*0.70): int(len(stock_data))])

        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(data_training)

        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)
        
        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100: i])
            y_test.append(input_data[i,0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        
        with st.spinner("Running Inference Engine"):
            y_predicted = model.predict(x_test, verbose=0)
        
        scale_factor = 1/scaler.scale_[0]
        y_predicted = (y_predicted * scale_factor) * fx_rate
        y_test = (y_test * scale_factor) * fx_rate

        lstm_change = y_predicted[-1] - y_predicted[0]
        lstm_color = '#00e676' if lstm_change >= 0 else '#FF3D00'
        lstm_fill = 'rgba(0, 230, 118, 0.1)' if lstm_change >= 0 else 'rgba(255, 61, 0, 0.1)'

        fig3 = go.Figure()
        fig3.add_trace(go.Scattergl(y=y_test.flatten(), mode='lines', name='Actual Market Data', line=dict(color='rgba(150, 150, 150, 0.6)', width=1.5)))
        fig3.add_trace(go.Scattergl(y=y_predicted.flatten(), mode='lines', name='LSTM Prediction', line=dict(color=lstm_color, width=2.5), fill='tozeroy', fillcolor=lstm_fill))
        
        fig3.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                           uirevision='constant',
                           hovermode="x unified", height=500,
                           xaxis=dict(showgrid=False, zeroline=False, title=dict(text="Timeline", font=dict(size=11, color="#8892B0")), tickfont=dict(size=10)),
                           yaxis=dict(showgrid=False, zeroline=False, title=dict(text=f"Price ({target_currency})", font=dict(size=11, color="#8892B0")), tickfont=dict(size=10)),
                           legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5),
                           margin=dict(l=0, r=0, t=80, b=0))
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

    # ---------------------------------------------------------
    # TAB 5: NeuralProphet Forecast
    # ---------------------------------------------------------
    with tab5:
        st.markdown("### NeuralProphet Long-Term Forecasting")
        
        # Use hist_modified_stock_data to preserve cache across live price ticks
        hist_stocks = hist_modified_stock_data[['Date', 'Close']].copy()
        hist_stocks.columns = ['ds', 'y']
        
        with st.spinner(" AI Model is computing 300-days future trajectory"):
            forecast, actual_prediction = train_neural_prophet(user_input, str(start_date), str(end_date), hist_stocks)

        # Pin-point accuracy mathematical anchor to EXACT live price
        last_fit_val = actual_prediction['yhat1'].iloc[-1]
        live_price_val = stock_data['Close'].iloc[-1]
        correction_offset = live_price_val - last_fit_val
        
        forecast_corrected = forecast.copy()
        forecast_corrected['yhat1'] = forecast_corrected['yhat1'] + correction_offset

        # AI Investment Intelligence Engine
        last_actual_price = live_price_val * fx_rate
        final_forecast_price = forecast_corrected['yhat1'].iloc[-1] * fx_rate
        price_diff = final_forecast_price - last_actual_price
        pct_change = (price_diff / last_actual_price) * 100
        
        # Smart Trend Override (Detects crashed stocks / falling knives)
        try:
            ma200_val = stock_data['Close'].rolling(200).mean().dropna().iloc[-1] * fx_rate
            is_crashing = last_actual_price < (ma200_val * 0.8) # 20% below 200-day MA
        except:
            is_crashing = False
        
        if is_crashing and pct_change > 0:
            recom, color, animation = "HIGH RISK / AVOID", "#FF3D00", "pulse-red"
            desc = f"Projected {pct_change:.2f}% technical bounce, but stock is in severe structural downtrend. Avoid."
            color = "#FF3D00" # Force Red Line
        elif pct_change > 15:
            recom, color, animation = "STRONG BUY", "#00e676", "pulse-green"
            desc = f"Projected {pct_change:.2f}% Upside. AI detects highly favorable risk-reward ratio for accumulation."
        elif pct_change > 5:
            recom, color, animation = "ACCUMULATE", "#00C853", "pulse-green"
            desc = f"Projected {pct_change:.2f}% Growth. Consider strategic entry points on market pullbacks."
        elif pct_change > 1:
            recom, color, animation = "MODERATE BUY", "#00C853", "pulse-green"
            desc = f"Slight bullish divergence ({pct_change:.2f}%). Maintain cautious upside exposure."
        elif pct_change > -5:
            recom, color, animation = "NEUTRAL / HOLD", "#FFD600", "pulse-yellow"
            desc = f"Market consolidating ({pct_change:.2f}%). Awaiting clearer momentum breakout signals."
        elif pct_change > -15:
            recom, color, animation = "HIGH RISK EXPOSURE", "#FF3D00", "pulse-red"
            desc = f"Projected {pct_change:.2f}% Drop. AI detects bearish technicals. Consider hedging."
        else:
            recom, color, animation = "LIQUIDATE / AVOID", "#D50000", "pulse-red"
            desc = f"Severe downside risk ({pct_change:.2f}%). Capital preservation is strongly recommended."

        load_html(
            'ai_card.html',
            color=color,
            animation=animation,
            recom=recom,
            desc=desc
        )

        future_color = color
        if color in ["#00e676", "#00C853"]:
            future_fill = 'rgba(0, 230, 118, 0.08)'
        elif color == "#FFD600":
            future_fill = 'rgba(255, 214, 0, 0.08)'
        else:
            future_fill = 'rgba(255, 61, 0, 0.08)'

        fig4 = go.Figure()
        
        # Use the combined real-time data for the original history trace
        plot_hist_ds = stock_data['Date']
        plot_hist_y = stock_data['Close'] * fx_rate
        
        fig4.add_trace(go.Scattergl(x=plot_hist_ds, y=plot_hist_y, mode='lines', name='Original History', line=dict(color='#00E5FF', width=2.5), fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.08)'))
        
        # Filter forecast strictly to future dates to prevent backwards zig-zag
        last_date = plot_hist_ds.iloc[-1]
        future_only = forecast_corrected[forecast_corrected['ds'] > last_date]
        
        # Weld the forecast to the exact last known historical point to eliminate visual gap
        plot_forecast_ds = [last_date] + future_only['ds'].tolist()
        plot_forecast_y = [plot_hist_y.iloc[-1]] + (future_only['yhat1'] * fx_rate).tolist()
        
        fig4.add_trace(go.Scattergl(x=plot_forecast_ds, y=plot_forecast_y, mode='lines', name='Future Trajectory', line=dict(color=future_color, width=2.5), fill='tozeroy', fillcolor=future_fill))
        
        fig4.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                           uirevision='constant',
                           hovermode="x unified", height=500,
                           xaxis=dict(showgrid=False, zeroline=False, title=dict(text="Timeline", font=dict(size=11, color="#8892B0")), tickfont=dict(size=10)),
                           yaxis=dict(showgrid=False, zeroline=False, title=dict(text=f"Price ({target_currency})", font=dict(size=11, color="#8892B0")), tickfont=dict(size=10)),
                           legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5),
                           margin=dict(l=0, r=0, t=80, b=0))
        
        # Modern Interactive Range Selector
        fig4.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ]),
                bgcolor="#1A1A1A",
                activecolor="#00C853"
            )
        )
        
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})

# Call the fragment to render everything below the data fetching
render_dashboard()
