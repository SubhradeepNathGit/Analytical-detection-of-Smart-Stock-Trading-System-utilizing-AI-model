import numpy as np
import pandas as pd
import yfinance as yf
# Using tf.keras for compatibility with TensorFlow 2.18+ / Keras 3
from neuralprophet import NeuralProphet
import streamlit as st
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import plotly.express as px
import torch
import neuralprophet.configure

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
st.set_page_config(page_title="Smart Stock AI", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Full Black Background & Modern Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Title Styling */
    .main-title {
        text-align: center;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 3.5rem;
        margin-bottom: 0px;
        letter-spacing: -1px;
    }
    
    .sub-title {
        text-align: center;
        color: #8892B0;
        font-weight: 300;
        margin-bottom: 15px;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-size: 0.9rem;
    }

    /* Metric Cards - Premium Glassmorphism */
    div[data-testid="metric-container"] {
        background: rgba(20, 20, 20, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        border: 1px solid rgba(0, 200, 83, 0.5);
        box-shadow: 0 0 20px rgba(0, 200, 83, 0.2);
        transform: translateY(-2px);
    }

    /* Sidebar Premium Styling */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
        border-bottom: none !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none; 
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important; 
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: nowrap !important;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #8892B0;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #FF0000 !important;
    }

    /* Responsive Mobile & Tablet Adjustments */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem !important;
            line-height: 1.2 !important;
        }
        .sub-title {
            font-size: 0.75rem !important;
            margin-bottom: 20px !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.85rem !important;
            padding-top: 8px !important;
            padding-bottom: 8px !important;
        }
        div[data-testid="stPlotlyChart"], div[data-testid="stPlotlyChart"] iframe {
            height: 400px !important;
        }
    }

    /* Hide Streamlit Branding but Keep Sidebar Toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;} - Removed to keep sidebar toggle visible */
    </style>
""", unsafe_allow_html=True)

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
    user_input = st.text_input('Stock Ticker', 'AAPL').upper()
    start_date = st.date_input('Start Date', pd.to_datetime('2016-01-01'))
    end_date = st.date_input('End Date', datetime.date.today())
    
    st.markdown("### Currency Converter")
    target_currency = st.selectbox(
        "Display Values In",
        ["USD", "EUR", "GBP", "INR", "JPY", "AUD", "CAD"]
    )
    
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
@st.cache_data(show_spinner=False, ttl=3600)
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

@st.cache_data(show_spinner=False)
def load_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.droplevel(1)
    stock_data.reset_index(inplace=True)
    modified_stock_data = stock_data.drop(columns=['Adj Close'], errors='ignore')
    return stock_data, modified_stock_data

# ---------------------------------------------------------
# Data Fetching & Processing
# ---------------------------------------------------------
with st.spinner(f"Connecting to Market Data Exchange..."):
    try:
        stock_data, modified_stock_data = load_stock_data(user_input, start_date, end_date)
        fx_rate, fx_symbol = get_exchange_rate(target_currency)
    except Exception as e:
        st.error(f"Failed to fetch data for {user_input}.")
        st.stop()

if stock_data.empty:
    st.warning(f"No data found for {user_input}.")
    st.stop()

# ---------------------------------------------------------
# Top Metrics Board (With Currency Conversion)
# ---------------------------------------------------------
current_price = stock_data['Close'].iloc[-1] * fx_rate
prev_price = stock_data['Close'].iloc[-2] * fx_rate
price_change = current_price - prev_price
pct_change = (price_change / prev_price) * 100
delta_color = "#00e676" if price_change >= 0 else "#FF3D00"
delta_arrow = "▲" if price_change >= 0 else "▼"

high_52 = stock_data['Close'].tail(252).max() * fx_rate
low_52 = stock_data['Close'].tail(252).min() * fx_rate
avg_vol = int(stock_data['Volume'].tail(30).mean())
today_high = stock_data['High'].iloc[-1] * fx_rate
today_low = stock_data['Low'].iloc[-1] * fx_rate

st.markdown(f"""
    <style>
    .metric-row {{
        display: flex;
        justify-content: space-between;
        flex-wrap: nowrap;
        gap: 12px;
        padding-bottom: 20px;
    }}
    .custom-metric-card {{
        flex: 1;
        min-width: 0;
        background: rgba(20, 20, 20, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 20px 10px;
        border-radius: 12px;
        text-align: center;
        backdrop-filter: blur(12px);
        font-family: 'Outfit', sans-serif;
        transition: transform 0.2s ease;
    }}
    .custom-metric-card:hover {{ transform: translateY(-2px); border-color: rgba(255,255,255,0.1); }}
    .m-label {{ color: #8892B0; font-size: clamp(10px, 1vw, 12px); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; font-weight: 400; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .m-value {{ color: #FFFFFF; font-size: clamp(16px, 1.5vw, 24px); font-weight: 600; margin-bottom: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .m-delta {{ color: {delta_color}; font-size: clamp(10px, 1vw, 13px); font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; }}
    
    @media (max-width: 768px) {{
        .metric-row {{ 
            flex-wrap: wrap; 
            justify-content: space-between; 
            gap: 10px; 
            padding-bottom: 5px; 
            overflow: visible;
        }}
        .custom-metric-card {{ 
            flex: 0 0 calc(50% - 6px) !important; 
            width: calc(50% - 6px) !important;
            min-width: unset; 
            padding: 15px 10px; 
        }}
        .m-value {{ font-size: 20px; }}
        .sub-title {{ font-size: 0.75rem !important; letter-spacing: 1px; }}
    }}
    </style>
    <div class="metric-row">
        <div class="custom-metric-card">
            <div class="m-label">Current Price</div>
            <div class="m-value">{fx_symbol}{current_price:.2f}</div>
            <div class="m-delta" style="color: {delta_color};">{delta_arrow} {abs(price_change):.2f} ({pct_change:.2f}%)</div>
        </div>
        <div class="custom-metric-card">
            <div class="m-label">Today's High</div>
            <div class="m-value">{fx_symbol}{today_high:.2f}</div>
        </div>
        <div class="custom-metric-card">
            <div class="m-label">Today's Low</div>
            <div class="m-value">{fx_symbol}{today_low:.2f}</div>
        </div>
        <div class="custom-metric-card">
            <div class="m-label">52W High</div>
            <div class="m-value">{fx_symbol}{high_52:.2f}</div>
        </div>
        <div class="custom-metric-card">
            <div class="m-label">52W Low</div>
            <div class="m-value">{fx_symbol}{low_52:.2f}</div>
        </div>
        <div class="custom-metric-card">
            <div class="m-label">Avg Volume</div>
            <div class="m-value">{avg_vol:,}</div>
        </div>
    </div>
""", unsafe_allow_html=True)

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
    fig2.add_trace(go.Scatter(x=stock_data['Date'], y=close_converted, mode='lines', name='Closing Price', line=dict(color='#FFFFFF', width=2)))
    fig2.add_trace(go.Scatter(x=stock_data['Date'], y=ma100, mode='lines', name='100 MA', line=dict(color='#00C853', width=2)))
    fig2.add_trace(go.Scatter(x=stock_data['Date'], y=ma200, mode='lines', name='200 MA', line=dict(color='#FF3D00', width=2)))
    
    fig2.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                       hovermode="x unified", height=500, 
                       legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5),
                       margin=dict(l=0, r=0, t=80, b=0))
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

# ---------------------------------------------------------
# TAB 4: LSTM AI Prediction
# ---------------------------------------------------------
with tab4:
    st.markdown("### Deep Learning LSTM Model Output")
    
    @st.cache_resource(show_spinner=False)
    def load_keras_model():
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keras_model.h5')
        model = tf.keras.models.load_model(model_path)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    with st.spinner("Initializing Deep Neural Network..."):
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
    
    with st.spinner("Running Inference Engine..."):
        y_predicted = model.predict(x_test, verbose=0)
    
    scale_factor = 1/scaler.scale_[0]
    y_predicted = (y_predicted * scale_factor) * fx_rate
    y_test = (y_test * scale_factor) * fx_rate

    lstm_change = y_predicted[-1] - y_predicted[0]
    lstm_color = '#00e676' if lstm_change >= 0 else '#FF3D00'
    lstm_fill = 'rgba(0, 230, 118, 0.1)' if lstm_change >= 0 else 'rgba(255, 61, 0, 0.1)'

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=y_test.flatten(), mode='lines', name='Actual Market Data', line=dict(color='rgba(255, 61, 0, 0.5)', width=1.5)))
    fig3.add_trace(go.Scatter(y=y_predicted.flatten(), mode='lines', name='LSTM Prediction', line=dict(color=lstm_color, width=2.5), fill='tozeroy', fillcolor=lstm_fill))
    
    fig3.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                       hovermode="x unified", height=500, xaxis_title="Days (Test Dataset)", yaxis_title=f"Price ({target_currency})",
                       legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5),
                       margin=dict(l=0, r=0, t=80, b=0))
    st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

# ---------------------------------------------------------
# TAB 5: NeuralProphet Forecast
# ---------------------------------------------------------
with tab5:
    st.markdown("### NeuralProphet Long-Term Forecasting")
    
    stocks = modified_stock_data[['Date', 'Close']].copy()
    stocks.columns = ['ds', 'y']
    
    @st.cache_resource(show_spinner=False)
    def train_neural_prophet(_stocks_df):
        model_neuralprophet = NeuralProphet(epochs=50, trainer_config={"enable_checkpointing": False})
        model_neuralprophet.fit(_stocks_df, freq='B')
        future = model_neuralprophet.make_future_dataframe(_stocks_df, periods=300)
        forecast = model_neuralprophet.predict(future)
        actual_prediction = model_neuralprophet.predict(_stocks_df)
        return forecast, actual_prediction

    with st.spinner(" AI Model is computing 300-days future trajectory"):
        forecast, actual_prediction = train_neural_prophet(stocks)

    # Pin-point accuracy mathematical anchor
    last_actual_val = stocks['y'].iloc[-1]
    last_fit_val = actual_prediction['yhat1'].iloc[-1]
    correction_offset = last_actual_val - last_fit_val
    forecast['yhat1'] = forecast['yhat1'] + correction_offset

    # AI Investment Intelligence Engine
    last_actual_price = stocks['y'].iloc[-1] * fx_rate
    final_forecast_price = forecast['yhat1'].iloc[-1] * fx_rate
    price_diff = final_forecast_price - last_actual_price
    pct_change = (price_diff / last_actual_price) * 100
    
    if pct_change > 5:
        recom, color, animation = "STRONG BUY", "#00e676", "pulse-green"
        desc = f"AI Predicts massive {pct_change:.2f}% Upside. Highly favorable entry point."
    elif pct_change > 1:
        recom, color, animation = "BUY", "#00C853", "pulse-green"
        desc = f"AI Predicts {pct_change:.2f}% Growth. Steady accumulation phase."
    elif pct_change > -1:
        recom, color, animation = "HOLD", "#FFD600", "pulse-yellow"
        desc = f"AI Predicts Flat Market ({pct_change:.2f}%). Wait for clear momentum breakout."
    elif pct_change > -5:
        recom, color, animation = "SELL", "#FF3D00", "pulse-red"
        desc = f"AI Predicts {pct_change:.2f}% Drop. Consider taking profits."
    else:
        recom, color, animation = "STRONG SELL", "#D50000", "pulse-red"
        desc = f"AI Predicts heavy {pct_change:.2f}% Plunge. Exit positions immediately."

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
        
        @keyframes pulse-green {{
            0% {{ box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.4); }}
            70% {{ box-shadow: 0 0 0 15px rgba(0, 230, 118, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(0, 230, 118, 0); }}
        }}
        @keyframes pulse-red {{
            0% {{ box-shadow: 0 0 0 0 rgba(213, 0, 0, 0.4); }}
            70% {{ box-shadow: 0 0 0 15px rgba(213, 0, 0, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(213, 0, 0, 0); }}
        }}
        @keyframes pulse-yellow {{
            0% {{ box-shadow: 0 0 0 0 rgba(255, 214, 0, 0.4); }}
            70% {{ box-shadow: 0 0 0 15px rgba(255, 214, 0, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(255, 214, 0, 0); }}
        }}
        .ai-card {{
            background: #000000;
            border: 1px solid #1A1A1A;
            border-radius: 8px;
            padding: 35px 20px;
            text-align: center;
            margin-bottom: 30px;
            position: relative;
            overflow: hidden;
            font-family: 'Inter', sans-serif;
        }}
        .ai-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: {color};
            box-shadow: 0 0 15px {color};
        }}
        .ai-title {{ color: #555555; font-size: 11px; text-transform: uppercase; letter-spacing: 4px; margin-bottom: 20px; font-weight: 600; }}
        .ai-recom {{ 
            color: #FFFFFF; 
            font-size: 42px; 
            font-weight: 300; 
            letter-spacing: 2px; 
            margin-bottom: 15px; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            gap: 15px; 
        }}
        .pulse-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: {color};
            box-shadow: 0 0 10px {color};
            animation: {animation} 2s infinite;
        }}
        .ai-desc {{ color: #888888; font-size: 15px; font-weight: 300; }}
        </style>
        
        <div class="ai-card">
            <div class="ai-title">Live AI Trading Intelligence</div>
            <div class="ai-recom">
                <div class="pulse-dot"></div>
                {recom}
            </div>
            <div class="ai-desc">{desc}</div>
        </div>
    """, unsafe_allow_html=True)

    future_color = '#00e676' if pct_change >= 0 else '#FF3D00'
    future_fill = 'rgba(0, 230, 118, 0.08)' if pct_change >= 0 else 'rgba(255, 61, 0, 0.08)'

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=stocks['ds'], y=stocks['y'] * fx_rate, mode='lines', name='Original History', line=dict(color='#00E5FF', width=2.5), fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.08)'))
    
    # Weld the forecast to the exact last known historical point to eliminate visual gap
    plot_forecast_ds = [stocks['ds'].iloc[-1]] + forecast['ds'].tolist()
    plot_forecast_y = [(stocks['y'].iloc[-1]) * fx_rate] + (forecast['yhat1'] * fx_rate).tolist()
    
    fig4.add_trace(go.Scatter(x=plot_forecast_ds, y=plot_forecast_y, mode='lines', name='Future Trajectory', line=dict(color=future_color, width=2.5), fill='tozeroy', fillcolor=future_fill))
    
    fig4.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                       hovermode="x unified", height=500, xaxis_title="Timeline", yaxis_title=f"Projected Price ({target_currency})",
                       legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5),
                       margin=dict(l=0, r=0, t=80, b=0))
    
    # Modern Interactive Range Selector
    fig4.update_xaxes(
        rangeslider_visible=True,
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
