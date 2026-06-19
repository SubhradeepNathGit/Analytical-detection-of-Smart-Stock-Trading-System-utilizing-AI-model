# 📈 Smart Stock AI: Analytical Trading Intelligence

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-NeuralProphet-EE4C2C.svg)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-5bc0de.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Smart Stock AI** is a comprehensive, production-ready stock market analysis and forecasting platform. Engineered with a premium, glassmorphic UI, it combines the predictive power of **Deep Learning (LSTM)** and **NeuralProphet** to deliver highly accurate short-term trend analysis and robust long-term forecasting.

This application is fully dynamic—it streams live market data, adapts perfectly to desktop and mobile screens, and generates actionable, algorithmic trading signals in real-time.

---

## 🌟 Comprehensive Feature Breakdown

### 1. Live Data Acquisition Engine
At its core, the app utilizes `yfinance` to establish a direct, real-time pipeline to Yahoo Finance. 
* **Dynamic Timeframes:** It fetches historical Open, High, Low, Close, and Volume (OHLCV) data from a default start date (2016) up to the exact **current day**. 
* **Live Currency Conversion:** Features an integrated Forex engine that dynamically converts the entire dataset (including historical charts, metric cards, and AI predictions) from USD into multiple global currencies (EUR, GBP, INR, JPY, AUD, CAD) on the fly.

### 2. The Dashboard & UI/UX Architecture
The UI is inspired by institutional-grade trading terminals (like Bloomberg Terminal and Robinhood).
* **Glassmorphic Design:** Built with custom CSS to provide a sleek, translucent dark-mode aesthetic (`#000000` background with `rgba` blur filters).
* **Responsive Metric Grid:** The top of the dashboard displays 6 live metric cards (Current Price, Today's High, Today's Low, 52W High, 52W Low, and Average Volume).
  * **Desktop:** The custom CSS strictly enforces a panoramic, unbroken single-row layout utilizing fluid typography (`clamp()`).
  * **Mobile:** The layout flawlessly collapses into an ultra-compact 2x3 matrix, eliminating horizontal scrolling and optimizing screen real-estate.

### 3. Data Visualization Suite (Plotly)
All charts are rendered using `Plotly`, optimized with the `plotly_dark` template and styled to match Google Finance's minimalist aesthetic.
* **Global Market Map:** A 3D orthographic globe visualizing major financial hubs (New York, London, Tokyo, Frankfurt, Mumbai, Singapore) connected by animated, translucent green data streams.
* **Raw Market Data:** An interactive tabular view highlighting recent dataframes.
* **Price Action (Candlesticks):** Professional candlestick charts with integrated 50-day and 100-day Simple Moving Averages (SMA) overlaid on the daily price action.
* **Moving Averages Analysis:** Dedicated line charts plotting the 100-day vs 200-day SMAs to identify golden crosses and death crosses.

### 4. Deep Learning Engine (LSTM)
Designed for **Short-Term Price Action**. 
* **Architecture:** Uses a pre-trained Keras/TensorFlow Sequential model (Long Short-Term Memory network).
* **Data Preprocessing:** The engine automatically applies a `MinMaxScaler` to normalize the data between `0` and `1`. It uses a **100-day rolling window** (looking at the past 100 days of closing prices) to predict the 101st day.
* **Visualization:** The LSTM chart compares the *Actual Market Data* against the *LSTM Prediction*, utilizing a seamless translucent green area-fill (`tozeroy`) to visually map the algorithm's accuracy.

### 5. Algorithmic Forecasting (NeuralProphet)
Designed for **Long-Term Trajectory (300 Days)**.
* **Architecture:** Built on PyTorch, NeuralProphet excels at handling time-series data with strong seasonality and trend changes.
* **Pin-Point Accuracy Anchoring:** The app mathematically calculates the offset between the model's fitted curve and the *actual live price* today. It then anchors the future trajectory directly to the live price, ensuring the 300-day forecast connects seamlessly with zero margin of error at day zero.
* **AI Investment Intelligence Engine:** Based on the 300-day projected percentage change, the engine runs a ruleset to issue a definitive trading signal:
  * `> 5%`: **STRONG BUY** (Pulsing Green)
  * `> 1%`: **BUY**
  * `> -1%`: **HOLD** (Pulsing Yellow)
  * `> -5%`: **SELL**
  * `< -5%`: **STRONG SELL** (Pulsing Red)

---

## 🛠️ Technology Stack & Dependencies

* **Frontend Framework:** Streamlit
* **Styling:** Custom CSS3 (Flexbox, CSS Grid, Media Queries, Keyframe Animations)
* **Backend:** Python 3.12+
* **Deep Learning Framework:** TensorFlow 2.x / Keras
* **Time-Series Forecasting:** NeuralProphet (PyTorch & PyTorch Lightning backend)
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning Utilities:** Scikit-learn (`MinMaxScaler`)
* **Live API Integration:** `yfinance`
* **Charting Library:** Plotly (`plotly.graph_objs`, `plotly.express`)

---

## 🚀 Local Installation & Setup

You can run this entire AI engine locally on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/SubhradeepNathGit/Analytical-detection-of-Smart-Stock-Trading-System-utilizing-AI-model.git
cd Analytical-detection-of-Smart-Stock-Trading-System-utilizing-AI-model
cd "stock trend prediction/streamlit"
```

### 2. Create a Virtual Environment (Highly Recommended)
Because TensorFlow and PyTorch have complex dependency trees, a virtual environment is heavily recommended.
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Boot the AI Engine
For Windows users, a 1-click executable batch script is provided:
```bash
RUN_APP.bat
```
Alternatively, launch it manually via Streamlit:
```bash
streamlit run app.py
```

---

## ☁️ Cloud Deployment (Streamlit Community Cloud)

This app is optimized for immediate deployment to Streamlit Community Cloud without PyTorch/TensorFlow conflicts.

1. Commit and push your code to a public or private GitHub repository.
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New App"**.
4. Select your GitHub repository and branch.
5. Set the Main file path to: `stock trend prediction/streamlit/app.py`.
6. Click **"Deploy"**.

**Cloud Optimization Notes:** The `requirements.txt` file has been meticulously stripped of problematic local-only packages and locked to stable versions of `neuralprophet` and `tensorflow` to ensure smooth server-side builds. The code also includes a custom PyTorch unpickling bypass (`weights_only=False`) to safely load legacy `.h5` and `.pt` models on modern PyTorch 2.6+ cloud instances.

---

## 📂 Repository Structure

```text
📁 stock trend prediction/streamlit/
│
├── app.py                 # The core Python application and Streamlit UI logic
├── keras_model.h5         # The pre-trained TensorFlow LSTM neural network weights
├── requirements.txt       # Cloud-optimized dependency lockfile
├── RUN_APP.bat            # Windows automation script for local launching
├── README.md              # Project documentation
└── .gitignore             # Ignored files (virtual environments, cache, etc.)
```

---

## ⚠️ Legal Disclaimer

**Not Financial Advice:** This software is built strictly for educational, analytical, algorithmic, and research purposes. The AI predictions, "Buy/Hold/Sell" signals, LSTM tracking, and NeuralProphet forecasts are mathematically generated estimations based purely on historical OHLCV data. 

Financial markets are subject to unprecedented volatility, news events, and human sentiment that these models cannot account for. **These outputs do not constitute professional financial advice.** Always conduct your own exhaustive due diligence or consult a licensed financial advisor before making real capital investments.

---

*Designed, Engineered, and Developed by **Subhradeep Nath**.*
