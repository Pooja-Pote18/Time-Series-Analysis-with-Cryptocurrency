import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.statespace.sarimax import SARIMAX
import mplfinance as mpf
import os
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Time Series Forecasting Dashboard",
    page_icon="🪙",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
DATA_PATH = "binance_crypto_data_file.csv"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()
data = df.copy()

# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------
def evaluate_model(actual, predicted, model_name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

# --------------------------------------------------
# SIDEBAR MENU
# --------------------------------------------------
if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

with st.sidebar:
    menu_items = ["Home", "Data View", "EDA","Forecasting Models","Model Evaluation","Power BI Dashboard"]
    for item in menu_items:
        is_active = st.session_state.active_page == item
        st.markdown(
            f"""
            <div style="border-radius: 8px;
                        margin-bottom: 8px;
                        background-color: {'#2b2f33' if is_active else 'transparent'};
                        border-left: {'5px solid #ffcc00' if is_active else '5px solid transparent'};">
            """,
            unsafe_allow_html=True,
        )
        if st.button(item, key=f"menu_{item}", use_container_width=True):
            st.session_state.active_page = item
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# DARK GREEN STYLE
# --------------------------------------------------
st.markdown("""
<style>
body, .main {
    background-color: #0f172a;
    color: #d1fae5;
}
h1, h2, h3 {
    color: #22c55e !important;
}
[data-testid="stSidebar"] {
    background-color: #022c22 !important;
}
.stButton>button {
    background-color: #022c22 !important;
    color: white !important;
    border-radius: 8px;
    border: 1px solid #22c55e !important;
}
.stButton>button:hover {
    background-color: #22c55e !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

page = st.session_state.active_page

# --------------------------------------------------
# HOME
# --------------------------------------------------
if page == "Home":

    st.title("📈 Time Series Analysis with Cryptocurrency")

    col1, col2 = st.columns([0.65, 0.35])

    with col1:
        st.markdown("<span style='color:#ffcc00;'>Project Description</span>", unsafe_allow_html=True)
        st.write("""
        This project focuses on **cryptocurrency price forecasting** using real-world data collected from the **Binance API**.  
        The dataset was **preprocessed** and analyzed through detailed **Exploratory Data Analysis (EDA)** to uncover patterns in price movement, volatility, and trading volume.
        """)

        st.write("""
        A series of **forecasting models** were implemented and compared to identify the most accurate and best-fitting approach for the dataset.  
        Model evaluation was carried out using **performance metrics** and **visualization insights** to assess accuracy and consistency.
        """)

        st.write("""
        To complement the analysis, **Power BI dashboards** were integrated to provide interactive visualizations and deeper insights into time series behavior.
        """)

        st.markdown("#### <span style='color:#ffcc00;'>Forecasting Models Implemented</span>", unsafe_allow_html=True)
        st.markdown("""
        - Prophet  
        - ARIMA  
        - LSTM  
        - SARIMA
        """)

    with col2:
        try:
            if os.path.exists("binance2.webp"):
                image = Image.open("binance2.webp")
                st.image(image, use_column_width=True)
            else:
                st.warning("Image not found.")
        except Exception as e:
            st.warning(f"Image error: {e}")

        st.markdown("---")
        st.markdown("### <span style='color:#ffcc00;'> 👥 Team Members</span>", unsafe_allow_html=True)
        st.markdown("""
        - Pooja Pote  
        - Anjali Kamble  
        - Piyush Chinde  
        - Dnyaneshwari Gurav
        """)

# --------------------------------------------------
# DATA VIEW
# --------------------------------------------------
elif page == "Data View":

    st.subheader("Data View")

    if not df.empty:
        st.dataframe(df, use_container_width=True)
        st.subheader("Summary")
        st.dataframe(df.describe(), use_container_width=True)
    else:
        st.warning("No data loaded.")

# --------------------------------------------------
# EDA SECTION (ALL ORIGINAL TABS)
# --------------------------------------------------
elif page == "EDA":

    st.subheader("Exploratory Data Analysis (EDA)")

    if df.empty:
        st.warning("No data loaded.")
    else:

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Closing Price Distribution",
            "Closing Price Trend",
            "Volume Trend",
            "Candlestick Charts",
            "Correlation Heatmap",
            "Volatility & Moving Averages",
            "Monthly Boxplot"
        ])

        # 1 Distribution
        with tab1:
            fig, ax = plt.subplots(figsize=(7,4))
            ax.hist(data['close'], bins=30)
            ax.set_title("Distribution of Closing Prices")
            st.pyplot(fig)

        # 2 Trend
        with tab2:
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(data['timestamp'], data['close'])
            ax.set_title("Closing Price Over Time")
            st.pyplot(fig)

        # 3 Volume
        with tab3:
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(data['timestamp'], data['volume'])
            ax.set_title("Volume Trend")
            st.pyplot(fig)

        # 4 Candlestick
        with tab4:
            df_candle = df.copy()
            df_candle.set_index("timestamp", inplace=True)
            candle_data = df_candle[['open','high','low','close','volume']]
            fig, _ = mpf.plot(candle_data, type='candle', style='yahoo', returnfig=True)
            st.pyplot(fig)

        # 5 Heatmap
        with tab5:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # 6 Volatility
        with tab6:
            df_vol = df.copy()
            df_vol['daily_return'] = df_vol['close'].pct_change()
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(df_vol['daily_return'])
            ax.set_title("Daily Returns")
            st.pyplot(fig)

        # 7 Boxplot
        with tab7:
            df_box = df.copy()
            df_box['Month'] = df_box['timestamp'].dt.month_name()
            df_box['daily_return'] = df_box['close'].pct_change()
            fig, ax = plt.subplots(figsize=(9,5))
            sns.boxplot(x='Month', y='daily_return', data=df_box, ax=ax)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

# --------------------------------------------------
# FORECASTING MODELS
# --------------------------------------------------

elif page == "Forecasting Models":
    st.subheader("Forecasting Models")

    # Prophet
    df_prophet = data[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
    train_size = int(len(df_prophet) * 0.8)
    train, test = df_prophet[:train_size], df_prophet[train_size:]
    model_p = Prophet()
    model_p.fit(train)
    future = model_p.make_future_dataframe(periods=len(test), freq='D')
    forecast = model_p.predict(future)
    prophet_pred = forecast['yhat'][-len(test):].values
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(train['ds'], train['y'], label='Train')
    ax1.plot(test['ds'], test['y'], label='Test', color='orange')
    ax1.plot(test['ds'], prophet_pred, label='Prophet', color='green', linestyle='--')
    ax1.legend(); ax1.set_title("Prophet Forecast")
    prophet_metrics = evaluate_model(test['y'], prophet_pred, "Prophet")

    # ARIMA
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    train_size = int(len(data) * 0.8)
    train, test = data['close'][:train_size], data['close'][train_size:]
    arima_fit = ARIMA(train, order=(5,1,0)).fit()
    arima_pred = arima_fit.forecast(steps=len(test))
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(train.index, train, label='Train')
    ax2.plot(test.index, test, label='Test', color='orange')
    ax2.plot(test.index, arima_pred, label='ARIMA', color='green', linestyle='--')
    ax2.legend(); ax2.set_title("ARIMA Forecast")
    arima_metrics = evaluate_model(test, arima_pred, "ARIMA")

    # SARIMA
    sarima_model = SARIMAX(train, order=(2,1,2), seasonal_order=(1,1,1,12)).fit(disp=False)
    sarima_pred = sarima_model.predict(start=len(train), end=len(data)-1, dynamic=False)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(train.index, train, label='Train')
    ax3.plot(test.index, test, label='Test', color='orange')
    ax3.plot(test.index, sarima_pred, label='SARIMA', color='green', linestyle='--')
    ax3.legend(); ax3.set_title("SARIMA Forecast")
    sarima_metrics = evaluate_model(test, sarima_pred, "SARIMA")

    # LSTM
    prices = data['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    split = int(len(scaled)*0.8)
    train_data, test_data = scaled[:split], scaled[split:]

    def create_dataset(ds, step=60):
        X, y = [], []
        for i in range(step, len(ds)):
            X.append(ds[i-step:i, 0])
            y.append(ds[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    lstm_pred = model.predict(X_test)
    lstm_pred = scaler.inverse_transform(lstm_pred)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.plot(actual_prices, label='Actual', color='blue')
    ax4.plot(lstm_pred, label='Predicted', color='red', linestyle='--')
    ax4.legend(); ax4.set_title("LSTM Prediction")
    lstm_metrics = evaluate_model(actual_prices, lstm_pred, "LSTM")

    

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Prophet Model")
        st.pyplot(fig1)
    with col2:
        st.markdown("#### ARIMA Model")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### SARIMA Model")
        st.pyplot(fig3)
    with col4:
        st.markdown("#### LSTM Model")
        st.pyplot(fig4)

    # Save results for evaluation tab
    st.session_state["model_metrics"] = [prophet_metrics, arima_metrics, sarima_metrics, lstm_metrics]




# --------------------------------------------------
# POWER BI
# --------------------------------------------------
elif page == "📊 Power BI Dashboard":

    st.markdown("## Interactive Power BI Dashboard")

    powerbi_url = "https://app.powerbi.com/view?r=eyJrIjoiYjA3YWQyN2MtMDM4ZC00YWUxLTlkNGQtNWIxYTc2MTZiZTI1IiwidCI6IjM0YTYzMzMwLWU2MWUtNGMwZC04ODIyLTQ4MjViZTk0YTNkYiJ9"

    components.iframe(powerbi_url, width=1200, height=650)







