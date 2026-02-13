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
    menu_items = ["Home", "Data View", "EDA", "Forecasting Models", "Model Evaluation", "Power BI Dashboard"]
    for item in menu_items:
        if st.button(item, use_container_width=True):
            st.session_state.active_page = item

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

    st.title("Time Series Analysis with Cryptocurrency")

    col1, col2 = st.columns([0.65, 0.35])

    with col1:
        st.markdown("<span style='color:#ffcc00;'>Project Description</span>", unsafe_allow_html=True)
        st.write("""
        This project focuses on cryptocurrency price forecasting using Binance API data.
        Exploratory Data Analysis (EDA) was performed to analyze trends and volatility.
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
        st.markdown("### <span style='color:#ffcc00;'>Team Members</span>", unsafe_allow_html=True)
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

    if st.button("Run Forecasting Models"):

        df_model = df[['timestamp','close']].rename(columns={'timestamp':'ds','close':'y'})
        train_size = int(len(df_model)*0.8)
        train, test = df_model[:train_size], df_model[train_size:]

        # Prophet
        model_p = Prophet()
        model_p.fit(train)
        future = model_p.make_future_dataframe(periods=len(test))
        forecast = model_p.predict(future)
        prophet_pred = forecast['yhat'][-len(test):]
        prophet_metrics = evaluate_model(test['y'], prophet_pred, "Prophet")

        # ARIMA
        arima_model = ARIMA(train['y'], order=(5,1,0)).fit()
        arima_pred = arima_model.forecast(steps=len(test))
        arima_metrics = evaluate_model(test['y'], arima_pred, "ARIMA")

        # SARIMA
        sarima_model = SARIMAX(train['y'], order=(2,1,2), seasonal_order=(1,1,1,12)).fit(disp=False)
        sarima_pred = sarima_model.forecast(steps=len(test))
        sarima_metrics = evaluate_model(test['y'], sarima_pred, "SARIMA")

        st.session_state["model_metrics"] = [prophet_metrics, arima_metrics, sarima_metrics]
        st.success("Models Trained Successfully!")

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------
elif page == "Model Evaluation":

    st.subheader("Model Evaluation Metrics")

    if "model_metrics" in st.session_state:
        metrics_df = pd.DataFrame(st.session_state["model_metrics"])
        metrics_df = metrics_df.sort_values(by="RMSE")
        st.dataframe(metrics_df, use_container_width=True)

        best_model = metrics_df.iloc[0]["Model"]
        st.success(f"Best Model (Lowest RMSE): {best_model}")

        fig, ax = plt.subplots()
        ax.bar(metrics_df["Model"], metrics_df["RMSE"])
        st.pyplot(fig)
    else:
        st.info("Run Forecasting Models first.")

# --------------------------------------------------
# POWER BI
# --------------------------------------------------
elif page == "Power BI Dashboard":

    st.markdown("## Interactive Power BI Dashboard")

    powerbi_url = "https://app.powerbi.com/view?r=eyJrIjoiYjA3YWQyN2MtMDM4ZC00YWUxLTlkNGQtNWIxYTc2MTZiZTI1IiwidCI6IjM0YTYzMzMwLWU2MWUtNGMwZC04ODIyLTQ4MjViZTk0YTNkYiJ9"

    components.iframe(powerbi_url, width=1200, height=650)
