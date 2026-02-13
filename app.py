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
from PIL import Image
import os

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
        df['timestamp'] = pd.to_datetime(df['timestamp'])
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

page = st.session_state.active_page

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "Home":

    st.title("Time Series Analysis with Cryptocurrency")

    col1, col2 = st.columns([0.65, 0.35])

    with col1:
        st.markdown("### 📌 Project Description")
        st.write("""
        This project performs cryptocurrency price forecasting using Binance data.
        Models implemented include Prophet, ARIMA, SARIMA, and LSTM.
        """)

        st.markdown("### 📈 Forecasting Models")
        st.markdown("""
        - Prophet  
        - ARIMA  
        - SARIMA  
        - LSTM
        """)

    with col2:
        try:
            image = Image.open("binance2.webp")
            st.image(image, use_column_width=True)
        except:
            st.warning("Image not found.")

        st.markdown("---")
        st.markdown("### 👥 Team Members")
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
    st.subheader("Dataset")
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    else:
        st.warning("Dataset not loaded.")

# --------------------------------------------------
# EDA
# --------------------------------------------------
elif page == "EDA":

    st.subheader("Exploratory Data Analysis")

    if df.empty:
        st.warning("Dataset not loaded.")
    else:

        tab1, tab2, tab3 = st.tabs(["Price Trend", "Volume Trend", "Correlation Heatmap"])

        with tab1:
            fig, ax = plt.subplots()
            ax.plot(df['timestamp'], df['close'])
            ax.set_title("Closing Price Over Time")
            st.pyplot(fig)

        with tab2:
            fig, ax = plt.subplots()
            ax.plot(df['timestamp'], df['volume'])
            ax.set_title("Volume Trend")
            st.pyplot(fig)

        with tab3:
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

# --------------------------------------------------
# FORECASTING MODELS
# --------------------------------------------------
elif page == "Forecasting Models":

    st.subheader("Run Forecasting Models")

    if st.button("Run Models"):

        df_model = df[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
        train_size = int(len(df_model) * 0.8)
        train, test = df_model[:train_size], df_model[train_size:]

        # Prophet
        model_p = Prophet()
        model_p.fit(train)
        future = model_p.make_future_dataframe(periods=len(test))
        forecast = model_p.predict(future)
        prophet_pred = forecast['yhat'][-len(test):].values
        prophet_metrics = evaluate_model(test['y'], prophet_pred, "Prophet")

        # ARIMA
        arima_model = ARIMA(train['y'], order=(5,1,0)).fit()
        arima_pred = arima_model.forecast(steps=len(test))
        arima_metrics = evaluate_model(test['y'], arima_pred, "ARIMA")

        # SARIMA
        sarima_model = SARIMAX(train['y'], order=(2,1,2), seasonal_order=(1,1,1,12)).fit(disp=False)
        sarima_pred = sarima_model.forecast(steps=len(test))
        sarima_metrics = evaluate_model(test['y'], sarima_pred, "SARIMA")

        # LSTM
        prices = df['close'].values.reshape(-1, 1)
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
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        lstm_pred = model.predict(X_test)
        lstm_pred = scaler.inverse_transform(lstm_pred)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        lstm_metrics = evaluate_model(actual, lstm_pred, "LSTM")

        st.session_state["model_metrics"] = [
            prophet_metrics, arima_metrics, sarima_metrics, lstm_metrics
        ]

        st.success("Models Trained Successfully!")

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------
elif page == "Model Evaluation":

    st.subheader("Model Evaluation")

    if "model_metrics" in st.session_state:

        metrics_df = pd.DataFrame(st.session_state["model_metrics"])
        metrics_df = metrics_df.sort_values(by="RMSE")

        st.dataframe(metrics_df, use_container_width=True)

        best_model = metrics_df.iloc[0]["Model"]
        st.success(f"🏆 Best Model (Lowest RMSE): {best_model}")

        fig, ax = plt.subplots()
        ax.bar(metrics_df["Model"], metrics_df["RMSE"])
        ax.set_ylabel("RMSE")
        st.pyplot(fig)

    else:
        st.info("Please run models first.")

# --------------------------------------------------
# POWER BI DASHBOARD
# --------------------------------------------------
elif page == "Power BI Dashboard":

    st.markdown("## 📊 Interactive Power BI Dashboard")

    powerbi_url = "https://app.powerbi.com/view?r=eyJrIjoiYjA3YWQyN2MtMDM4ZC00YWUxLTlkNGQtNWIxYTc2MTZiZTI1IiwidCI6IjM0YTYzMzMwLWU2MWUtNGMwZC04ODIyLTQ4MjViZTk0YTNkYiJ9"

    components.iframe(powerbi_url, width=1200, height=650)
