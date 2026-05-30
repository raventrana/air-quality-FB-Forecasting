import streamlit as st
import pandas as pd
import numpy as np
import os
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

st.title("Air Quality Forecasting App")
st.write("Analyze and forecast pollutant levels using historical Air Quality UCI dataset.")

# --- Dataset Citation & Info ---
with st.expander("ℹ️ Dataset Information & Source Citation", expanded=False):
    st.markdown("""
    **Dataset:** [Air Quality Data Set](https://archive.ics.uci.edu/dataset/360/air+quality)
    **Source:** UCI Machine Learning Repository
    
    **Description:** This dataset contains 9,358 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The device was located on the field in a significantly polluted area, at road level, within an Italian city. Data was recorded from **March 2004 to April 2005** (one full year).
    
    **Ground Truth Reference:** Cross-references are provided for co-located reference certified analyzers measuring Carbon Monoxide ($CO$), Non-Methane Hydrocarbons ($NMHC$), Benzene ($C_6H_6$), Total Nitrogen Oxides ($NO_x$), and Nitrogen Dioxide ($NO_2$).
    """)

# --- 1. Data Loading (Local or Uploaded) ---
LOCAL_FILE = "AirQualityUCI.csv"

if os.path.exists(LOCAL_FILE):
    raw_df = pd.read_csv(LOCAL_FILE, sep=';', decimal=',')
    st.info(f"Loaded `{LOCAL_FILE}` directly from local storage (UCI Air Quality Dataset).")
else:
    uploaded_file = st.file_uploader("Local AirQualityUCI.csv not found. Please upload it manually:", type="csv")
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file, sep=';', decimal=',')
    else:
        st.warning("Please place `AirQualityUCI.csv` in the app directory or upload it above to proceed.")
        st.stop()

# --- 2. Data Cleaning Pipeline ---
before_shape = raw_df.shape
before_missing = (raw_df == -200).sum().sum() + raw_df.isna().sum().sum()

df = raw_df.iloc[:, :-2]
df = df.head(9357)

df = df.replace(to_replace=-200, value=np.nan)
df = df.fillna(df.mean(numeric_only=True))

time_info = df['Time'].apply(lambda x: str(x).replace('.', ':'))
combined_datetime = df['Date'].astype(str) + ' ' + time_info
df['DateTime'] = pd.to_datetime(combined_datetime, dayfirst=True, errors='coerce')
df = df.dropna(subset=['DateTime'])

after_shape = df.shape
after_missing = df.isna().sum().sum()


# --- 3. UI Dashboard Layout ---
st.header("📊 Data Lifecycle & Cleaning Status")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Before Cleaning")
    st.write(f"**Dimensions:** {before_shape[0]} rows × {before_shape[1]} columns")
    st.write(f"**Missing values (incl. -200):** {before_missing}")
    st.dataframe(raw_df.head(3), use_container_width=True)

with col2:
    st.subheader("After Cleaning")
    st.write(f"**Dimensions:** {after_shape[0]} rows × {after_shape[1]} columns")
    st.write(f"**Missing values remaining:** {after_missing}")
    st.dataframe(df.head(3), use_container_width=True)

st.write("### Data Cleaning Pipeline Steps")
st.graphviz_chart("""
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, color=lightblue, fontname="Helvetica"];
        A [label="Raw UCI CSV Input"] -> B [label="Drop Trailing Blank Columns"];
        B -> C [label="Replace -200 with NaN & Impute Means"];
        C -> D [label="Merge & Format [Date + Time] String"];
        D -> E [label="Final Cleaned Dataset"];
    }
""")

st.markdown("---")

# --- 4. Dynamic Target and Forecast Configurations ---
st.header("🔮 Forecast Configuration")

available_targets = [col for col in df.columns if col not in ['Date', 'Time', 'DateTime']]
target_var = st.selectbox("Select target variable (y) to forecast:", options=available_targets, index=0)

days_to_predict = st.slider("Days to forecast into the future:", min_value=1, max_value=30, value=7)

prophet_df = pd.DataFrame()
prophet_df['ds'] = df['DateTime']
prophet_df['y'] = df[target_var]

if st.button("Run Model Engine"):
    with st.spinner(f"Training Prophet model for {target_var}..."):
        m = Prophet()
        m.fit(prophet_df)
        
        future = m.make_future_dataframe(periods=days_to_predict * 24, freq='h')
        forecast = m.predict(future)
        
    st.success("Forecasting complete!")
    
    st.subheader(f"Forecast Graph for {target_var}")
    fig_forecast = plot_plotly(m, forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.subheader("📅 Seasonality & Trend Component Analysis")
    st.write("Breaks down the timeline into underlying hourly, weekly, and yearly trends.")
    fig_components = plot_components_plotly(m, forecast)
    st.plotly_chart(fig_components, use_container_width=True)
    
    st.subheader("📋 Predicted Data View (Latest Intervals)")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))
