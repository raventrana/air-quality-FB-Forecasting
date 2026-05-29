import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly

st.title("Air Quality Forecasting App")
st.write("Upload an Air Quality UCI CSV file to clean data and forecast CO(GT) levels.")

# 1. File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # 2. Data Collection & Cleaning (from your notebook)
    # Using specific separator and decimal separator as noted in your notebook
    df = pd.read_csv(uploaded_file, sep=';', decimal=',')
    
    # Drop the trailing empty columns and rows
    df = df.iloc[:, :-2]
    df = df.head(9357)
    
    # Handle missing values labeled as -200
    df = df.replace(to_replace=-200, value=np.nan)
    df = df.fillna(df.mean(numeric_only=True))
    
    # Process DateTime strings cleanly
    time_info = df['Time'].apply(lambda x: str(x).replace('.', ':'))
    
    # Combine the strings first, then parse specifying dayfirst=True since the CSV is DD/MM/YYYY
    combined_datetime = df['Date'].astype(str) + ' ' + time_info
    df['DateTime'] = pd.to_datetime(combined_datetime, dayfirst=True, errors='coerce')
    
    st.success("Data successfully loaded and cleaned!")
    st.write("### Preview of Cleaned Data", df.head())

    # 3. Prepare data for Prophet
    prophet_df = pd.DataFrame()
    prophet_df['ds'] = df['DateTime']
    prophet_df['y'] = df['CO(GT)']

    # 4. Forecasting Section
    st.subheader("Forecast Configuration")
    days_to_predict = st.slider("Days to forecast into the future:", min_value=1, max_value=30, value=7)
    
    if st.button("Run Forecast"):
        with st.spinner("Training Prophet model..."):
            # Initialize and fit model
            m = Prophet()
            m.fit(prophet_df)
            
            # Create future dataframe (converting days to hours since data is hourly)
            future = m.make_future_dataframe(periods=days_to_predict * 24, freq='h')
            forecast = m.predict(future)
            
        st.success("Forecasting complete!")
        
        # 5. Interactive Visualization
        st.subheader("Forecast Graph")
        fig = plot_plotly(m, forecast)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show predicted values data frame
        st.subheader("Predicted Data View")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
