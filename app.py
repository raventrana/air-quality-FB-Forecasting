import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

st.set_page_config(page_title="Ultimate Air Quality Hub", layout="wide")

st.title("Comprehensive Air Quality Analytics & Forecasting Hub")
st.write("Upload any environmental CSV dataset to automatically clean, parse, and scale predictions across metrics concurrently.")

# 1. Flexible File Uploader
uploaded_file = st.file_uploader("Choose an Air Quality CSV file", type="csv")

if uploaded_file is not None:
    # --- FLEXIBLE DATA CLEANING PIPELINE ---
    try:
        # Read first line to inspect delimiter tokens
        first_line = uploaded_file.readline().decode('utf-8')
        uploaded_file.seek(0) # reset file pointer
        
        # Check for alternative separator types
        if ';' in first_line and ',' in first_line:
            df = pd.read_csv(uploaded_file, sep=';', decimal=',')
        elif ';' in first_line:
            df = pd.read_csv(uploaded_file, sep=';')
        else:
            df = pd.read_csv(uploaded_file, sep=',')
    except Exception as e:
        st.error(f"Error parsing file structure: {e}")
        st.stop()

    # Clean empty margins
    df = df.dropna(how='all', axis=0)
    df = df.dropna(how='all', axis=1)
    for bad_col in ['Unnamed: 15', 'Unnamed: 16']:
        if bad_col in df.columns: 
            df.drop(columns=[bad_col], errors='ignore', inplace=True)

    st.subheader("Data Processing & Diagnostics")
    
    # Smart DateTime Reconstruction
    if 'Date' in df.columns and 'Time' in df.columns:
        time_info = df['Time'].astype(str).apply(lambda x: x.replace('.', ':'))
        combined_dt = df['Date'].astype(str) + ' ' + time_info
        df['DateTime'] = pd.to_datetime(combined_dt, dayfirst=True, errors='coerce')
        df.drop(columns=['Date', 'Time'], errors='ignore', inplace=True)
        st.info("🔄 Reconstructed explicit timestamp vector by combining 'Date' and 'Time' matrices.")
    else:
        dt_col = None
        for col in df.columns:
            if col.lower() in ['date', 'time', 'timestamp', 'datetime', 'sampling_date']:
                dt_col = col
                break
                
        if dt_col:
            df['DateTime'] = pd.to_datetime(df[dt_col], dayfirst=True, errors='coerce')
            if dt_col != 'DateTime':
                df.drop(columns=[dt_col], errors='ignore', inplace=True)
            st.info(f"🔄 Automatically aligned timeline structure from native parameter: '{dt_col}'")
        else:
            st.error("🚨 Missing vital Date/Time index matrix framework.")
            st.stop()

    df = df.dropna(subset=['DateTime'])
    df = df.sort_values(by='DateTime').reset_index(drop=True)

    # Multi-Step Sensor Cleansing Outliers
    df = df.replace(to_replace=-200, value=np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.error("🚨 Dataset lacks numeric parameters to execute target processing configurations.")
        st.stop()

    # Drop zero out sensor dropouts (Air metrics shouldn't hit structural complete absolute zero)
    for col in numeric_cols:
        invalid_mask = (df[col] <= 0.0)
        if invalid_mask.sum() > 0:
            df.loc[invalid_mask, col] = np.nan

    # Fill empty values using column means
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    st.success("✅ Complete cleaning pipeline executed: Data boundaries checked, zeros corrected, missing metrics imputed.")
    
    with st.expander("🔍 View Cleaned Data Frame Matrix"):
        st.write(df)

    # --- STEP 4: MASSIVE MULTI-VARIABLE CONFIGURATION MAP ---
    st.markdown("---")
    st.subheader("Bulk Forecasting Configuration Engine")
    
    # Multi-select options allows selecting ALL features together cleanly
    selected_targets = st.multiselect(
        "Select all the Air Quality metrics you want to forecast simultaneously:",
        options=numeric_cols,
        default=[numeric_cols[0]] # defaults to the first one available
    )
    
    days_to_predict = st.slider("Forecast window range depth (Days ahead):", min_value=1, max_value=60, value=14)

    # Dynamically extract timeframe frequency delta step signature
    if len(df) > 1:
        time_delta = df['DateTime'].iloc[1] - df['DateTime'].iloc[0]
        freq_setting = 'h' if time_delta.total_seconds() < 86400 else 'D'
    else:
        freq_setting = 'D'
        
    periods_setting = days_to_predict * 24 if freq_setting == 'h' else days_to_predict

    if len(selected_targets) == 0:
        st.warning("⚠️ Please select at least one air quality target parameter above to process analytics.")
    else:
        if st.button("🚀 Run Comprehensive Predictive Analytics Pipeline"):
            # Create interactive layout tabs to hold each metric neatly
            tab_objs = st.tabs([f"📊 {target}" for target in selected_targets])
            
            for index, target_variable in enumerate(selected_targets):
                with tab_objs[index]:
                    st.markdown(f"## Trend Horizon Analysis for `{target_variable}`")
                    
                    # Prepare dedicated subset split
                    prophet_df = pd.DataFrame()
                    prophet_df['ds'] = df['DateTime']
                    prophet_df['y'] = df[target_variable]
                    
                    # Prevent model optimization failures if features are heavily flat/empty
                    if prophet_df['y'].nunique() <= 1:
                        st.error(f"❌ Invalid target bounds: {target_variable} contains entirely flat or missing variances.")
                        continue
                        
                    with st.spinner(f"Computing historical seasonality profiles for {target_variable}..."):
                        try:
                            # Model instantiation and runtime optimization execution
                            m = Prophet()
                            m.fit(prophet_df)
                            
                            future = m.make_future_dataframe(periods=periods_setting, freq=freq_setting)
                            forecast = m.predict(future)
                            
                            st.success(f"Predictions processed for {target_variable}!")
                            
                            # Figure 1: Main Forecast Visualization
                            st.subheader("🎯 Primary Forecast Horizon")
                            fig1 = plot_plotly(m, forecast)
                            st.plotly_chart(fig1, use_container_width=True)
                            
                            # Figure 2: Dynamic Core Trend Components (New Additions)
                            st.subheader("⏳ Time-Series Seasonality Components")
                            fig2 = plot_components_plotly(m, forecast)
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            # Matrix calculations view display
                            st.subheader("📋 Future Predicted Target Core Boundaries")
                            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(max(5, days_to_predict)))
                            
                        except Exception as calc_error:
                            st.error(f"Error compiling prediction intervals for target variable {target_variable}: {calc_error}")
