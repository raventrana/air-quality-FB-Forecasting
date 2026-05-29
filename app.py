import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly

st.title("Flexible Air Quality Forecasting App")
st.write("Upload any Air Quality CSV file to automatically clean data and forecast trends.")

# 1. Flexible File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # --- FLEXIBLE DATA CLEANING PIPELINE ---
    
    # Step A: Robust Delimiter Detection & Parsing
    try:
        # Read first line to inspect delimiter tokens
        first_line = uploaded_file.readline().decode('utf-8')
        uploaded_file.seek(0) # reset file pointer
        
        # If semicolons are present but commas aren't splitting properly, adapt dynamically
        if ';' in first_line and ',' in first_line:
            df = pd.read_csv(uploaded_file, sep=';', decimal=',')
        elif ';' in first_line:
            df = pd.read_csv(uploaded_file, sep=';')
        else:
            df = pd.read_csv(uploaded_file, sep=',')
    except Exception as e:
        st.error(format(f"Error parsing file structure: {e}"))
        st.stop()

    # Step B: Remove entirely empty trailing rows/columns if they exist
    df = df.dropna(how='all', axis=0)
    df = df.dropna(how='all', axis=1)
    if 'Unnamed: 15' in df.columns: df.drop(columns=['Unnamed: 15'], errors='ignore', inplace=True)
    if 'Unnamed: 16' in df.columns: df.drop(columns=['Unnamed: 16'], errors='ignore', inplace=True)

    # Step C: Smart DateTime Stamp Reconstruction
    st.subheader("Data Cleaning Log")
    
    # Case 1: Separate 'Date' and 'Time' columns exist (like your original UCI dataset)
    if 'Date' in df.columns and 'Time' in df.columns:
        time_info = df['Time'].astype(str).apply(lambda x: x.replace('.', ':'))
        combined_dt = df['Date'].astype(str) + ' ' + time_info
        df['DateTime'] = pd.to_datetime(combined_dt, dayfirst=True, errors='coerce')
        df.drop(columns=['Date', 'Time'], errors='ignore', inplace=True)
        st.info("🔄 Combined separate 'Date' and 'Time' columns into a single timestamp.")
        
    # Case 2: A single column name looks like date/time (e.g., 'Date', 'Timestamp', 'DateTime')
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
            st.info(f"🔄 Automatically detected and parsed timestamp from column: '{dt_col}'")
        else:
            st.error("🚨 Could not identify a valid Date/Time column. Please check your CSV column headers.")
            st.stop()

    # Drop rows where timestamp parsing completely failed
    df = df.dropna(subset=['DateTime'])

    # Step D: Dynamic Multi-Step Air Quality Value Cleaning
    # 1. Handle explicit placeholder null flags (-200)
    df = df.replace(to_replace=-200, value=np.nan)
    
    # Get all numeric columns to apply targeted rules
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.error("🚨 No numeric parameters found in the file to forecast.")
        st.stop()

    # 2. Treat Negative Values and complete Zero Values (0.0) as missing data (NaN)
    # Note: complete 0.0 or negative readings in Air Quality are usually sensor dropout errors
    for col in numeric_cols:
        invalid_mask = (df[col] <= 0.0)
        if invalid_mask.sum() > 0:
            df.loc[invalid_mask, col] = np.nan

    # 3. Handle Remaining NaNs by imputing with the column mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    st.info("✅ Replaced all placeholder flags (-200), negative values, and complete 0.0 dropouts with column mean averages.")

    # Show summary preview
    st.success("Data successfully optimized!")
    st.write("### Cleaned Data Preview", df.head())

    # --- STEP 4: DYNAMIC FORECAST TARGET SELECTION ---
    st.markdown("---")
    st.subheader("Interactive Forecasting Options")
    
    # Allow the user to select ANY valid clean numeric column for Prophet
    target_variable = st.selectbox(
        "Select the air quality target metric you want to forecast ('y'):",
        options=numeric_cols
    )

    # Reconstruct the explicit dataframe needed by Prophet
    prophet_df = pd.DataFrame()
    prophet_df['ds'] = df['DateTime']
    prophet_df['y'] = df[target_variable]

    # Forecast configuration sliders
    days_to_predict = st.slider("Days to forecast into the future:", min_value=1, max_value=30, value=7)
    
    # Determine frequency context (Hourly vs Daily) dynamically based on dataset intervals
    time_delta = df['DateTime'].iloc[1] - df['DateTime'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
    freq_setting = 'h' if time_delta.total_seconds() < 86400 else 'D'
    periods_setting = days_to_predict * 24 if freq_setting == 'h' else days_to_predict

    if st.button("Proceed with Forecasting"):
        with st.spinner(f"Training Prophet model to predict {target_variable}..."):
            # Initialize and fit model
            m = Prophet()
            m.fit(prophet_df)
            
            # Generate future layout mapping intervals cleanly
            future = m.make_future_dataframe(periods=periods_setting, freq=freq_setting)
            forecast = m.predict(future)
            
        st.success(f"Forecasting complete for {target_variable}!")
        
        # Plotly graph component visualization
        st.subheader("Forecast Visual Graph Map")
        fig = plot_plotly(m, forecast)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display explicit prediction dataframe matrix numbers
        st.subheader("Forecast Data View matrix")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
