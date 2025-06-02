import streamlit as st
import pandas as pd
import logging
from trading_bot import fetch_data, calculate_indicators, generate_signals, create_chart

# Configure logging - Optional for Streamlit app, but can be helpful
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    st.set_page_config(layout="wide")
    st.title("Currency Trading Bot Visualization")

    currency_pairs = ["EURUSD=X", "AUDUSD=X", "GBPUSD=X", "USDJPY=X", "USDCAD=X", "BTC-USD", "ETH-USD"]
    selected_pair = st.selectbox("Select currency pair", currency_pairs)

    if selected_pair:
        st.subheader(f"Analysis for {selected_pair}")

        # Step 1: Fetch Data
        data_load_state = st.text(f"Fetching data for {selected_pair}...")
        raw_data = fetch_data(selected_pair)

        if raw_data is None or raw_data.empty:
            data_load_state.error(f"Failed to fetch data for {selected_pair}. Please check the ticker or try again later.")
            st.error(f"Data fetching failed for {selected_pair}.")
            return

        # Limit data for performance if necessary, e.g., last 500 periods for charting
        # yfinance already fetches 30d/1h, which is ~720 points. Charting all might be slow.
        # The create_chart function in trading_bot itself slices the last 200 periods.
        data_load_state.text(f"Data fetched for {selected_pair}. Calculating indicators...")

        # Step 2: Calculate Indicators
        # Ensure enough data for the longest indicator window (EMA50)
        # calculate_indicators itself checks for length for each indicator.
        data_with_indicators = calculate_indicators(raw_data.copy())

        if data_with_indicators is None or data_with_indicators.empty:
            data_load_state.error(f"Failed to calculate indicators for {selected_pair}.")
            st.error("Indicator calculation failed.")
            return

        # Check if essential indicator columns were actually created (they might not if data was too short)
        # We need at least 'Close', 'RSI', 'MACD', 'MACD_Signal', 'EMA20', 'EMA50' for signals and chart.
        # The generate_signals and create_chart functions have their own checks.

        data_load_state.text(f"Indicators calculated for {selected_pair}. Generating signals...")

        # Step 3: Generate Signals
        # The generate_signals function expects data that has already skipped initial NaN periods from indicators.
        # Longest indicator window is EMA50 (50 periods).
        min_indicator_lookback = 50
        if len(data_with_indicators) <= min_indicator_lookback:
            st.warning(f"Data for {selected_pair} is too short ({len(data_with_indicators)} rows) after indicator calculation for reliable signal generation. Need > {min_indicator_lookback} rows.")
            # Optionally, still try to chart what we have, or return.
            # For now, let's try to chart the raw data with available indicators if signals can't be made.
            fig = create_chart(data_with_indicators.tail(200).copy(), selected_pair) # Chart last 200
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to create chart even with partial data.")
            return

        data_for_signals = data_with_indicators.iloc[min_indicator_lookback:].copy()
        if data_for_signals.empty:
            st.warning(f"Not enough data remaining after filtering for signal generation for {selected_pair}.")
            fig = create_chart(data_with_indicators.tail(200).copy(), selected_pair)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to create chart for partial data.")
            return

        data_with_signals = generate_signals(data_for_signals)
        data_load_state.text(f"Signals generated for {selected_pair}. Creating chart...")

        # Step 4: Create Chart
        # create_chart from trading_bot.py now returns a figure object.
        # It also internally slices to the last 200 periods by default from its main example,
        # but here we pass the already processed (and potentially sliced) data_with_signals.
        # Let's ensure we are passing a dataframe that create_chart can handle.
        # We'll use the data_with_signals which is already sliced from min_indicator_lookback.
        # For charting, it's good practice to chart a reasonable number of points.
        # The create_chart function in trading_bot.py takes the df and currency_pair.

        # If data_with_signals is very long, slice it for charting performance.
        # However, create_chart in trading_bot.py was modified to chart `df.tail(200)` from its main.
        # Here, we pass the df that might be already significantly processed.
        # Let's rely on the robustness of create_chart or pass a consistently sliced df.

        # The `data_with_signals` dataframe has its index aligned with the original `raw_data`
        # but starts from `min_indicator_lookback`. We need to ensure OHLC data is present for charting.
        # `create_chart` expects OHLC to be in the df. `generate_signals` only adds a 'signal' column.
        # So, `data_with_signals` should be fine.

        # Let's ensure the df passed to create_chart is not empty.
        if data_with_signals.empty:
            st.error(f"No data available for charting after signal generation for {selected_pair}.")
            return

        fig = create_chart(data_with_signals.copy(), selected_pair) # Pass a copy

        if fig:
            data_load_state.success(f"Chart created for {selected_pair}. Displaying now.")
            st.plotly_chart(fig, use_container_width=True)
        else:
            data_load_state.error(f"Failed to create chart for {selected_pair}.")
            st.error("Chart creation failed.")
            # As a fallback, show some data if chart fails
            st.subheader("Data with Signals (if available)")
            st.dataframe(data_with_signals.tail())


if __name__ == "__main__":
    main()
