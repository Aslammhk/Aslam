import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def fetch_data(currency_pair: str) -> pd.DataFrame | None:
    """
    Fetches the last 30 days of historical data with a 1-hour interval for a given currency pair.

    Args:
        currency_pair: The currency pair symbol (e.g., "AUDUSD=X").

    Returns:
        A pandas DataFrame with historical data, or None if an error occurs.
    """
    print(f"Fetching data for {currency_pair}...")
    try:
        ticker = yf.Ticker(currency_pair)
        # Fetch data: 30 days, 1-hour interval
        # Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        # Max period for 1h interval is 730 days.
        data = ticker.history(period="30d", interval="1h")

        if data.empty:
            print(f"No data found for {currency_pair}. Ticker might be invalid or delisted.")
            return None

        print(f"Successfully fetched {len(data)} data points for {currency_pair}.")
        return data
    except Exception as e:
        print(f"Error fetching data for {currency_pair}: {e}")
        return None

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Calculates RSI, MACD, EMA20, and EMA50 indicators and adds them to the DataFrame.

    Args:
        df: Pandas DataFrame with historical price data (must include 'Close' column).

    Returns:
        DataFrame with calculated indicators, or None if input is invalid.
    """
    if df is None or df.empty:
        print("Input DataFrame is None or empty. Cannot calculate indicators.")
        return None

    if 'Close' not in df.columns:
        print("'Close' column not found in DataFrame. Cannot calculate indicators.")
        return None

    print("Calculating technical indicators...")

    # RSI (Relative Strength Index)
    # Default window for RSI is 14
    min_rsi_period = 14
    if len(df) < min_rsi_period:
        print(f"DataFrame has less than {min_rsi_period} rows. Cannot calculate RSI with window {min_rsi_period}.")
    else:
        try:
            rsi_indicator = ta.momentum.RSIIndicator(close=df['Close'], window=min_rsi_period, fillna=False)
            df['RSI'] = rsi_indicator.rsi()
            print("RSI calculated.")
        except Exception as e:
            print(f"Error calculating RSI: {e}")

    # MACD (Moving Average Convergence Divergence)
    # Default windows: fast=12, slow=26, signal=9
    min_macd_period = 26 # MACD slow window is the longest period needed
    if len(df) < min_macd_period:
        print(f"DataFrame has less than {min_macd_period} rows. Cannot calculate MACD.")
    else:
        try:
            macd_indicator = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
            df['MACD'] = macd_indicator.macd()
            df['MACD_Signal'] = macd_indicator.macd_signal()
            df['MACD_Hist'] = macd_indicator.macd_diff()
            print("MACD, MACD Signal, and MACD Histogram calculated.")
        except Exception as e:
            print(f"Error calculating MACD: {e}")

    # EMA (Exponential Moving Average)
    min_ema20_period = 20
    if len(df) < min_ema20_period:
        print(f"DataFrame has less than {min_ema20_period} rows. Cannot calculate EMA20.")
    else:
        try:
            ema20_indicator = ta.trend.EMAIndicator(close=df['Close'], window=min_ema20_period, fillna=False)
            df['EMA20'] = ema20_indicator.ema_indicator()
            print("EMA20 calculated.")
        except Exception as e:
            print(f"Error calculating EMA20: {e}")

    min_ema50_period = 50
    if len(df) < min_ema50_period:
        print(f"DataFrame has less than {min_ema50_period} rows. Cannot calculate EMA50.")
    else:
        try:
            ema50_indicator = ta.trend.EMAIndicator(close=df['Close'], window=min_ema50_period, fillna=False)
            df['EMA50'] = ema50_indicator.ema_indicator()
            print("EMA50 calculated.")
        except Exception as e:
            print(f"Error calculating EMA50: {e}")

    print("Finished calculating indicators.")
    return df

if __name__ == "__main__":
    print("Trading bot script starting...")

    # Example usage (can be uncommented for direct testing)
    # sample_pair = "EURUSD=X" # Use a common pair for testing
    # raw_data = fetch_data(sample_pair)

    # if raw_data is not None:
    #     print(f"\nRaw data for {sample_pair} (first 5 rows):")
    #     print(raw_data.head())

    #     data_with_indicators = calculate_indicators(raw_data.copy()) # Use .copy() to avoid modifying original

    #     if data_with_indicators is not None:
    #         print(f"\nData with indicators for {sample_pair} (last 5 rows):")
    #         # Display columns that are likely to exist, plus new indicators
    #         cols_to_show = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'EMA20', 'EMA50']
    #         # Filter out columns that might not have been created if data was too short
    #         existing_cols_to_show = [col for col in cols_to_show if col in data_with_indicators.columns]
    #         print(data_with_indicators[existing_cols_to_show].tail())
    # else:
    #     print(f"Could not retrieve or process data for {sample_pair}")

    # Example including signal generation and charting
    sample_pair = "EURUSD=X"
    raw_data = fetch_data(sample_pair) # Fetches 30d, 1h data

    if raw_data is not None:
        print(f"\nRaw data for {sample_pair} (first 5 rows):")
        print(raw_data.head())

        data_with_indicators = calculate_indicators(raw_data.copy()) # Use .copy()

        if data_with_indicators is not None:
            # Filter out rows where indicators might still be NaN (especially at the beginning)
            # before passing to signal generation
            min_indicator_period = 50 # Longest window used for indicators
            # Ensure there's enough data AFTER this filtering for signal calculation (which itself needs some lookback)
            if len(data_with_indicators) > min_indicator_period + 5: # +5 for some buffer for signal calc
                data_for_signals = data_with_indicators.iloc[min_indicator_period:].copy()

                if not data_for_signals.empty:
                    data_with_signals = generate_signals(data_for_signals) # Pass the copy
                    print(f"\nData with signals for {sample_pair} (last 15 rows with signals):")

                    cols_to_show = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'EMA20', 'EMA50', 'signal']
                    existing_cols_to_show = [col for col in cols_to_show if col in data_with_signals.columns]

                    # Show rows where a signal is present
                    signalled_rows = data_with_signals[data_with_signals['signal'] != ""]
                    if not signalled_rows.empty:
                        print(signalled_rows[existing_cols_to_show].tail(15))
                        # Create chart only if signals were generated and data is available
                        # Use a smaller slice of data for charting if it's too large, e.g., last 100-200 periods
                        chart_df = data_with_signals.tail(200).copy() # Chart last 200 periods
                        create_chart(chart_df, sample_pair)
                    else:
                        print("No signals generated in the last period shown, or in data.")
                        print(data_with_signals[existing_cols_to_show].tail(5)) # Show last 5 rows anyway
                        # Still try to create a chart for the last 200 periods even without signals
                        chart_df = data_with_signals.tail(200).copy()
                        create_chart(chart_df, sample_pair)
                else:
                    print("Not enough data after indicator calculation to generate signals.")
            else:
                print(f"Dataframe too short ({len(data_with_indicators)} rows) even after indicator calculation for reliable signal generation and charting.")
        else:
            print(f"Could not calculate indicators for {sample_pair}")
    else:
        print(f"Could not retrieve data for {sample_pair}")

    print("Trading bot script finished.")


def create_chart(df: pd.DataFrame, currency_pair: str):
    """
    Creates a Plotly chart with candlestick, volume, RSI, EMAs, and trading signals,
    and saves it to an HTML file.

    Args:
        df: Pandas DataFrame with price data, indicators, and signals.
        currency_pair: The currency pair symbol (e.g., "AUDUSD=X") for the chart title and filename.
    """
    if df is None or df.empty:
        print("Input DataFrame is None or empty. Cannot create chart.")
        return

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'EMA20', 'EMA50', 'signal']
    # Check if all required columns are present and not entirely NaN
    missing_or_empty_cols = []
    for col in required_cols:
        if col not in df.columns:
            missing_or_empty_cols.append(col + " (missing)")
        elif df[col].isnull().all(): # Check if the column exists but is all NaN
             # For 'signal' column, it's okay if it's all empty strings or NaN if no signals generated
            if col == 'signal' and (df[col].replace("", pd.NA).isnull().all()): # Treat empty strings as NaN for this check
                pass # Allow 'signal' to be all empty/NaN
            else:
                missing_or_empty_cols.append(col + " (all NaN)")


    if missing_or_empty_cols:
        print(f"Missing, empty, or all-NaN required columns for creating chart: {missing_or_empty_cols}. Cannot create chart.")
        return

    print(f"Creating chart for {currency_pair}...")

    # Create figure with subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2], # Main chart, Volume, RSI
                        specs=[[{"secondary_y": False}],
                               [{"secondary_y": False}],
                               [{"secondary_y": False}]])

    # 1. Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Candlestick'), row=1, col=1)

    # 2. EMA Traces
    if 'EMA20' in df.columns and not df['EMA20'].isnull().all():
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines',
                                 line=dict(color='blue', width=1), name='EMA20'), row=1, col=1)
    if 'EMA50' in df.columns and not df['EMA50'].isnull().all():
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], mode='lines',
                                 line=dict(color='orange', width=1), name='EMA50'), row=1, col=1)

    # 3. Volume Trace
    if 'Volume' in df.columns and not df['Volume'].isnull().all():
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='rgba(100,100,100,0.5)'), row=2, col=1)

    # 4. RSI Trace
    if 'RSI' in df.columns and not df['RSI'].isnull().all():
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines',
                                 line=dict(color='purple', width=1), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=3, col=1)

    # 5. Signal Markers and Text
    if 'signal' in df.columns and not df['signal'].replace("", pd.NA).isnull().all(): # Check if there are any signals
        for i in range(len(df)):
            signal = df['signal'].iloc[i]
            if pd.isna(signal) or signal == "":
                continue

            marker_symbol = None
            marker_color = None
            marker_y = None
            text = ""
            text_position = ""

            # Ensure Low and High values are available for this row
            current_low = df['Low'].iloc[i]
            current_high = df['High'].iloc[i]
            if pd.isna(current_low) or pd.isna(current_high):
                continue


            if signal == "BUY" or signal == "STRONG_BUY":
                marker_symbol = 'triangle-up'
                marker_color = 'green'
                marker_y = current_low * 0.98 # Slightly below the low
                text = "BUY" if signal == "BUY" else "S-BUY"
                text_position = "bottom center"
            elif signal == "SELL" or signal == "STRONG_SELL":
                marker_symbol = 'triangle-down'
                marker_color = 'red'
                marker_y = current_high * 1.02 # Slightly above the high
                text = "SELL" if signal == "SELL" else "S-SELL"
                text_position = "top center"
            elif signal == "SPEC_BUY":
                marker_symbol = 'circle'
                marker_color = 'blue'
                marker_y = current_low * 0.985
                text = "SPEC-B"
                text_position = "bottom center"
            elif signal == "SPEC_SELL":
                marker_symbol = 'circle'
                marker_color = 'purple'
                marker_y = current_high * 1.015
                text = "SPEC-S"
                text_position = "top center"

            if marker_symbol:
                fig.add_trace(go.Scatter(
                    x=[df.index[i]],
                    y=[marker_y],
                    mode='markers+text',
                    marker=dict(symbol=marker_symbol, color=marker_color, size=10),
                    text=[text],
                    textposition=text_position,
                    textfont=dict(color=marker_color, size=10),
                    name=signal,
                    showlegend=False
                ), row=1, col=1)

    # Layout configuration
    fig.update_layout(
        title=f'{currency_pair} Trading Signals Chart (Last {len(df)} periods)',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=800
    )
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)

    fig.update_xaxes(showticklabels=True, row=1, col=1)
    fig.update_xaxes(showticklabels=True, row=2, col=1)
    fig.update_xaxes(showticklabels=True, row=3, col=1, title_text="Date")


    # Save chart to HTML
    # Sanitize currency_pair string for filename
    safe_currency_pair = currency_pair.replace('=X', '').replace('/', '_').replace('\\', '_')
    filename = f"{safe_currency_pair}_chart.html"
    try:
        fig.write_html(filename)
        print(f"Chart saved to {filename}")
    except Exception as e:
        print(f"Error saving chart to HTML: {e}")

    return fig # Return the figure object

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates trading signals based on calculated indicators.

    Args:
        df: Pandas DataFrame with price data and calculated indicators.

    Returns:
        DataFrame with an added 'signal' column.
    """
    if df is None or df.empty:
        print("Input DataFrame is None or empty. Cannot generate signals.")
        return df # Return original if it's None or empty

    required_cols = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'EMA20', 'EMA50']
    missing_cols = [col for col in required_cols if col not in df.columns or df[col].isnull().all()]

    if missing_cols:
        print(f"Missing or all-NaN required columns for generating signals: {missing_cols}. Cannot generate signals.")
        # Return original df, it might have some indicators but not all needed for signals
        return df

    print("Generating trading signals...")
    # Initialize signal column
    df['signal'] = "" # Using empty string for no signal

    # Ensure we have enough data points to look back for crosses
    # Start iterating from a point where all indicators and potential lookbacks are valid
    # For MACD cross, we need at least one previous point. For SPEC signals (3 periods back), need more.
    # Smallest window for indicators is RSI (14), MACD (26), EMA20 (20), EMA50 (50).
    # So, signals can only be reliably generated after the longest window (50) + lookback (3 for SPEC)
    start_index = max(50, 3) # Max of indicator window and lookback for SPEC

    for i in range(start_index, len(df)):
        # --- Condition Flags ---
        # Crosses: current is True (e.g. MACD > Signal), previous was False (e.g. MACD <= Signal)

        # MACD Cross
        macd_crossed_above_signal = (df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]) and \
                                    (df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1])
        macd_crossed_below_signal = (df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i]) and \
                                     (df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1])

        # Close Price vs EMA20 Cross
        close_crossed_above_ema20 = (df['Close'].iloc[i] > df['EMA20'].iloc[i]) and \
                                    (df['Close'].iloc[i-1] <= df['EMA20'].iloc[i-1])
        close_crossed_below_ema20 = (df['Close'].iloc[i] < df['EMA20'].iloc[i]) and \
                                     (df['Close'].iloc[i-1] >= df['EMA20'].iloc[i-1])

        # RSI Cross
        rsi_crossed_above_30 = (df['RSI'].iloc[i] > 30) and (df['RSI'].iloc[i-1] <= 30)
        rsi_crossed_below_70 = (df['RSI'].iloc[i] < 70) and (df['RSI'].iloc[i-1] >= 70)

        # --- Regular Signals ---
        is_buy_signal = rsi_crossed_above_30 and macd_crossed_above_signal and close_crossed_above_ema20
        is_sell_signal = rsi_crossed_below_70 and macd_crossed_below_signal and close_crossed_below_ema20

        # --- STRONG Signals ---
        # Conditions for STRONG signals (current state, not necessarily cross for all)
        strong_buy_conditions_met = (df['RSI'].iloc[i] > 35) and \
                                    (df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]) and \
                                    (df['Close'].iloc[i] > df['EMA20'].iloc[i]) and \
                                    (df['EMA20'].iloc[i] > df['EMA50'].iloc[i])

        strong_sell_conditions_met = (df['RSI'].iloc[i] < 65) and \
                                     (df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i]) and \
                                     (df['Close'].iloc[i] < df['EMA20'].iloc[i]) and \
                                     (df['EMA20'].iloc[i] < df['EMA50'].iloc[i])

        # --- SPEC Signals ---
        # MACD recently crossed (within last 3 periods)
        macd_recently_crossed_above = False
        for k in range(1, 4): # Check i-1, i-2, i-3
            if i-k < 0: break # Boundary check
            if (df['MACD'].iloc[i-k+1] > df['MACD_Signal'].iloc[i-k+1]) and \
               (df['MACD'].iloc[i-k] <= df['MACD_Signal'].iloc[i-k]):
                macd_recently_crossed_above = True
                break

        macd_recently_crossed_below = False
        for k in range(1, 4): # Check i-1, i-2, i-3
            if i-k < 0: break
            if (df['MACD'].iloc[i-k+1] < df['MACD_Signal'].iloc[i-k+1]) and \
               (df['MACD'].iloc[i-k] >= df['MACD_Signal'].iloc[i-k]):
                macd_recently_crossed_below = True
                break

        spec_buy_conditions_met = macd_recently_crossed_above and \
                                  (df['RSI'].iloc[i] > 20 and df['RSI'].iloc[i] < 30) and \
                                  not (df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]) # MACD not necessarily above signal *currently*

        spec_sell_conditions_met = macd_recently_crossed_below and \
                                   (df['RSI'].iloc[i] > 70 and df['RSI'].iloc[i] < 80) and \
                                   not (df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i]) # MACD not necessarily below signal *currently*

        # --- Assigning Signals (priority can be adjusted) ---
        # Strong signals take precedence
        if strong_buy_conditions_met:
            df.loc[df.index[i], 'signal'] = "STRONG_BUY"
        elif strong_sell_conditions_met:
            df.loc[df.index[i], 'signal'] = "STRONG_SELL"
        elif is_buy_signal:
            df.loc[df.index[i], 'signal'] = "BUY"
        elif is_sell_signal:
            df.loc[df.index[i], 'signal'] = "SELL"
        elif spec_buy_conditions_met:
             df.loc[df.index[i], 'signal'] = "SPEC_BUY"
        elif spec_sell_conditions_met:
             df.loc[df.index[i], 'signal'] = "SPEC_SELL"

    print("Finished generating signals.")
    return df
