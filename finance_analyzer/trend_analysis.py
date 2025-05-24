# finance_analyzer/trend_analysis.py
import pandas as pd

# --- (Keep existing calculate_sma, identify_sma_crossovers, identify_consecutive_trends functions here) ---
def calculate_sma(data_df, window):
    """
    Calculates the Simple Moving Average (SMA) for a given window.

    Args:
        data_df (pd.DataFrame): DataFrame with stock data, must contain a 'Close' column.
        window (int): The window period for the SMA (e.g., 20 for 20-day SMA).

    Returns:
        pd.Series: A Series containing the SMA values, indexed by date.
                   Returns an empty Series if 'Close' column is missing or window is invalid.
    """
    if 'Close' not in data_df.columns:
        print(f"Error: 'Close' column not found in DataFrame for SMA calculation.")
        return pd.Series(dtype=float) 
    if not isinstance(window, int) or window <= 0:
        print(f"Error: SMA window must be a positive integer. Received: {window}")
        return pd.Series(dtype=float)
    if window > len(data_df):
        # print(f"Warning: SMA window ({window}) is larger than the data length ({len(data_df)}). SMA will be all NaN.")
        # Pandas handles this by returning NaNs for periods < window, which is fine.
        pass
        
    try:
        sma_series = data_df['Close'].rolling(window=window, min_periods=1).mean()
        return sma_series
    except Exception as e:
        print(f"Error calculating SMA for window {window}: {e}")
        return pd.Series(dtype=float)

def identify_sma_crossovers(sma_short_series, sma_long_series):
    """
    Identifies Simple Moving Average (SMA) crossover points.
    - Bullish Crossover (Golden Cross): Short-term SMA crosses above Long-term SMA.
    - Bearish Crossover (Death Cross): Short-term SMA crosses below Long-term SMA.

    Args:
        sma_short_series (pd.Series): Series of shorter-term SMA values.
        sma_long_series (pd.Series): Series of longer-term SMA values.

    Returns:
        pd.DataFrame: A DataFrame with columns 'Date', 'Signal' ('Bullish Crossover' or 
                      'Bearish Crossover'), 'Short_SMA', 'Long_SMA'.
                      Index of DataFrame is original date index from SMA series.
                      Returns an empty DataFrame if inputs are invalid or no crossovers.
    """
    if sma_short_series.empty or sma_long_series.empty:
        # print("Error: SMA series for crossover detection cannot be empty.") # Already handled if SMAs are empty
        return pd.DataFrame(columns=['Signal', 'Short_SMA', 'Long_SMA'])

    combined_df = pd.DataFrame({
        'Short_SMA': sma_short_series,
        'Long_SMA': sma_long_series
    }).dropna() 

    if combined_df.empty or len(combined_df) < 2: 
        # print("Not enough data points after aligning SMAs to detect crossovers.")
        return pd.DataFrame(columns=['Signal', 'Short_SMA', 'Long_SMA'])

    combined_df['Position'] = 0
    combined_df.loc[combined_df['Short_SMA'] > combined_df['Long_SMA'], 'Position'] = 1
    combined_df.loc[combined_df['Short_SMA'] < combined_df['Long_SMA'], 'Position'] = -1
    combined_df['Crossover'] = combined_df['Position'].diff()

    crossover_points = []
    for date, row in combined_df.iterrows():
        signal = None
        if row['Crossover'] == 2: 
            signal = "Bullish Crossover"
        elif row['Crossover'] == -2: 
            signal = "Bearish Crossover"
        
        if signal:
            crossover_points.append({
                'Date': date, 
                'Signal': signal,
                'Short_SMA': row['Short_SMA'],
                'Long_SMA': row['Long_SMA']
            })
            
    if not crossover_points:
        return pd.DataFrame(columns=['Signal', 'Short_SMA', 'Long_SMA'])
        
    return pd.DataFrame(crossover_points).set_index('Date')


def identify_consecutive_trends(data_df, column='Close', num_periods=3):
    """
    Identifies X consecutive periods of price increase (bullish) or decrease (bearish).

    Args:
        data_df (pd.DataFrame): DataFrame with stock data.
        column (str): The column to check for trends (e.g., 'Close', 'Open').
        num_periods (int): The number of consecutive periods to define a trend.

    Returns:
        pd.DataFrame: DataFrame with columns 'Date', 'Trend_Signal' ('X-Period Bullish' or 
                      'X-Period Bearish'), and the value of the 'column'.
                      Returns an empty DataFrame if no such trends are found or input is invalid.
    """
    if column not in data_df.columns:
        print(f"Error: Column '{column}' not found for trend identification.")
        return pd.DataFrame(columns=['Trend_Signal', column])
    if not isinstance(num_periods, int) or num_periods <= 1:
        print("Error: Number of periods for consecutive trend must be an integer greater than 1.")
        return pd.DataFrame(columns=['Trend_Signal', column])
    if len(data_df) < num_periods:
        # print("Not enough data to identify consecutive trends for the given number of periods.")
        return pd.DataFrame(columns=['Trend_Signal', column])

    # Create a copy to avoid SettingWithCopyWarning if data_df is a slice
    df_copy = data_df.copy()
    df_copy['Diff'] = df_copy[column].diff()
    df_copy['Direction'] = 0
    df_copy.loc[df_copy['Diff'] > 0, 'Direction'] = 1
    df_copy.loc[df_copy['Diff'] < 0, 'Direction'] = -1

    trends = []
    # Rolling window to check sum of directions
    # A sum of num_periods indicates all 1s (bullish)
    # A sum of -num_periods indicates all -1s (bearish)
    direction_sum = df_copy['Direction'].rolling(window=num_periods).sum()

    for i in range(num_periods - 1, len(df_copy)):
        current_sum = direction_sum.iloc[i]
        if current_sum == num_periods:
            trends.append({
                'Date': df_copy.index[i],
                'Trend_Signal': f'{num_periods}-Period Bullish',
                column: df_copy[column].iloc[i]
            })
        elif current_sum == -num_periods:
            trends.append({
                'Date': df_copy.index[i],
                'Trend_Signal': f'{num_periods}-Period Bearish',
                column: df_copy[column].iloc[i]
            })
            
    if not trends:
        return pd.DataFrame(columns=['Trend_Signal', column])
        
    return pd.DataFrame(trends).set_index('Date')

# New function for buy/sell indicators
def generate_buy_sell_indicators(data_df, sma_short_series, sma_long_series, crossover_signals_df):
    """
    Generates simple rule-based "Buy" or "Sell" indicators.
    This is illustrative and NOT financial advice.

    Rules for this example:
    - "Potential Buy Signal":
        1. A "Bullish Crossover" just occurred (e.g., within the last 1-3 periods).
        2. AND current Close price is above the Long SMA.
    - "Potential Sell Signal":
        1. A "Bearish Crossover" just occurred (e.g., within the last 1-3 periods).
        2. AND current Close price is below the Long SMA.
    - "Hold/Neutral": Otherwise.

    Args:
        data_df (pd.DataFrame): Original stock data with 'Close' prices.
        sma_short_series (pd.Series): Short-term SMA.
        sma_long_series (pd.Series): Long-term SMA.
        crossover_signals_df (pd.DataFrame): DataFrame from identify_sma_crossovers.

    Returns:
        pd.DataFrame: DataFrame with 'Date' index and 'Indicator' column 
                      ('Potential Buy', 'Potential Sell', 'Hold/Neutral').
    """
    if data_df.empty or sma_long_series.empty: # sma_short is implicitly covered by crossover_signals
        print("Error: Data, short SMA, or long SMA series is empty for generating indicators.")
        return pd.DataFrame(columns=['Indicator'])

    # Align all data to the main DataFrame's index
    indicators_df = pd.DataFrame(index=data_df.index)
    indicators_df['Close'] = data_df['Close']
    indicators_df['Long_SMA'] = sma_long_series
    indicators_df['Indicator'] = "Hold/Neutral" # Default state

    # Define a short window to check for recent crossovers (e.g., last 3 periods)
    crossover_lookback_window = 3 

    for i in range(len(indicators_df)):
        current_date = indicators_df.index[i]
        current_close = indicators_df.loc[current_date, 'Close']
        current_long_sma = indicators_df.loc[current_date, 'Long_SMA']

        if pd.isna(current_long_sma): # Skip if long SMA is NaN (e.g. at the beginning)
            continue

        # Check for recent crossovers
        # Iterate backwards from current_date for crossover_lookback_window days
        recent_signal = None
        for lookback_days in range(crossover_lookback_window):
            check_date = current_date - pd.Timedelta(days=lookback_days)
            if check_date in crossover_signals_df.index:
                recent_signal = crossover_signals_df.loc[check_date, 'Signal']
                break # Found the most recent signal within the window

        if recent_signal:
            if recent_signal == "Bullish Crossover" and current_close > current_long_sma:
                indicators_df.loc[current_date, 'Indicator'] = "Potential Buy"
            elif recent_signal == "Bearish Crossover" and current_close < current_long_sma:
                indicators_df.loc[current_date, 'Indicator'] = "Potential Sell"
    
    return indicators_df[['Indicator']] # Return only the 'Indicator' column, indexed by Date


if __name__ == '__main__':
    # --- (Keep existing __main__ test code for SMA, Crossovers, Consecutive Trends) ---
    dates = pd.to_datetime([
        '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
        '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
        '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15'
    ])
    close_prices = [
        100, 102, 101, 103, 105, 104, 102, 100, 103, 106, 
        108, 105, 103, 100, 98 
    ]
    sample_df = pd.DataFrame({'Close': close_prices, 'Open': close_prices, 'High': close_prices, 'Low': close_prices, 'Volume': close_prices}, index=dates) # Added other cols for full df
    sample_df.index.name = 'Date'

    print("--- Testing SMA Calculation ---")
    sma_5 = calculate_sma(sample_df.copy(), 5)
    if not sma_5.empty: print(f"5-period SMA (first 5):\n{sma_5.head()}")
    
    print("\n--- Testing SMA Crossovers ---")
    sma_short_for_test = calculate_sma(sample_df.copy(), 3)
    sma_long_for_test = calculate_sma(sample_df.copy(), 6)
    crossovers_df_for_test = pd.DataFrame()
    if not sma_short_for_test.empty and not sma_long_for_test.empty:
        crossovers_df_for_test = identify_sma_crossovers(sma_short_for_test, sma_long_for_test)
        if not crossovers_df_for_test.empty: print(f"SMA Crossovers Found:\n{crossovers_df_for_test}")
        else: print("No SMA crossovers found in sample data for main test.")
    else: print("Could not calculate SMAs for crossover test in main.")

    print("\n--- Testing Consecutive Trends ---")
    consecutive_trends_df_for_test = identify_consecutive_trends(sample_df.copy(), column='Close', num_periods=3)
    if not consecutive_trends_df_for_test.empty: print(f"3-Period Consecutive Trends:\n{consecutive_trends_df_for_test.head()}")
    else: print("No 3-period consecutive trends found in sample data for main test.")

    # --- New test for Buy/Sell Indicators ---
    print("\n--- Testing Buy/Sell Indicators ---")
    if not sample_df.empty and not sma_long_for_test.empty and not crossovers_df_for_test.empty :
        # We need a crossover_signals_df that makes sense with sample_df
        # Let's create a sample bullish crossover for this test at '2023-01-09'
        # At '2023-01-09', Close=103, LongSMA (SMA6 for sample) is approx 102.5. So Close > LongSMA
        
        # Manually create a crossover for testing indicator logic
        manual_crossover_dates = pd.to_datetime(['2023-01-09', '2023-01-14'])
        manual_crossover_signals = ["Bullish Crossover", "Bearish Crossover"]
        manual_crossovers_df = pd.DataFrame({
            'Signal': manual_crossover_signals
        }, index=manual_crossover_dates)
        
        indicator_df = generate_buy_sell_indicators(sample_df, sma_short_for_test, sma_long_for_test, manual_crossovers_df)
        if not indicator_df.empty:
            print("Buy/Sell Indicators (showing rows around signals):")
            # Print a few rows around the crossover dates to see the effect
            print(indicator_df[indicator_df.index >= '2023-01-08'])
        else:
            print("Could not generate buy/sell indicators.")
    else:
        print("Skipping Buy/Sell indicator test due to missing base data (sample_df, SMAs, or crossovers).")
    
    print("\nTrend analysis module test finished (including indicators).")
