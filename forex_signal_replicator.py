import pandas as pd
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_forex_signals(df, short_window=50, long_window=200):
    """
    Generates trading signals based on moving average crossovers.

    Args:
        df (pd.DataFrame): DataFrame with 'Close' prices.
        short_window (int): Window for the short-term moving average.
        long_window (int): Window for the long-term moving average.

    Returns:
        pd.DataFrame: DataFrame with 'signal' and 'position' columns.
    """
    # Calculate Exponential Moving Averages (EMA)
    df['short_mavg'] = talib.EMA(df['Close'].values, timeperiod=short_window)
    df['long_mavg'] = talib.EMA(df['Close'].values, timeperiod=long_window)

    # Initialize signal columns
    df['buy_signal'] = False
    df['sell_signal'] = False
    df['strong_buy_signal'] = False
    df['strong_sell_signal'] = False
    df['spec_buy_signal'] = False
    df['spec_sell_signal'] = False

    # Determine the start index for iteration
    # RSI default period is 14. EMA output has `window-1` NaNs.
    # We need valid data for df.index[i-1] as well.
    # So, start index should be at least 1, and also ensure all indicators are valid.
    # talib.RSI produces `period` NaNs. talib.EMA produces `timeperiod-1` NaNs.
    # So, data for RSI is valid from `rsi_period`. Data for long_mavg from `long_window-1`.
    # Iteration needs `i` and `i-1`.
    rsi_period = 14 # Assuming default if not passed, or get from df['rsi'] creation if possible
    start_index = max(1, long_window, rsi_period) # Ensure all data series have started

    for i in range(start_index, len(df)):
        # Ensure data at i and i-1 is valid for all required series
        if df['short_mavg'].iloc[i-1:i+1].isna().any() or \
           df['long_mavg'].iloc[i-1:i+1].isna().any() or \
           df['rsi'].iloc[i-1:i+1].isna().any() or \
           df['Close'].iloc[i-1:i+1].isna().any():
            continue

        current_rsi = df['rsi'].iloc[i]

        is_bullish_crossover = df['short_mavg'].iloc[i] > df['long_mavg'].iloc[i] and \
                               df['short_mavg'].iloc[i-1] <= df['long_mavg'].iloc[i-1]

        is_bearish_crossover = df['short_mavg'].iloc[i] < df['long_mavg'].iloc[i] and \
                                df['short_mavg'].iloc[i-1] >= df['long_mavg'].iloc[i-1]

        if is_bullish_crossover:
            if current_rsi < 70:  # Potential buy
                price_action_strong_buy = df['Close'].iloc[i] > df['short_mavg'].iloc[i] * 1.005
                if current_rsi < 50 or price_action_strong_buy:
                    df.loc[df.index[i], 'strong_buy_signal'] = True
                elif current_rsi < 40: # SPEC buy condition (RSI < 40 implies RSI < 70 is met)
                    df.loc[df.index[i], 'spec_buy_signal'] = True
                else: # Normal buy (RSI between 40 and 70, no strong PA)
                    df.loc[df.index[i], 'buy_signal'] = True

        elif is_bearish_crossover:
            if current_rsi > 30:  # Potential sell
                price_action_strong_sell = df['Close'].iloc[i] < df['short_mavg'].iloc[i] * 0.995
                if current_rsi > 50 or price_action_strong_sell:
                    df.loc[df.index[i], 'strong_sell_signal'] = True
                elif current_rsi > 60: # SPEC sell condition (RSI > 60 implies RSI > 30 is met)
                    df.loc[df.index[i], 'spec_sell_signal'] = True
                else: # Normal sell (RSI between 30 and 60, no strong PA)
                    df.loc[df.index[i], 'sell_signal'] = True

    return df


def plot_forex_signals(df, symbol='EUR/USD'):
    """
    Plots the forex signals with buy/sell markers.

    Args:
        df (pd.DataFrame): DataFrame with OHLC, EMAs, RSI, and signal columns.
        symbol (str): The trading symbol to display in the plot title.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # --- Row 1: Price, EMAs, Signals ---
    # Plot candlestick chart for price
    fig.add_trace(go.Candlestick(x=df.index,
                               open=df['Open'],
                               high=df['High'],
                               low=df['Low'],
                               close=df['Close'],
                               name='Price'), row=1, col=1)

    # Plot moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['short_mavg'], name='Short MA',
                             line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['long_mavg'], name='Long MA',
                             line=dict(color='orange')), row=1, col=1)

    # Plot Buy Signals
    fig.add_trace(go.Scatter(
        x=df[df['buy_signal']].index,
        y=df['Low'][df['buy_signal']] * 0.99,
        mode='markers', name='Buy',
        marker=dict(symbol='triangle-up', size=8, color='green')), row=1, col=1)

    # Plot Sell Signals
    fig.add_trace(go.Scatter(
        x=df[df['sell_signal']].index,
        y=df['High'][df['sell_signal']] * 1.01,
        mode='markers', name='Sell',
        marker=dict(symbol='triangle-down', size=8, color='red')), row=1, col=1)

    # Plot STRONG Buy Signals
    fig.add_trace(go.Scatter(
        x=df[df['strong_buy_signal']].index,
        y=df['Low'][df['strong_buy_signal']] * 0.98, # Slightly lower than normal buy
        mode='markers', name='STRONG Buy',
        marker=dict(symbol='circle', size=10, color='darkgreen')), row=1, col=1)

    # Plot STRONG Sell Signals
    fig.add_trace(go.Scatter(
        x=df[df['strong_sell_signal']].index,
        y=df['High'][df['strong_sell_signal']] * 1.02, # Slightly higher than normal sell
        mode='markers', name='STRONG Sell',
        marker=dict(symbol='circle', size=10, color='darkred')), row=1, col=1)

    # Plot SPEC Buy Signals
    fig.add_trace(go.Scatter(
        x=df[df['spec_buy_signal']].index,
        y=df['Low'][df['spec_buy_signal']] * 0.99, # Same level as normal buy, different marker
        mode='markers', name='SPEC Buy',
        marker=dict(symbol='diamond-tall', size=8, color='blue')), row=1, col=1)

    # Plot SPEC Sell Signals
    fig.add_trace(go.Scatter(
        x=df[df['spec_sell_signal']].index,
        y=df['High'][df['spec_sell_signal']] * 1.01, # Same level as normal sell, different marker
        mode='markers', name='SPEC Sell',
        marker=dict(symbol='diamond-tall', size=8, color='purple')), row=1, col=1)

    # --- Row 2: RSI ---
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI',
                             line=dict(color='black')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="blue", row=2, col=1)

    # --- Layout Updates ---
    fig.update_layout(
        title=f'{symbol} Trading Signals and RSI',
        legend_title='Legend',
        height=800 # Increased height for better visibility of subplots
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False) # Hide range slider for price chart

    fig.show()


def calculate_rsi(df, period=14):
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        df (pd.DataFrame): DataFrame with 'Close' prices.
        period (int): The RSI calculation period.

    Returns:
        pd.DataFrame: DataFrame with 'rsi' column.
    """
    df['rsi'] = talib.RSI(df['Close'].values, timeperiod=period)
    return df


if __name__ == '__main__':
    # Create a more illustrative sample DataFrame (approx. 60 data points)
    # This data is designed to showcase various signal types.

    # Phase 1: Uptrend (Days 1-20) - Aim for buy signals, RSI climbing
    # Phase 2: Consolidation/Topping (Days 21-35) - Price moves sideways, RSI may fluctuate
    # Phase 3: Downtrend (Days 36-60) - Aim for sell signals, RSI falling

    dates = pd.to_datetime(['2023-01-%02d' % i for i in range(1, 32)] + \
                           ['2023-02-%02d' % i for i in range(1, 29)]) # 31+28 = 59 days

    data = {
        'Date': dates,
        'Open':  [ # Phase 1: Uptrend
                  1.050, 1.052, 1.055, 1.053, 1.058, 1.060, 1.062, 1.065, 1.063, 1.068, #10
                  1.070, 1.072, 1.075, 1.073, 1.078, 1.080, 1.082, 1.085, 1.083, 1.088, #20
                  # Phase 2: Consolidation/Topping
                  1.085, 1.083, 1.080, 1.082, 1.079, 1.081, 1.078, 1.080, 1.077, 1.075, #30
                  1.073, 1.070, 1.072, 1.069, 1.067,                                   #35
                  # Phase 3: Downtrend
                  1.065, 1.063, 1.060, 1.058, 1.055, 1.053, 1.050, 1.048, 1.045, 1.043, #45
                  1.040, 1.038, 1.035, 1.033, 1.030, 1.028, 1.025, 1.023, 1.020, 1.018, #55
                  1.015, 1.013, 1.010, 1.008                                            #59
                  ],
        'High':  [ # Phase 1
                  1.058, 1.060, 1.063, 1.061, 1.066, 1.068, 1.070, 1.073, 1.071, 1.076, #10
                  1.078, 1.080, 1.083, 1.081, 1.086, 1.088, 1.090, 1.093, 1.091, 1.096, #20
                  # Phase 2
                  1.092, 1.089, 1.085, 1.087, 1.084, 1.086, 1.083, 1.085, 1.082, 1.080, #30
                  1.078, 1.075, 1.077, 1.074, 1.072,                                   #35
                  # Phase 3
                  1.070, 1.068, 1.065, 1.063, 1.060, 1.058, 1.055, 1.053, 1.050, 1.048, #45
                  1.045, 1.043, 1.040, 1.038, 1.035, 1.033, 1.030, 1.028, 1.025, 1.023, #55
                  1.020, 1.018, 1.015, 1.013                                            #59
                  ],
        'Low':   [ # Phase 1
                  1.048, 1.050, 1.052, 1.050, 1.055, 1.057, 1.059, 1.062, 1.060, 1.065, #10
                  1.067, 1.069, 1.072, 1.070, 1.075, 1.077, 1.079, 1.082, 1.080, 1.085, #20
                  # Phase 2
                  1.082, 1.080, 1.077, 1.079, 1.076, 1.078, 1.075, 1.077, 1.074, 1.072, #30
                  1.070, 1.067, 1.069, 1.066, 1.064,                                   #35
                  # Phase 3
                  1.062, 1.060, 1.057, 1.055, 1.052, 1.050, 1.047, 1.045, 1.042, 1.040, #45
                  1.037, 1.035, 1.032, 1.030, 1.027, 1.025, 1.022, 1.020, 1.017, 1.015, #55
                  1.012, 1.010, 1.007, 1.005                                            #59
                  ],
        'Close': [ # Phase 1
                  1.052, 1.056, 1.058, 1.059, 1.063, 1.065, 1.068, 1.070, 1.072, 1.075, #10
                  1.076, 1.079, 1.080, 1.080, 1.085, 1.086, 1.089, 1.090, 1.090, 1.092, #20
                  # Phase 2
                  1.088, 1.082, 1.083, 1.080, 1.080, 1.079, 1.079, 1.076, 1.076, 1.072, #30
                  1.071, 1.068, 1.070, 1.068, 1.065,                                   #35
                  # Phase 3
                  1.062, 1.060, 1.057, 1.054, 1.052, 1.050, 1.047, 1.044, 1.042, 1.040, #45
                  1.037, 1.034, 1.032, 1.030, 1.027, 1.024, 1.022, 1.020, 1.018, 1.016, #55
                  1.013, 1.010, 1.008, 1.006                                            #59
                  ]
    }
    sample_df = pd.DataFrame(data)
    sample_df.set_index('Date', inplace=True)

    # --- Indicator Calculation ---
    # Calculate RSI (default period 14)
    sample_df = calculate_rsi(sample_df)

    # Generate trading signals using EMAs (e.g., 10-period short, 20-period long)
    # These window parameters are chosen to likely generate crossovers in the sample data.
    signals_df = generate_forex_signals(sample_df, short_window=10, long_window=20)

    # --- Plotting ---
    # Plot the signals. The plot_forex_signals function is designed to handle the new signal columns.
    plot_forex_signals(signals_df, symbol='Sample EUR/USD')
