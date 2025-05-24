# finance_analyzer/charting.py
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt # For adding custom legends or markers if needed

def plot_stock_chart(stock_data_df, ticker_symbol, sma_short_series=None, sma_long_series=None, crossover_signals_df=None):
    """
    Plots a candlestick chart for the given stock data, optionally overlaying SMAs and crossover signals.

    Args:
        stock_data_df (pandas.DataFrame): DataFrame containing stock data (OHLC, Volume).
                                          Index should be DatetimeIndex.
        ticker_symbol (str): The stock ticker symbol for chart titling.
        sma_short_series (pd.Series, optional): Series of short-term SMA values.
        sma_long_series (pd.Series, optional): Series of long-term SMA values.
        crossover_signals_df (pd.DataFrame, optional): DataFrame with crossover signals,
                                                       expected columns: 'Signal', 'Short_SMA', 'Long_SMA'
                                                       and DatetimeIndex.
    Returns:
        None: Displays the chart.
    """
    if stock_data_df is None or stock_data_df.empty:
        print(f"No data provided for {ticker_symbol} to plot.")
        return

    if not isinstance(stock_data_df.index, pd.DatetimeIndex):
        try:
            stock_data_df.index = pd.to_datetime(stock_data_df.index)
            print("Converted DataFrame index to DatetimeIndex for plotting.")
        except Exception as e:
            print(f"Error: Charting requires a DatetimeIndex. Failed to convert index: {e}")
            return
            
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in stock_data_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns for plotting: {', '.join(missing_cols)}")
        return

    try:
        print(f"Plotting chart for {ticker_symbol}...")
        
        # Prepare additional plot elements (addplots) for mplfinance
        apds = [] # List to hold addplot dictionaries

        # Add SMAs if provided
        if sma_short_series is not None and not sma_short_series.empty:
            apds.append(mpf.make_addplot(sma_short_series, panel=0, color='blue', width=0.7,
                                         title=f"SMA {sma_short_series.name or 'Short'}")) # Use series name if available
                                         
        if sma_long_series is not None and not sma_long_series.empty:
            apds.append(mpf.make_addplot(sma_long_series, panel=0, color='orange', width=0.7,
                                         title=f"SMA {sma_long_series.name or 'Long'}"))

        # Add Crossover signals if provided
        # We need to plot markers for these. mplfinance addplot can take scatter markers.
        if crossover_signals_df is not None and not crossover_signals_df.empty:
            # Ensure crossover_signals_df index is DatetimeIndex and aligns with stock_data_df
            crossover_signals_df.index = pd.to_datetime(crossover_signals_df.index)
            
            # Align signals with the main stock data index for plotting markers correctly on OHLC data.
            # We'll plot markers on the 'Close' price or slightly above/below the Long SMA value at crossover points.
            
            bullish_markers = []
            bearish_markers = []

            for date, row in crossover_signals_df.iterrows():
                if date in stock_data_df.index: # Ensure the date exists in the main data
                    plot_value = stock_data_df.loc[date, 'Low'] * 0.98 # Default below low for bullish
                    if row['Signal'] == "Bullish Crossover":
                        # Plot slightly below the low of the crossover day
                        bullish_markers.append(plot_value) 
                        bearish_markers.append(float('nan')) # NaN for other marker type
                    elif row['Signal'] == "Bearish Crossover":
                        # Plot slightly above the high of the crossover day
                        plot_value = stock_data_df.loc[date, 'High'] * 1.02
                        bearish_markers.append(plot_value)
                        bullish_markers.append(float('nan')) # NaN for other marker type
                    else: # Should not happen if df is clean
                        bullish_markers.append(float('nan'))
                        bearish_markers.append(float('nan'))
                else: # Date not in stock data, can't plot marker accurately
                    bullish_markers.append(float('nan'))
                    bearish_markers.append(float('nan'))
            
            # Create Series for scatter plot, aligned with stock_data_df.index
            # Fill non-signal days with NaN so markers only appear on signal days.
            common_index = stock_data_df.index
            
            bullish_plot_series = pd.Series(index=common_index, dtype=float)
            bearish_plot_series = pd.Series(index=common_index, dtype=float)

            for date, row in crossover_signals_df.iterrows():
                if date in common_index:
                    if row['Signal'] == "Bullish Crossover":
                        bullish_plot_series[date] = stock_data_df.loc[date, 'Low'] * 0.97
                    elif row['Signal'] == "Bearish Crossover":
                        bearish_plot_series[date] = stock_data_df.loc[date, 'High'] * 1.03
            
            if not bullish_plot_series.dropna().empty:
                 apds.append(mpf.make_addplot(bullish_plot_series, type='scatter', marker='^', markersize=100, color='green', panel=0, title="Bullish Cross"))
            if not bearish_plot_series.dropna().empty:
                 apds.append(mpf.make_addplot(bearish_plot_series, type='scatter', marker='v', markersize=100, color='red', panel=0, title="Bearish Cross"))


        # Create the plot
        fig, axes = mpf.plot(stock_data_df, 
                             type='candle', 
                             style='yahoo',
                             title=f'{ticker_symbol} Stock Price History',
                             ylabel='Price ($)',
                             volume=True,
                             ylabel_lower='Volume',
                             addplot=apds if apds else None, # Pass addplots if any
                             figsize=(15, 9), # Adjusted for potentially more legends
                             returnfig=True, # Return fig and axes for further modification if needed
                             warn_too_much_data=10000 
                            )
        
        # Add a legend for custom titles in addplot if they are not automatically generated well
        # mplfinance's handling of addplot legends can be tricky.
        # If titles in make_addplot don't show up as desired, a manual legend might be needed.
        # For now, let's rely on make_addplot titles. If issues, we can use fig.legend()
        # Example:
        # if apds:
        #    axes[0].legend([plot['title'] for plot in apds if plot.get('title')])


        mpf.show() # Ensure chart is displayed if not in interactive environment
        print(f"Chart for {ticker_symbol} displayed.")

    except Exception as e:
        print(f"Error plotting chart for {ticker_symbol}: {e}")
        # import traceback
        # traceback.print_exc() # For more detailed error during development

if __name__ == '__main__':
    try:
        from stock_data import download_stock_data
        from trend_analysis import calculate_sma, identify_sma_crossovers
    except ImportError:
        print("Could not import from stock_data or trend_analysis. Standalone test limited.")
        # Create dummy data for charting_py standalone test
        dummy_dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        ohlcv = {
            'Open': [100+i+((i%5)*2) for i in range(30)],
            'High': [100+i+5+((i%3)*3) for i in range(30)],
            'Low': [100+i-5-((i%4)*2) for i in range(30)],
            'Close': [100+i for i in range(30)],
            'Volume': [10000+i*100 for i in range(30)]
        }
        sample_df = pd.DataFrame(ohlcv, index=dummy_dates)
        sample_df.index.name = 'Date'
        ticker_to_plot = "DUMMY_ADV"
        
        sma5 = calculate_sma(sample_df, 5)
        sma5.name = "SMA5"
        sma10 = calculate_sma(sample_df, 10)
        sma10.name = "SMA10"
        crossovers = identify_sma_crossovers(sma5, sma10)
        
        print("--- Using DUMMY data for ADVANCED charting example ---")
        plot_stock_chart(sample_df, ticker_to_plot, sma_short_series=sma5, sma_long_series=sma10, crossover_signals_df=crossovers)

    else:
        from datetime import datetime, timedelta
        ticker_to_plot = "NVDA" # NVIDIA Corp.
        end_chart_date = datetime.today().strftime('%Y-%m-%d')
        start_chart_date = (datetime.today() - timedelta(days=180)).strftime('%Y-%m-%d') # Approx 6 months daily
        
        print(f"--- Downloading {ticker_to_plot} data for advanced charting example ---")
        stock_df_for_chart = download_stock_data(ticker_to_plot, start_chart_date, end_chart_date, interval="1d")
        
        if stock_df_for_chart is not None and not stock_df_for_chart.empty:
            short_window = 20
            long_window = 50
            
            print(f"Calculating SMAs ({short_window}, {long_window}) for {ticker_to_plot}...")
            sma_short = calculate_sma(stock_df_for_chart, short_window)
            sma_short.name = f"SMA{short_window}" # Name the series for potential legend use
            
            sma_long = calculate_sma(stock_df_for_chart, long_window)
            sma_long.name = f"SMA{long_window}"
            
            print(f"Identifying SMA Crossovers for {ticker_to_plot}...")
            crossover_signals = identify_sma_crossovers(sma_short, sma_long)
            if not crossover_signals.empty:
                print("Crossover signals found:")
                print(crossover_signals)
            else:
                print("No crossover signals found.")
            
            plot_stock_chart(stock_df_for_chart, ticker_to_plot, 
                             sma_short_series=sma_short, 
                             sma_long_series=sma_long, 
                             crossover_signals_df=crossover_signals)
        else:
            print(f"Could not download data for {ticker_to_plot}, so cannot plot advanced chart.")

    print("\nAdvanced charting module test finished.")
    print("Note: Charts are displayed in separate windows. Close them to continue.")
