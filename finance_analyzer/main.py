# finance_analyzer/main.py
from stock_data import download_stock_data
from charting import plot_stock_chart
from trend_analysis import calculate_sma, identify_sma_crossovers, generate_buy_sell_indicators #, identify_consecutive_trends
from datetime import datetime, timedelta
import pandas as pd # Ensure pandas is imported for date checks if needed

def main():
    print("Welcome to the Finance Chart Analyzer!")
    print("="*30)
    
    ticker_symbol = input("Enter the stock ticker symbol (e.g., AAPL, MSFT, GOOGL): ").upper()
    if not ticker_symbol:
        print("No ticker symbol entered. Exiting.")
        return

    end_date_str = datetime.today().strftime('%Y-%m-%d')
    start_date_str = (datetime.today() - timedelta(days=365 * 1)).strftime('%Y-%m-%d')
    interval_type = "1d" 
    
    short_sma_window = 20
    long_sma_window = 50

    print(f"\nFetching data for {ticker_symbol} from {start_date_str} to {end_date_str} ({interval_type} interval).")
    stock_df = download_stock_data(ticker_symbol, start_date_str, end_date_str, interval=interval_type)

    if stock_df is not None and not stock_df.empty:
        print(f"Data for {ticker_symbol} downloaded successfully.")

        print(f"\nCalculating SMAs (Short: {short_sma_window}, Long: {long_sma_window})...")
        sma_short = calculate_sma(stock_df, short_sma_window)
        sma_short.name = f"SMA{short_sma_window}"
        
        sma_long = calculate_sma(stock_df, long_sma_window)
        sma_long.name = f"SMA{long_sma_window}"

        print("Identifying SMA Crossovers...")
        crossover_signals = identify_sma_crossovers(sma_short, sma_long)

        if not crossover_signals.empty:
            print("\n--- SMA Crossover Signals (Recent) ---")
            print(crossover_signals.tail())
        else:
            print("No significant SMA crossover signals found in the selected period.")
        
        # Generate Buy/Sell Indicators
        print("\nGenerating Buy/Sell/Hold Indicators based on SMA Crossovers...")
        indicator_df = generate_buy_sell_indicators(stock_df, sma_short, sma_long, crossover_signals)

        if not indicator_df.empty and 'Indicator' in indicator_df.columns:
            # Display the latest indicator
            latest_indicator_date = indicator_df.index[-1]
            latest_signal = indicator_df['Indicator'].iloc[-1]
            print("\n--- Latest Trading Indicator ---")
            print(f"As of {pd.to_datetime(latest_indicator_date).strftime('%Y-%m-%d')}: {latest_signal}")
            # Display a few recent indicator changes if any
            recent_indicators_with_change = indicator_df[indicator_df['Indicator'] != indicator_df['Indicator'].shift(1)]
            if not recent_indicators_with_change.empty :
                 print("\n--- Recent Indicator Changes ---")
                 print(recent_indicators_with_change.tail()) # Show last few changes
            elif not indicator_df.empty: # If no changes, but indicators exist, show last few days
                 print("\n--- Recent Indicator Status (last 5 days) ---")
                 print(indicator_df.tail())


        else:
            print("Could not generate buy/sell indicators.")

        print("\nDisplaying chart with SMAs and Crossover signals...")
        # Note: plot_stock_chart currently does not show the buy/sell text indicators on the chart itself.
        # That would be a further enhancement to charting.py.
        plot_stock_chart(stock_df, ticker_symbol, 
                         sma_short_series=sma_short, 
                         sma_long_series=sma_long, 
                         crossover_signals_df=crossover_signals)
        
        print("\nNote: Close the chart window to exit the program.")
        print("The textual Buy/Sell indicators are based on the rules defined in trend_analysis.py.")
        print("THIS IS NOT FINANCIAL ADVICE.")
    else:
        print(f"Could not retrieve or process data for {ticker_symbol}. Analysis cannot proceed.")

    print("="*30)
    print("Finance Chart Analyzer finished.")

if __name__ == '__main__':
    try:
        import yfinance
        import pandas
        import matplotlib
        import mplfinance
    except ImportError as e:
        print(f"CRITICAL ERROR: A required library is not installed: {e}")
        print("Please install all libraries from requirements.txt using: pip install -r requirements.txt")
    else:
        main()
