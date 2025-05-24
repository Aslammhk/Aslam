# finance_analyzer/stock_data.py
import yfinance as yf
import pandas as pd

def download_stock_data(ticker_symbol, start_date, end_date, interval="1mo"):
    """
    Downloads historical stock data for a given ticker symbol and period.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., "AAPL").
        start_date (str): The start date for the data (e.g., "2020-01-01").
        end_date (str): The end date for the data (e.g., "2023-01-01").
        interval (str): Data interval. Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 
                        1h, 1d, 5d, 1wk, 1mo, 3mo. Default is "1mo" for monthly data.

    Returns:
        pandas.DataFrame: A DataFrame containing the stock data (OHLC, Volume, etc.),
                          or None if an error occurs.
    """
    print(f"Attempting to download data for {ticker_symbol} from {start_date} to {end_date} with {interval} interval...")
    try:
        stock = yf.Ticker(ticker_symbol)
        # Note: yfinance interval parameter for hist is different from download.
        # For hist, common intervals are '1d', '1wk', '1mo'.
        # 'period' can be used instead of start/end, e.g., period="1y"
        data = stock.history(start=start_date, end=end_date, interval=interval)

        if data.empty:
            print(f"No data found for {ticker_symbol}. It might be an invalid ticker or delisted.")
            return None

        # Regarding "number of peoples trading":
        # yfinance provides 'Volume' data, which is the number of shares traded.
        # This is a common proxy for trading activity/interest but not a direct count of individual traders.
        # Explicit data on the "number of unique traders" is typically not available in standard free APIs.
        if 'Volume' not in data.columns:
            print(f"Volume data not available for {ticker_symbol}.")
        else:
            print("Volume data (proxy for 'number of people trading') included.")
            
        data.index.name = 'Date' # Ensure the index has a name for clarity
        print(f"Data downloaded successfully for {ticker_symbol}.")
        return data

    except Exception as e:
        print(f"Error downloading stock data for {ticker_symbol}: {e}")
        return None

if __name__ == '__main__':
    # Example Usage:
    ticker = "AAPL" # Apple Inc.
    # Let's try to get data for the last 3 years, monthly
    from datetime import datetime, timedelta
    end = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=3*365)).strftime('%Y-%m-%d')

    print(f"--- Testing {ticker} ---")
    stock_df = download_stock_data(ticker, start, end, interval="1mo")
    if stock_df is not None:
        print(f"\n{ticker} Data (first 5 rows):")
        print(stock_df.head())
        print(f"\n{ticker} Data (last 5 rows):")
        print(stock_df.tail())
        print(f"\nData columns: {stock_df.columns.tolist()}")
        if 'Volume' in stock_df.columns:
            print(f"Average monthly volume for {ticker} over the period: {stock_df['Volume'].mean():.0f}")

    print("\n--- Testing Invalid Ticker ---")
    invalid_ticker = "INVALIDTICKERXYZ"
    invalid_stock_df = download_stock_data(invalid_ticker, "2022-01-01", "2023-01-01")
    if invalid_stock_df is None:
        print("Correctly handled invalid ticker.")

    print("\n--- Testing Ticker with potentially no recent monthly data (example: a delisted stock, if known) ---")
    # This is hard to make a consistent example for, as yfinance might still find some data
    # or the ticker might become valid later. Let's use a very old or obscure one if possible.
    # For now, we'll rely on the "No data found" message if a ticker is valid but has no data for the period.
    # Example: A stock that was delisted long ago.
    # Note: This test might still download data if the ticker is reused or has some historical trace.
    # For now, we'll just show an example of a less common ticker.
    # test_delisted_ticker = "Sears" # SHLDQ was a symbol, might not return monthly data easily.
    # print("\n--- Testing a potentially tricky ticker (e.g., previously delisted) ---")
    # tricky_df = download_stock_data("SHLDQ", "2010-01-01", "2012-01-01", interval="1mo")
    # if tricky_df is not None:
    #    print(f"SHLDQ Data (first 5 rows):\n{tricky_df.head()}")
    # else:
    #    print("SHLDQ data not found or error during download, as might be expected.")
    print("\nStock data download module test finished.")
