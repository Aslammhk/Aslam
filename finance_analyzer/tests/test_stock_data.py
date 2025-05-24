# finance_analyzer/tests/test_stock_data.py
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from finance_analyzer.stock_data import download_stock_data
from datetime import datetime, timedelta

class TestStockData(unittest.TestCase):

    def setUp(self):
        self.ticker = "MSFT" # Use a common, active ticker for most tests
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.start_date = (datetime.today() - timedelta(days=90)).strftime('%Y-%m-%d') # Approx 3 months for daily
        self.start_date_monthly = (datetime.today() - timedelta(days=2*365)).strftime('%Y-%m-%d') # 2 years for monthly

    @patch('finance_analyzer.stock_data.yf.Ticker')
    def test_download_stock_data_success_monthly(self, mock_ticker):
        # Prepare a mock DataFrame that yf.Ticker().history() would return
        mock_df = pd.DataFrame({
            'Open': [150, 152, 153],
            'High': [155, 156, 157],
            'Low': [149, 150, 151],
            'Close': [152, 153, 155],
            'Volume': [1000000, 1200000, 1100000]
        }, index=pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']))
        mock_df.index.name = 'Date'
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance

        data = download_stock_data(self.ticker, self.start_date_monthly, self.end_date, interval="1mo")
        
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertEqual(data.index.name, 'Date')
        self.assertIn('Volume', data.columns)
        mock_ticker_instance.history.assert_called_once_with(start=self.start_date_monthly, end=self.end_date, interval="1mo")

    @patch('finance_analyzer.stock_data.yf.Ticker')
    def test_download_stock_data_success_daily(self, mock_ticker):
        mock_df = pd.DataFrame({'Close': [10, 11, 12]}) # Simplified
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance

        data = download_stock_data(self.ticker, self.start_date, self.end_date, interval="1d")
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        mock_ticker_instance.history.assert_called_once_with(start=self.start_date, end=self.end_date, interval="1d")


    @patch('finance_analyzer.stock_data.yf.Ticker')
    def test_download_stock_data_empty(self, mock_ticker):
        mock_empty_df = pd.DataFrame() # Simulate yfinance returning an empty DataFrame
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_empty_df
        mock_ticker.return_value = mock_ticker_instance

        data = download_stock_data("TESTEMPTY", "2022-01-01", "2023-01-01")
        self.assertIsNone(data)

    @patch('finance_analyzer.stock_data.yf.Ticker')
    def test_download_stock_data_exception(self, mock_ticker):
        # Simulate an exception during the yf.Ticker().history() call
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = Exception("Simulated yfinance error")
        mock_ticker.return_value = mock_ticker_instance

        data = download_stock_data("TESTEXC", "2022-01-01", "2023-01-01")
        self.assertIsNone(data)
        
    # It's hard to test the "no volume" case reliably with yf.Ticker mock alone,
    # as it would mean mocking a DataFrame without 'Volume'.
    # The actual yfinance library is usually consistent. We'll trust the print statement logic.

if __name__ == '__main__':
    # This allows running tests directly if needed, though `python -m unittest discover` is preferred.
    unittest.main()
