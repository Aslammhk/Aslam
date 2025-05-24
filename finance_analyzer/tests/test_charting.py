# finance_analyzer/tests/test_charting.py
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock, ANY # ANY is useful for complex args
from finance_analyzer.charting import plot_stock_chart

class TestCharting(unittest.TestCase):

    def setUp(self):
        self.dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
        self.sample_stock_data = pd.DataFrame({
            'Open': [100, 102, 101, 103, 105],
            'High': [105, 106, 104, 106, 107],
            'Low': [98, 101, 100, 102, 103],
            'Close': [102, 103, 103, 105, 106],
            'Volume': [10000, 12000, 8000, 15000, 13000]
        }, index=self.dates)
        self.sample_stock_data.index.name = 'Date'
        self.ticker_symbol = "TESTADV"

        self.sma_short = pd.Series([101, 102, 102, 104, 105], index=self.dates, name="SMA2")
        self.sma_long = pd.Series([100, 101, 101.5, 102.5, 104], index=self.dates, name="SMA4")
        
        self.crossover_data = pd.DataFrame({
            'Signal': ["Bullish Crossover", "Bearish Crossover"],
            'Short_SMA': [103, 104], # Example values
            'Long_SMA': [102, 105]   # Example values
        }, index=pd.to_datetime(['2023-01-03', '2023-01-05']))
        self.crossover_data.index.name = 'Date'


    @patch('finance_analyzer.charting.mpf.plot')
    @patch('finance_analyzer.charting.mpf.show') # Also mock show if it's called explicitly
    def test_plot_stock_chart_success_basic(self, mock_mpf_show, mock_mpf_plot):
        plot_stock_chart(self.sample_stock_data, self.ticker_symbol)
        mock_mpf_plot.assert_called_once()
        args, kwargs = mock_mpf_plot.call_args
        self.assertEqual(kwargs.get('type'), 'candle')
        self.assertEqual(kwargs.get('volume'), True)
        self.assertIsNone(kwargs.get('addplot')) # No SMAs or crossovers passed

    @patch('finance_analyzer.charting.mpf.plot')
    @patch('finance_analyzer.charting.mpf.show')
    def test_plot_stock_chart_with_smas(self, mock_mpf_show, mock_mpf_plot):
        plot_stock_chart(self.sample_stock_data, self.ticker_symbol, 
                         sma_short_series=self.sma_short, 
                         sma_long_series=self.sma_long)
        mock_mpf_plot.assert_called_once()
        args, kwargs = mock_mpf_plot.call_args
        addplots = kwargs.get('addplot')
        self.assertIsNotNone(addplots)
        self.assertEqual(len(addplots), 2)
        # Check if make_addplot was called correctly (indirectly by checking titles)
        self.assertTrue(any(ap.get('title') == f"SMA {self.sma_short.name}" for ap in addplots))
        self.assertTrue(any(ap.get('title') == f"SMA {self.sma_long.name}" for ap in addplots))


    @patch('finance_analyzer.charting.mpf.plot')
    @patch('finance_analyzer.charting.mpf.show')
    def test_plot_stock_chart_with_crossovers(self, mock_mpf_show, mock_mpf_plot):
        plot_stock_chart(self.sample_stock_data, self.ticker_symbol, 
                         crossover_signals_df=self.crossover_data)
        mock_mpf_plot.assert_called_once()
        args, kwargs = mock_mpf_plot.call_args
        addplots = kwargs.get('addplot')
        self.assertIsNotNone(addplots)
        self.assertEqual(len(addplots), 2) # One for bullish, one for bearish markers

        # Check for scatter plot markers for crossovers
        # This is a bit more involved as we need to check the series passed to make_addplot
        # For simplicity, we'll check that 'scatter' type addplots were created
        scatter_plots = [ap for ap in addplots if ap.get('type') == 'scatter']
        self.assertEqual(len(scatter_plots), 2) # Expecting one for bullish, one for bearish
        
        # Check if the series for scatter plots are correctly formed (at least one non-NaN value)
        # This relies on the internal logic of how bullish_plot_series / bearish_plot_series are made
        bullish_marker_addplot = next((ap for ap in addplots if ap.get('title') == "Bullish Cross"), None)
        bearish_marker_addplot = next((ap for ap in addplots if ap.get('title') == "Bearish Cross"), None)
        
        self.assertIsNotNone(bullish_marker_addplot, "Bullish Cross addplot missing")
        self.assertFalse(bullish_marker_addplot['data'].dropna().empty, "Bullish marker series should have data")
        
        self.assertIsNotNone(bearish_marker_addplot, "Bearish Cross addplot missing")
        self.assertFalse(bearish_marker_addplot['data'].dropna().empty, "Bearish marker series should have data")


    @patch('finance_analyzer.charting.mpf.plot')
    @patch('finance_analyzer.charting.mpf.show')
    def test_plot_stock_chart_with_smas_and_crossovers(self, mock_mpf_show, mock_mpf_plot):
        plot_stock_chart(self.sample_stock_data, self.ticker_symbol, 
                         sma_short_series=self.sma_short, 
                         sma_long_series=self.sma_long,
                         crossover_signals_df=self.crossover_data)
        mock_mpf_plot.assert_called_once()
        args, kwargs = mock_mpf_plot.call_args
        addplots = kwargs.get('addplot')
        self.assertIsNotNone(addplots)
        self.assertEqual(len(addplots), 4) # 2 for SMAs, 2 for crossover markers


    @patch('builtins.print') 
    @patch('finance_analyzer.charting.mpf.plot')
    @patch('finance_analyzer.charting.mpf.show')
    def test_plot_stock_chart_none_data(self, mock_mpf_show, mock_mpf_plot, mock_print):
        plot_stock_chart(None, self.ticker_symbol)
        mock_print.assert_any_call(f"No data provided for {self.ticker_symbol} to plot.")
        mock_mpf_plot.assert_not_called()

    # ... (keep other existing tests like test_plot_stock_chart_empty_data, 
    #      test_plot_stock_chart_missing_columns, test_plot_stock_chart_incorrect_index,
    #      test_plot_stock_chart_index_conversion_success from previous version of test_charting.py)

    @patch('builtins.print')
    @patch('finance_analyzer.charting.mpf.plot')
    @patch('finance_analyzer.charting.mpf.show')
    def test_plot_stock_chart_empty_data(self, mock_mpf_show, mock_mpf_plot, mock_print):
        empty_df = pd.DataFrame()
        plot_stock_chart(empty_df, self.ticker_symbol)
        mock_print.assert_any_call(f"No data provided for {self.ticker_symbol} to plot.")
        mock_mpf_plot.assert_not_called()

    @patch('builtins.print')
    @patch('finance_analyzer.charting.mpf.plot')
    @patch('finance_analyzer.charting.mpf.show')
    def test_plot_stock_chart_missing_columns(self, mock_mpf_show, mock_mpf_plot, mock_print):
        missing_cols_df = pd.DataFrame({'Open': [10]}, index=pd.to_datetime(['2023-01-01']))
        plot_stock_chart(missing_cols_df, self.ticker_symbol)
        # Check if the print call for missing columns contains all expected missing columns
        # This makes the test more robust to the order of columns in the message
        printed_message = ""
        for call_args in mock_print.call_args_list:
            if "Error: Missing required columns for plotting:" in call_args[0][0]:
                printed_message = call_args[0][0]
                break
        self.assertIn("High", printed_message)
        self.assertIn("Low", printed_message)
        self.assertIn("Close", printed_message)
        self.assertIn("Volume", printed_message)
        mock_mpf_plot.assert_not_called()


    @patch('builtins.print')
    @patch('finance_analyzer.charting.mpf.plot')
    @patch('finance_analyzer.charting.mpf.show')
    def test_plot_stock_chart_incorrect_index(self, mock_mpf_show, mock_mpf_plot, mock_print):
        incorrect_index_df = pd.DataFrame({
            'Open': [100], 'High': [105], 'Low': [98], 'Close': [102], 'Volume': [10000]
        }, index=['StringDate'])
        plot_stock_chart(incorrect_index_df, self.ticker_symbol)
        # Check that the specific error message about index conversion failure is printed
        # The exact pandas error message can vary slightly.
        found_error_message = False
        for call_arg in mock_print.call_args_list:
            if "Error: Charting requires a DatetimeIndex. Failed to convert index:" in call_arg[0][0]:
                found_error_message = True
                self.assertIn("StringDate", call_arg[0][0].lower()) # Make check case-insensitive for robustness
                break
        self.assertTrue(found_error_message, "Expected error message for index conversion failure not found.")
        mock_mpf_plot.assert_not_called()


    @patch('finance_analyzer.charting.mpf.plot')
    @patch('finance_analyzer.charting.mpf.show')
    def test_plot_stock_chart_index_conversion_success(self, mock_mpf_show, mock_mpf_plot):
        convertible_index_df = pd.DataFrame({
            'Open': [100], 'High': [105], 'Low': [98], 'Close': [102], 'Volume': [10000]
        }, index=['2023-01-01']) 
        convertible_index_df.index.name = 'Date'
        plot_stock_chart(convertible_index_df, self.ticker_symbol)
        mock_mpf_plot.assert_called_once()


if __name__ == '__main__':
    unittest.main()
