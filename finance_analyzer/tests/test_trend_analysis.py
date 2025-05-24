# finance_analyzer/tests/test_trend_analysis.py
import unittest
import pandas as pd
import numpy as np
from finance_analyzer.trend_analysis import (
    calculate_sma, 
    identify_sma_crossovers,
    identify_consecutive_trends,
    generate_buy_sell_indicators # Import new function
)

class TestTrendAnalysis(unittest.TestCase):

    def setUp(self):
        self.dates = pd.to_datetime([
            '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
            '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10'
        ])
        self.close_prices = [100, 102, 101, 103, 105, 104, 102, 100, 103, 106]
        # Added Open, High, Low, Volume for completeness if data_df is used directly
        self.sample_df = pd.DataFrame({
            'Open': self.close_prices, 'High': [x+1 for x in self.close_prices], 
            'Low': [x-1 for x in self.close_prices], 'Close': self.close_prices, 
            'Volume': [1000*x for x in self.close_prices]
        }, index=self.dates)
        self.sample_df.index.name = 'Date'

        # SMAs for testing indicators
        self.sma_short = calculate_sma(self.sample_df, 3) # Example short SMA
        self.sma_long = calculate_sma(self.sample_df, 5)  # Example long SMA

    # --- (Keep existing test methods for calculate_sma, identify_sma_crossovers, identify_consecutive_trends) ---
    def test_calculate_sma_valid(self):
        sma_3 = calculate_sma(self.sample_df, 3)
        self.assertIsInstance(sma_3, pd.Series) # Basic check
        # ... (more detailed assertions if needed, or use existing ones)
        # Expected: (100)/1=100, (100+102)/2=101, (100+102+101)/3=101, (102+101+103)/3=102, ...
        expected_sma_3 = pd.Series([
            100.0, 101.0, 101.0, 102.0, 103.0, 104.0, 103.66666666666667, 
            102.0, 101.66666666666667, 103.0
        ], index=self.dates, name='Close')
        pd.testing.assert_series_equal(sma_3, expected_sma_3, check_dtype=False)


    def test_identify_sma_crossovers_bullish(self):
        # Simplified test case for clarity
        short_sma = pd.Series([10, 12, 15], index=self.dates[:3]) 
        long_sma  = pd.Series([11, 12, 13], index=self.dates[:3]) 
        crossovers = identify_sma_crossovers(short_sma, long_sma)
        if not crossovers.empty: # Check if not empty before asserting details
            self.assertEqual(crossovers.index[0], self.dates[2])
            self.assertEqual(crossovers['Signal'].iloc[0], "Bullish Crossover")
        else: self.fail("Expected a bullish crossover but found none.")


    def test_identify_consecutive_trends_bullish(self):
        df = pd.DataFrame({'Close': [10,11,12,10,13,14,15]}, index=self.dates[:7])
        trends = identify_consecutive_trends(df, num_periods=3)
        if not trends.empty:
            self.assertEqual(trends.index[0], self.dates[6]) # '2023-01-07'
            self.assertEqual(trends['Trend_Signal'].iloc[0], "3-Period Bullish")
        else: self.fail("Expected a 3-period bullish trend but found none.")


    # --- New tests for generate_buy_sell_indicators ---
    def test_generate_buy_sell_indicators_potential_buy(self):
        # Create a scenario for a buy signal:
        # Bullish crossover recently, and current Close > Long SMA
        # Crossover on self.dates[4] ('2023-01-05')
        # self.sample_df['Close'].loc[self.dates[4]] = 105
        # self.sma_short.loc[self.dates[4]] = (101+103+105)/3 = 103
        # self.sma_long.loc[self.dates[4]] = (100+102+101+103+105)/5 = 102.2
        # Before that, on dates[3]: Close=103, ShortSMA=(102+101+103)/3=102, LongSMA=(X+100+102+101+103)/5 = ...
        # Let's make a clear crossover at dates[4]
        temp_short_sma = pd.Series([100,100,100,101,103], index=self.dates[:5]) # Short crosses above long at dates[4]
        temp_long_sma  = pd.Series([101,101,101,101,102], index=self.dates[:5])
        crossover_signals_df = identify_sma_crossovers(temp_short_sma, temp_long_sma) # Should give Bullish at dates[4]
        
        # Verify crossover was detected as expected
        self.assertFalse(crossover_signals_df.empty, "Test setup failed: No crossover detected for buy signal test.")
        self.assertEqual(crossover_signals_df.loc[self.dates[4], 'Signal'], "Bullish Crossover")

        # Now check indicator:
        # On self.dates[5] ('2023-01-06'): Close is 104.
        # self.sma_long (SMA5 for sample_df) at self.dates[5] is (101+103+105+104+102)/5 = 103.
        # Condition: Close (104) > LongSMA (103) is TRUE.
        # Crossover at dates[4] is 1 day before dates[5] (within 3-day lookback).
        
        indicator_df = generate_buy_sell_indicators(
            self.sample_df, temp_short_sma, self.sma_long, crossover_signals_df # Use actual self.sma_long for condition check
        )
        self.assertFalse(indicator_df.empty)
        self.assertEqual(indicator_df.loc[self.dates[5], 'Indicator'], "Potential Buy")
        # Check for dates[6] ('2023-01-07'): Close=102. LongSMA=(103+105+104+102+100)/5=102.8. Close < LongSMA. So Hold.
        self.assertEqual(indicator_df.loc[self.dates[6], 'Indicator'], "Hold/Neutral") 


    def test_generate_buy_sell_indicators_potential_sell(self):
        # Create a scenario for a sell signal:
        # Bearish crossover recently, and current Close < Long SMA
        # Let's make a clear crossover at dates[6] ('2023-01-07')
        temp_short_sma = pd.Series([105,105,105,104,103,102,100], index=self.dates[:7]) # Short crosses below long at dates[6]
        temp_long_sma  = pd.Series([103,103,103,103,103,103,101], index=self.dates[:7])
        crossover_signals_df = identify_sma_crossovers(temp_short_sma, temp_long_sma)
        
        self.assertFalse(crossover_signals_df.empty, "Test setup failed: No crossover detected for sell signal test.")
        self.assertEqual(crossover_signals_df.loc[self.dates[6], 'Signal'], "Bearish Crossover")

        # Now check indicator:
        # On self.dates[7] ('2023-01-08'): Close is 100.
        # self.sma_long (SMA5 for sample_df) at self.dates[7] is (105+104+102+100+103)/5 = 102.8
        # Condition: Close (100) < LongSMA (102.8) is TRUE.
        # Crossover at dates[6] is 1 day before dates[7] (within 3-day lookback).
        indicator_df = generate_buy_sell_indicators(
            self.sample_df, temp_short_sma, self.sma_long, crossover_signals_df
        )
        self.assertFalse(indicator_df.empty)
        self.assertEqual(indicator_df.loc[self.dates[7], 'Indicator'], "Potential Sell")
        # Check for dates[8] ('2023-01-09'): Close=103. LongSMA=(104+102+100+103+106)/5 = 103. Close == LongSMA. So Hold.
        self.assertEqual(indicator_df.loc[self.dates[8], 'Indicator'], "Hold/Neutral")


    def test_generate_buy_sell_indicators_hold_neutral(self):
        # No recent crossovers
        no_crossovers_df = pd.DataFrame(columns=['Signal'])
        indicator_df = generate_buy_sell_indicators(
            self.sample_df, self.sma_short, self.sma_long, no_crossovers_df
        )
        self.assertTrue((indicator_df['Indicator'] == "Hold/Neutral").all())

        # Crossover happened, but condition (Close vs LongSMA) not met
        # Bullish crossover on dates[4], but let's make Close < LongSMA on dates[5]
        crossover_dates = [self.dates[4]]
        crossover_signals_df = pd.DataFrame({'Signal': ["Bullish Crossover"]}, index=crossover_dates)
        
        modified_df = self.sample_df.copy()
        # On dates[5] ('2023-01-06'), Close is 104. Make LongSMA > 104 for this test.
        # We need to ensure the self.sma_long passed reflects this modified condition.
        # It's easier to modify the Close price for the test df.
        modified_df.loc[self.dates[5], 'Close'] = 100 
        # Original self.sma_long at dates[5] is 103. Now Close (100) < LongSMA (103).
        
        indicator_df_cond_not_met = generate_buy_sell_indicators(
            modified_df, self.sma_short, self.sma_long, crossover_signals_df
        )
        self.assertEqual(indicator_df_cond_not_met.loc[self.dates[5], 'Indicator'], "Hold/Neutral")


    def test_generate_buy_sell_indicators_empty_inputs(self):
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype=float)
        # Ensure crossover_signals_df is also empty for a clean test of empty inputs
        empty_crossovers_df = pd.DataFrame(columns=['Signal'])


        result = generate_buy_sell_indicators(empty_df, empty_series, empty_series, empty_crossovers_df)
        self.assertTrue(result.empty)
        
        # Test with valid main df but empty long SMA
        result_empty_long_sma = generate_buy_sell_indicators(
            self.sample_df, self.sma_short, empty_series, empty_crossovers_df
        )
        self.assertTrue(result_empty_long_sma.empty)


if __name__ == '__main__':
    unittest.main()
