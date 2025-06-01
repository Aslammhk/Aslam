# Smart Forex Buy/Sell Signal Replicator

This script replicates a TradingView indicator like "Smart Forex Buy Sell Signal by Lukas" using Python, Pandas, TA-Lib, and Plotly. It generates Buy, Sell, STRONG Buy/Sell, and SPEC Buy/Sell signals based on EMA crossovers and RSI thresholds.

## Features

- Calculates Exponential Moving Averages (EMAs).
- Calculates the Relative Strength Index (RSI).
- Generates multiple types of trading signals:
    - **Buy/Sell**: Basic signals based on EMA crossover and RSI confirmation.
    - **STRONG Buy/Sell**: Signals with stronger confirmation (e.g., RSI momentum alignment or significant price action).
    - **SPEC Buy/Sell**: Speculative signals that might indicate early opportunities or require more caution.
- Visualizes price data (OHLC) as a candlestick chart.
- Displays EMAs and all signal types on the price chart.
- Shows an RSI subplot with overbought, oversold, and mid-level thresholds.

## Setup

### Dependencies

You'll need Python 3.x and the following libraries:

- pandas
- plotly
- numpy (usually a dependency of pandas or TA-Lib)
- TA-Lib

### Installing TA-Lib

TA-Lib can sometimes be tricky to install.

**Using `talib-binary` (recommended for ease):**
```bash
pip install talib-binary pandas plotly
```

**If `talib-binary` does not work, you may need to install the TA-Lib C library first.**
Instructions vary by operating system:

-   **Windows**:
    1.  Download the TA-Lib C library binaries. Unofficial binaries can be found at [lfd.uci.edu/~gohlke/pythonlibs/#ta-lib](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib). Download the .whl file that matches your Python version and system architecture.
    2.  Install the downloaded .whl file: `pip install TA_Lib‑0.4.xx‑cp3x‑cp3xm‑win_amd64.whl` (adjust filename).
    3.  Then install the Python wrapper: `pip install TA-Lib`
-   **macOS**:
    ```bash
    brew install ta-lib
    pip install TA-Lib
    ```
-   **Linux (Debian/Ubuntu)**:
    ```bash
    sudo apt-get install libta-lib-dev
    pip install TA-Lib
    ```
-   **Linux (Other distributions)**:
    Search for TA-Lib installation instructions specific to your distribution. You generally need to build it from source or find a pre-built package for your package manager.

After installing TA-Lib, install the other Python libraries:
```bash
pip install pandas plotly
```

## How to Run

1.  Ensure all dependencies are installed correctly.
2.  Clone this repository or download the `forex_signal_replicator.py` script.
3.  Run the script from your terminal:
    ```bash
    python forex_signal_replicator.py
    ```
This will generate a plot and display it in your default web browser or Plotly's default rendering environment. The script uses sample data internally for demonstration.

## Signal Logic Overview

-   **EMAs**: Two EMAs are used (e.g., 10-period and 20-period). Crossovers form the basis of signals.
-   **RSI**: A 14-period RSI is used to gauge momentum and overbought/oversold conditions.
    -   Overbought: RSI >= 70
    -   Oversold: RSI <= 30
-   **Signal Hierarchy**:
    1.  **EMA Crossover**: A bullish (short EMA > long EMA) or bearish (short EMA < long EMA) crossover must occur.
    2.  **Basic Confirmation**:
        -   Buy: Bullish crossover AND RSI < 70.
        -   Sell: Bearish crossover AND RSI > 30.
    3.  **STRONG Signal**: If basic conditions met, a STRONG signal is triggered if:
        -   STRONG Buy: RSI < 50 (stronger momentum) OR specific bullish price action.
        -   STRONG Sell: RSI > 50 (stronger momentum) OR specific bearish price action.
        *(A STRONG signal replaces a basic signal).*
    4.  **SPEC Signal**: If basic conditions met but not STRONG, a SPEC signal is triggered if:
        -   SPEC Buy: RSI < 40 (more stringent RSI than basic, but not necessarily indicating full momentum like STRONG).
        -   SPEC Sell: RSI > 60 (more stringent RSI than basic).
        *(A SPEC signal replaces a basic signal if the SPEC conditions are met and STRONG conditions are not).*

The script visualizes these signals on a chart, allowing for analysis of how they correspond to price movements, EMAs, and RSI levels.
