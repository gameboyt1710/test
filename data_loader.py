# data_loader.py
import yfinance as yf
import pandas as pd

def download_stock_data(ticker, start_date, end_date):
    """
    Download historical stock data from Yahoo Finance.
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

def preprocess_data(data):
    """
    Preprocess the stock data for training.
    Add moving averages and other indicators as features.
    """
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day simple moving average
    data['SMA_200'] = data['Close'].rolling(window=200).mean()  # 200-day simple moving average
    data['RSI'] = calculate_rsi(data['Close'], window=14)
    data.dropna(inplace=True)  # Drop rows with NaN values
    return data

def calculate_rsi(series, window):
    """
    Calculate Relative Strength Index (RSI).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
