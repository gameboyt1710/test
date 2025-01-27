import yfinance as yf
import pandas as pd
import numpy as np

def download_stock_data(ticker, start_date, end_date):
    """
    Download stock data using Yahoo Finance.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    """
    Preprocess stock data.
    """
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]  # Selecting relevant columns
    data['Date'] = pd.to_datetime(data.index)
    data = data.set_index('Date')
    data = data.dropna()
    return data

def add_technical_indicators(df):
    """
    Adds technical indicators to the stock data.
    """
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    return df

def compute_rsi(series, period=14):
    """
    Computes the Relative Strength Index (RSI) for a given series.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
