# Financial Stock Analysis Project
# ==============================================================================
# Author: Matthew Hardmon
# -----------------------------------------------------------------------
# This Python script demonstrates an agent-oriented pipeline for analyzing up to 
# two stock tickers using IBM watsonx services and Microsoft Azure storage.
# 
# It fetches 3 years of historical stock data, computes technical indicators 
# (including moving averages, volatility, EMA, RSI), generates charts, performs 
# forecasting with Linear Regression and an LSTM model, produces an AI-generated 
# narrative, and uploads results to Azure Blob storage.
# 
# The code is structured into "agents" (classes) to facilitate reuse in IBM watsonx 
# Orchestrate, where each task can be an independent skill.
#
# Agent‑oriented pipeline:
#   • DataFetchAgent     – pulls OHLCV via yfinance
#   • PlotAgent          – matplotlib visuals
#   • LinearModelAgent   – scikit-learn LinearRegression forecast
#   • LSTMAgent          – LSTM sequence model forecast
#   • LLMNarrativeAgent  – Granite LLM summary (watsonx.ai)
#   • AzureUploadAgent   – uploads artifacts to Azure Blob
#   • AnalysisAgent      – orchestrates all steps per ticker
# ==============================================================================

from __future__ import annotations
import os, sys
from datetime import timedelta
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5)
import yfinance as yf
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras import layers, models

# Azure and watsonx credentials 
from azure.storage.blob import BlobServiceClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextGenParameters

-----------------------------------------------------------------------------------------------------------------------------------------------------

# Configuration: Modify IBM Watsonx and Azure Credentials accordingly
IBM_WATSONX_API_KEY = "IBM_WATSONX_API_KEY"
IBM_WATSONX_URL = "IBM_WATSONX_URL" 
IBM_WATSONX_PROJECT_ID = "IBM_WATSONX_PROJECT_ID" 
AZURE_STORAGE_CONNECTION_STRING = "AZURE_STORAGE_CONNECTION_STRING" 
AZURE_CONTAINER_NAME = "stock-analysis-results" 

-----------------------------------------------------------------------------------------------------------------------------------------------------

# Utility Function
def azure_upload(local: str, blob_name: str):
    """
    Upload artifacts to Azure Blob Storage.
    """
    conn = AZURE_STORAGE_CONNECTION_STRING
    container = AZURE_CONTAINER_NAME
    if not conn or BlobServiceClient is None:
        print(f"[Azure] Skipped upload of {blob_name} (no connection or SDK).")
        return
    service = BlobServiceClient.from_connection_string(conn)
    container_client = service.get_container_client(container)
    try:
        container_client.create_container()
    except Exception:
        pass
    with open(local, 'rb') as f:
        container_client.upload_blob(name=blob_name, data=f, overwrite=True)
    print(f"[Azure] Uploaded {blob_name}")

-----------------------------------------------------------------------------------------------------------------------------------------------------

# Utility Function
def call_granite(prompt: str) -> str:
    """
    Call IBM watsonx.ai Granite LLM to generate text for the given prompt.
    """
    if Credentials is None or ModelInference is None:
        return "(watsonx-ai SDK not installed)"
    if not IBM_WATSONX_API_KEY or not IBM_WATSONX_PROJECT_ID:
        return "(watsonx creds missing)"
    try:
        llm = ModelInference(
            credentials=Credentials(api_key=IBM_WATSONX_API_KEY, url=IBM_WATSONX_URL), 
            project_id=IBM_WATSONX_PROJECT_ID, 
            model_id="ibm/granite-3-2-8b-instruct",
            params=TextGenParameters(max_new_tokens=400)
        )
        response = llm.generate_text(prompt)
        return response
    except Exception as e:
        return f"(Granite call failed: {e})"

-----------------------------------------------------------------------------------------------------------------------------------------------------

# Base Agent class
class Agent:
    def __init__(self, name: str):
        self.name = name
    def log(self, msg: str):
        print(f"[{self.name}] {msg}")

-----------------------------------------------------------------------------------------------------------------------------------------------------

# Agent Class
class DataFetchAgent(Agent):
    """Fetches historical stock data (plus S&P 500 benchmark) via yfinance."""
    def run(self, ticker: str, period: str = '3y') -> tuple[pd.DataFrame, pd.DataFrame]:
        print(f"fetching data for {ticker}")
        stock_df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
        bench_df = yf.download('^GSPC', period=period, auto_adjust=False, progress=False)
        if stock_df.empty:
            raise ValueError(f"No data for ticker {ticker}")
        # Align benchmark dates with stock data range
        bench_df = bench_df.loc[stock_df.index.min(): stock_df.index.max()]
        print(f"(Fetching {ticker} data complete")
        return stock_df, bench_df

-----------------------------------------------------------------------------------------------------------------------------------------------------

# Agent Class        
class PlotAgent(Agent):
    """Agent to generate and save plots from enriched DataFrame."""
    def run(self, df: pd.DataFrame, ticker: str):
        # Generate all required charts and save as PNG files
        print(f"Generating plots for {ticker}")
        print(f"Generating price chart")
        self._price(df, ticker)
        print(f"Generating ma chart")
        self._ma(df, ticker)
        print(f"Generating vol chart")
        self._vol(df, ticker)
        print(f"Generating volbar chart")
        self._volbar(df, ticker)
        print(f"Done Generating charts for {ticker}")
        
    def _price(self, df, t):
        plt.figure()
        plt.plot(df.index.to_pydatetime(), df['Adj Close'])
        plt.title(f'{t} Price (3Y)')
        plt.ylabel('USD')
        plt.tight_layout()
        plt.savefig(f"{t}_price.png")
        plt.show()
        
    def _ma(self, df, t):
        plt.figure()
        plt.plot(df.index.to_pydatetime(), df['Adj Close'], color='gray', label='Adj Close')
        plt.plot(df.index.to_pydatetime(), df['MA50'], label='MA50', color='orange')
        plt.plot(df.index.to_pydatetime(), df['MA200'], label='MA200', color='red')
        plt.plot(df.index.to_pydatetime(), df['EMA50'], label='EMA50', color='blue')
        plt.title(f'{t}: Moving Averages (50/200) & EMA50')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{t}_ma.png")
        plt.show()
        
    def _vol(self, df, t):
        plt.figure()
        plt.plot(df.index.to_pydatetime(), df['Vol30d'] * 100, color='purple')
        plt.title(f'{t} 30-Day Volatility (%)')
        plt.tight_layout()
        plt.savefig(f"{t}_vol.png")
        plt.show()
        
    def _volbar(self, df, t):
        plt.figure(figsize=(10, 3))
        dates = df.index.to_pydatetime()
        vols  = df['Volume'].astype(float).values
        plt.vlines(dates, [0], vols, color='grey', linewidth=2)
        plt.title(f'{t} Daily Volume')
        plt.tight_layout()
        plt.savefig(f"{t}_volume.png")
        plt.show()

-----------------------------------------------------------------------------------------------------------------------------------------------------

# Agent Class
class LinearModelAgent(Agent):
    """Fits a linear regression (time index -> price) and forecasts future prices."""
    def run(self, df: pd.DataFrame, horizon: int = 60):
        print(f"Training linear regression model")
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['Adj Close'].values
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)
        future_X = np.arange(len(df), len(df) + horizon).reshape(-1, 1)
        preds = model.predict(future_X)
        print(f"Linear regression model training done")
        return preds, r2

-----------------------------------------------------------------------------------------------------------------------------------------------------

# Agent Class
class LSTMAgent(Agent):
    """Agent for LSTM-based sequence forecasting."""
    def run(self, df: pd.DataFrame, horizon: int = 60):
        print("preparing data for LSTM")
        seq_len = 60
        price_series = df['Adj Close'].astype(float).values
        if len(price_series) < seq_len + 1:
            raise ValueError("Not enough data for LSTM")

        # Standardize
        mean, std = price_series.mean(), price_series.std()
        scaled = (price_series - mean) / std

        X, y = [], []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i-seq_len:i])
            y.append(scaled[i])
        X = np.array(X).reshape(-1, seq_len, 1)
        y = np.array(y)

        model = models.Sequential([
            layers.Input(shape=(seq_len, 1)),
            layers.LSTM(32),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=20, verbose=0)

        r2 = 1 - model.evaluate(X, y, verbose=0) / np.var(y)
        print(f"LSTM R2={r2:.3f}")

        # Forecast
        seq = scaled[-seq_len:].copy()
        preds = []
        for _ in range(horizon):
            pred_scaled = float(model.predict(seq.reshape(1, seq_len, 1), verbose=0)[0, 0])
            pred_price = pred_scaled * std + mean
            preds.append(pred_price)
            seq = np.append(seq[1:], pred_scaled)
        return np.array(preds), r2

-----------------------------------------------------------------------------------------------------------------------------------------------------

# Agent Class
class LLMNarrativeAgent(Agent):
    """Agent to generate summary narrative using Granite (IBM watsonx.ai)."""
    def run(self, prompt: str):
        print(f"Calling Granite")
        return call_granite(prompt)

# Agent Class
class AzureUploadAgent(Agent):
    """Agent to upload a local file to Azure Blob Storage."""
    def run(self, local: str, blob: str):
        print(f"Uploading data to Azure")
        azure_upload(local, blob)

-----------------------------------------------------------------------------------------------------------------------------------------------------

# Coordinator Agent
class AnalysisAgent(Agent):
    """Orchestrates the end-to-end analysis for a single stock ticker."""
    def __init__(self):
        super().__init__('Analysis')
        self.fetch = DataFetchAgent('Fetch')
        self.plot  = PlotAgent('Plot')
        self.lin   = LinearModelAgent('LinearReg')
        self.lstm  = LSTMAgent('LSTM')
        self.llm   = LLMNarrativeAgent('Granite')
        self.up    = AzureUploadAgent('Azure')
        
    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to raw price DataFrame."""
        d = df.copy()
        d['DailyReturn'] = d['Adj Close'].pct_change()
        d['MA50']   = d['Adj Close'].rolling(50).mean()
        d['MA200']  = d['Adj Close'].rolling(200).mean()
        d['Vol30d'] = d['DailyReturn'].rolling(30).std() * np.sqrt(252)
        # Add Exponential Moving Average and Relative Strength Index
        d['EMA50']  = d['Adj Close'].ewm(span=50, adjust=False).mean()
        delta = d['Adj Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        roll_up = gain.rolling(14).mean()
        roll_down = loss.rolling(14).mean()
        RS = roll_up / roll_down
        d['RSI14'] = 100 - 100/(1 + RS)
        return d.dropna()
        
    def run(self, ticker: str) -> Dict:
        # 1) Data retrieval & enrichment
        print(f"Starting analysis for {ticker}")
        raw_df, bench_df = self.fetch.run(ticker)
        df = self._enrich(raw_df)

        # 2) Calculate 3-year return for stock vs S&P 500
        stock_ret = float((df['Adj Close'].iloc[-1] / df['Adj Close'].iloc[0] - 1) * 100)
        bench_df = bench_df.dropna()
        bench_ret = float((bench_df['Adj Close'].iloc[-1] / bench_df['Adj Close'].iloc[0] - 1) * 100)

        # 3) Visualizations (price, MA, volatility, volume charts)
        self.plot.run(df, ticker)

        # 4) Forecasts using two models (Linear and LSTM)
        lin_pred, lin_r2 = self.lin.run(df)
        lstm_pred, lstm_r2 = self.lstm.run(df)

        # 5) Plot 30-day forecast comparison
        print(f"Generating forecast comparison chart")
        future_dates = [df.index[-1] + timedelta(days=i + 1) for i in range(len(lin_pred))]
        plt.figure()
        plt.plot(df.index, df['Adj Close'], label='Historical')
        plt.plot(future_dates, lin_pred, '--', label='Linear Reg')
        plt.plot(future_dates, lstm_pred,  ':', label='LSTM')
        plt.title(f'{ticker} 60-Day Forecasts')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{ticker}_forecast.png")
        plt.show()
        # 6) Generate Granite LLM narrative summary
        prompt = (
            "You are a financial analysis assistant. "
            f"Analyze the provided data for {ticker}. "
            f"The stock's 3‑year return is {stock_ret:.1f}% vs S&P 500 {bench_ret:.1f}%. "
            f"Linear model R² = {lin_r2:.3f}; LSTM R² = {lstm_r2:.3f}. "
            "Summarize technical indicators and explain the differences between the 60‑day "
            "linear and LSTM forecasts. Provide actionable insight."
        )
        narrative = self.llm.run(prompt)
        print("Granite Narrative:", narrative)

        # 7) Persist artifacts to Azure Blob (CSV, TXT, PNG files)
        print(f"Uploading artifacts to Azure")
        csv_name = f"{ticker}_data.csv"
        df.to_csv(csv_name)
        self.up.run(csv_name, csv_name)
        txt_name = f"{ticker}_narrative.txt"
        open(txt_name, 'w').write(str(narrative))
        self.up.run(txt_name, txt_name)
        # Upload all chart images
        for img in [f"{ticker}_price.png", f"{ticker}_ma.png", f"{ticker}_vol.png", f"{ticker}_volume.png", f"{ticker}_forecast.png"]:
            self.up.run(img, img)
        print(f"Finished analysis for {ticker}")
        # 8) Return summary results (for optional comparative narrative if two tickers)
        return {
            'ticker': ticker,
            'return_pct': stock_ret,
            'lin_pred_final': float(lin_pred[-1]),
            'lstm_pred_final': float(lstm_pred[-1]),
            'lin_r2': lin_r2,
            'lstm_r2': lstm_r2,
            'narrative': narrative,
        }

-----------------------------------------------------------------------------------------------------------------------------------------------------

# Script entry point
if __name__ == "__main__":
    tickers_input = input("Enter up to two comma-separated tickers: ").upper().strip()
    tickers = [t.strip() for t in tickers_input.split(',') if t.strip()][:2]
    if not tickers:
        sys.exit("No tickers provided.")

    runner = AnalysisAgent()
    results = [runner.run(t) for t in tickers]

    # If two tickers were provided, generate a comparative narrative
    if len(results) == 2:
        t1, t2 = results[0]['ticker'], results[1]['ticker']
        p1, p2 = results[0]['return_pct'], results[1]['return_pct']
        prompt = (
            f"Compare {t1} and {t2}. {t1} returned {p1:.1f}% and {t2} returned {p2:.1f}% over the last year. "
            "Highlight differences in volatility and in the two forecasting models."
        )
        comp_text = runner.llm.run(prompt)
        print("Comparative Narrative:", comp_text)
        open("comparative_narrative.txt", 'w').write(str(comp_text))
        runner.up.run("comparative_narrative.txt", "comparative_narrative.txt")

    print("Analysis complete.")
