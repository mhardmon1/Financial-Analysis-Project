# Multi-Cloud AI-driven Financial Analysis Project (IBM watsonx + Microsoft Azure)

*A modular, agent‑oriented Python pipeline for multi‑ticker equity analysis, ML forecasting, AI narrative generation, and cloud artifact publishing for downstream BI tools.*

---

## Project Goal & Business Value

**Goal:** Build a Python pipeline that utilizes IBM watsonx and Microsoft Azure to perform end‑to‑end financial stock analysis on one or two stocks per execution, combining:

* **Historical price analytics**
* **Technical indicators** 
* **Dual forecasting ML models**
* **Generative AI narrative**
* **Automated artifact publishing to cloud storage**

**Business Value:**

| Stakeholder | Benefit                                                                                                                                     |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------|
| Quant / Analyst          | Rapid exploratory analysis + ML baselines                                                                                      |
| Product / Ops            | Automated scheduled equity reports with cloud output for dashboards                                                            |
| Advisory / Wealth        | Client‑friendly visuals + natural‑language interpretation                                                                      |
| Data / AI Platform Teams | Demonstrates how to decompose an analytics workflow into **orchestratable micro‑agents** usable in IBM watsonx Orchestrate or other automation frameworks

---

## What the Pipeline Produces

For each ticker you provide (up to two):

* Enriched 3‑year historical dataset (CSV).
* Four technical charts (Price, Moving Averages, Volatility, Volume).
* One 60‑day forecast comparison chart (Linear vs LSTM).
* AI‑generated narrative text summary.
* Automatic upload of all artifacts to Azure Blob.

If two tickers are supplied, an additional **comparative AI narrative** is generated and uploaded.

---

## Architecture & Agent Roles

The code is intentionally written in an **agent‑oriented** style to mirror how these functions could be published as *skills* in IBM watsonx Orchestrate or decomposed into microservices.

```
AnalysisAgent  (Orchestrator)
├── DataFetchAgent      # Fetch 3y history for ticker + ^GSPC benchmark (yfinance)
├── PlotAgent           # Generate 4 technical charts (PNG)
├── LinearModelAgent    # Train fit; forecast N future days (default 60)
├── LSTMAgent           # TensorFlow LSTM seq model; forecast N future days (default 60)
├── LLMNarrativeAgent   # Granite LLM text summary via watsonx.ai
└── AzureUploadAgent    # Upload CSV / TXT / PNG to Azure Blob
```

---

## File Outputs (Per Ticker)

| Filename                    | Description                                                                                           |
| --------------------------- | ----------------------------------------------------------------------------------------------------- |
| `<TICKER>_data.csv`         | Enriched price history w/ indicators: Adj Close, DailyReturn, MA50, MA200, EMA50, Vol30d, RSI14, etc. |
| `<TICKER>_price.png`        | 3‑year adjusted close price line chart.                                                               |
| `<TICKER>_ma.png`           | Price + MA50 + MA200 + EMA50 overlay.                                                                 |
| `<TICKER>_vol.png`          | 30‑day rolling annualized volatility (%).                                                             |
| `<TICKER>_volume.png`       | Daily share volume (vertical lines).                                                                  |
| `<TICKER>_forecast.png`     | 60‑day forecast overlay: Linear vs LSTM vs history.                                                   |
| `<TICKER>_narrative.txt`    | AI‑generated narrative summary for this ticker.                                                       |
| `comparative_narrative.txt` | (Only when 2 tickers input): AI text comparing performance, volatility, and forecasts.                |

All files are written locally and then pushed to Azure Blob Storage.

---

## 6. Requirements
### Python Packages

Minimal `requirements.txt`:

```text
yfinance
pandas
matplotlib
scikit-learn
tensorflow
azure-storage-blob
ibm-watsonx-ai
```

### IBM watsonx Credentials
You need:

* **IBM Cloud API Key** with access to watsonx.ai.
* **watsonx.ai URL / region** (e.g., `https://us-south.ml.cloud.ibm.com`).
* **Project ID** where the Granite model is enabled.

The script calls Granite via the `ModelInference` class. By default it uses model id:
```
ibm/granite-3-2-8b-instruct
```

### Azure Storage Credentials
You need an **Azure Storage Account** (Blob Storage) and a **connection string** with write permission. Also choose a **container name** (will be created if missing).

---

## Configuration in the Script
At the top of `financial_stock_analysis.py` you’ll find constants:

```python
IBM_WATSONX_API_KEY = "IBM_WATSONX_API_KEY"
IBM_WATSONX_URL = "IBM_WATSONX_URL"
IBM_WATSONX_PROJECT_ID = "IBM_WATSONX_PROJECT_ID"
AZURE_STORAGE_CONNECTION_STRING = "AZURE_STORAGE_CONNECTION_STRING"
AZURE_CONTAINER_NAME = "AZURE_CONTAINER_NAME"
```

---

## Running the Script
### Interactive Mode (Prompted)

Run the script and you’ll be prompted for up to two comma‑separated tickers:

```bash
python AI_Financial_Analysis_Project.py
Enter up to two comma-separated tickers: AAPL, MSFT
```

---

## End‑to‑End Pipeline Walkthrough
### Data Retrieval & Benchmark Align

`DataFetchAgent.run(ticker, period='3y')` downloads 3 years of daily OHLCV data using yfinance for:

* The requested ticker.
* Benchmark: `^GSPC` (S\&P 500 index).

### Data Enrichment: Technical Indicators

`AnalysisAgent._enrich()` adds the following columns:

| Column        | Description                                                          |
| ------------- | -------------------------------------------------------------------- |
| `DailyReturn` | Daily % change of Adj Close.                                         |
| `MA50`        | 50‑day simple moving average of Adj Close.                           |
| `MA200`       | 200‑day simple moving average.                                       |
| `Vol30d`      | Rolling 30‑day std dev of `DailyReturn`, annualized by $\sqrt{252}$. |
| `EMA50`       | 50‑day exponential moving average.                                   |
| `RSI14`       | 14‑day Relative Strength Index (Wilder style: avg gain/avg loss).    |

Rows with insufficient lookback are dropped, so earliest \~200 rows may be removed.

### Visualization Set

`PlotAgent.run()` generates four technical charts and saves them as PNGs:

1. **Price Trend** – 3‑year Adj Close.
2. **Moving Averages** – Adj Close + MA50 + MA200 + EMA50.
3. **Volatility** – 30‑day rolling annualized volatility (%).
4. **Volume** – Daily volume as vertical lines.

Each call ends with plt.show() so you see plots interactively; they’re also saved to file.

### Forecast Models (Linear + LSTM)

**LinearModelAgent**

* Fits `sklearn.linear_model.LinearRegression` of *row index -> Adj Close price*.
* Computes in‑sample R².
* Forecast horizon default: 60 trading days (configurable).

**LSTMAgent**

* Uses last 60 standardized price points to predict next point iteratively.
* Architecture: `[LSTM(32) → Dense(1)]`, optimizer='adam', loss='mse'.
* Trains for 20 epochs silently.
* Produces 60‑step ahead forecast by recursive feeding.
* Computes an approximate R² = 1 − (MSE / Var(y)) on training window.

### Forecast Comparison Chart

Back in `AnalysisAgent.run()` a combined chart overlays:

* Historical Adj Close.
* Linear Regression forecast.
* LSTM forecast.

Saved as `<TICKER>_forecast.png`.

### Granite LLM Narrative

A prompt template is built with:

* Ticker symbol.
* Stock 3‑year return % vs S\&P 500.
* Linear & LSTM R².

Granite is called via `LLMNarrativeAgent.run()` -> `call_granite(prompt)`.
The returned text (string) is printed and saved to `<TICKER>_narrative.txt`.

### Azure Artifact Upload

`AzureUploadAgent.run(local, blob)` calls `azure_upload()`:

* Creates the container if it doesn’t exist.
* Uploads the local file (overwrite=True).
* Logs success.

All CSVs, narratives, and images are uploaded for each ticker.

### Comparative Narrative (Two Tickers)

When two tickers are provided, after both runs finish the script:

1. Extracts each ticker’s 3‑year % return from the result dicts.
2. Builds a comparison prompt
3. Calls Granite LLM to generate narrative response.
4. Saves to `comparative_narrative.txt` and uploads to Azure.

---

## Interpreting the Metrics
### 3‑Year Return vs S\&P 500

Computed as:

`3_year_return = ((Adj_close_last / Adj_close_first) - 1) * 100%`

Benchmark return computed analogously using the S\&P 500 index (`^GSPC`) aligned to the stock’s date range.

### Linear Regression R²

In‑sample coefficient of determination from scikit‑learn. Measures variance explained by a *straight‑line* fit over the 3‑year period. High R² implies strong linear drift; low R² suggests mean‑reversion, noise, or nonlinear trend.

### LSTM R² (Approximation)

The LSTM’s R² is **computed manually** as `1 - (MSE / Var(y))` on the training data (scaled series). It is not a hold‑out or forward‑looking score; treat it as *fit quality*, not a forecast accuracy guarantee.

*For production, add a true train/validation split and compute out‑of‑sample metrics (MAE, RMSE, directional hit‑rate, etc.).*

### RSI, Moving Averages, Volatility

* **RSI14** above \~70 suggests overbought; below \~30 suggests oversold (heuristics, not signals).
* **MA50 vs MA200** crossovers often used to mark medium/long‑trend shifts.
* **Vol30d** gives an annualized volatility proxy; compare across tickers for relative risk.

---

## Implementation for IBM watsonx Orchestrate

Because each major operation is encapsulated in an "agent," you can turn the pipeline into a set of **Orchestrate skills**:

| Orchestrate Skill     | Maps To                 | Input           | Output                   |
| --------------------- | ----------------------- | --------------- | ------------------------ |
| Fetch Historical Data | `DataFetchAgent`        | Ticker, Period  | DataFrame / JSON payload |
| Compute Indicators    | `AnalysisAgent._enrich` | Price DF        | Enriched DF              |
| Generate Charts       | `PlotAgent`             | Enriched DF     | Image artifacts          |
| Run Linear Forecast   | `LinearModelAgent`      | Enriched DF     | Forecast array, R²       |
| Run LSTM Forecast     | `LSTMAgent`             | Enriched DF     | Forecast array, R²       |
| Generate Narrative    | `LLMNarrativeAgent`     | Prompt meta     | Text summary             |
| Publish to Azure      | `AzureUploadAgent`      | Local file path | Cloud URI                |

An Orchestrate flow could:

1. Accept user ticker(s).
2. Fan‑out per ticker to fetch + enrich.
3. Parallelize model runs.
4. Gather results; call Granite for narrative.
5. Publish output bundle & notify user.

---

## Automation, Scheduling & Containerization

**Cron / Airflow / GitHub Actions:**

* Wrap the script in a scheduler to refresh nightly or weekly.
* Use stdin or environment variables for tickers.

**Docker:** example:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY financial_stock_analysis.py ./
ENV PYTHONUNBUFFERED=1
CMD ["python", "financial_stock_analysis.py"]
```

Pass secrets at runtime (`docker run -e IBM_WATSONX_API_KEY=...`).

**Kubernetes Job / CronJob:** Mount secrets as environment variables or files; schedule periodic runs; artifacts land in Azure.

---

## Tableau Dashboard Integration 

This section shows how to build an interactive Tableau dashboard using the artifacts produced by the pipeline and stored in Azure Blob Storage.

### Prepare Data in Azure

After running the script you should have (per ticker) at least:

* `<TICKER>_data.csv`
* `<TICKER>_narrative.txt`
* `<TICKER>_forecast.png`
* Additional PNGs (price, MA, vol, volume)

### Connect Tableau to Azure Files

**Download Locally & Use Text/CSV Connector**

* Manually download from Azure or use Azure Storage Explorer to download all CSV and TXT files.
* In Tableau Desktop: *Connect > Text File*; select one or more CSVs.

### Build Tableau Data Model

1. Open Tableau Desktop → *Connect > Text File* → choose `{ticker}_data.csv`.
2. Drag sheet into canvas; ensure Date column parses as *Date*.
3. (If 2nd ticker was provided) Repeat for `{ticker2}_data.csv`; create a **Union** so both load into one table with an auto‑added `[Table Name]` or `[Path]` column. Rename to `Ticker`.

### Create Core Worksheets

**Worksheet: Price & Moving Averages**

* Columns: Date.
* Rows: Adj Close.
* Add MA50, MA200, EMA50 as additional measures.

**Worksheet: Volatility & Volume**

* Columns: Date.
* Rows: Vol30d (%). Add a second axis for Volume (bars). 

**Worksheet: RSI14**

* Columns: Date.
* Rows: RSI14.

### Build the Dashboard Layout

Example layout (desktop 1280×800 min):

| Region           | Content                                                        |
| ---------------- | -------------------------------------------------------------- |
| Top Header       | Parameter drop‑down to pick ticker(s); last refresh date.      |
| Upper            | Price & Moving Averages line chart.                            |                          |
| Lower Left       | Volatility & Volume dual‑axis.                                 |
| Lower Right      | RSI14 w/ bands.                                                |
| Sidebar / Footer | Narrative text + KPI tiles (3‑Year Return vs S\&P, Model R²s). |

### Add Narrative + KPI Tiles

**Narrative Text:**

1. Import `<TICKER>_narrative.txt` as a *text data source* (one row with narrative field) or paste narrative into a Tableau parameter updated by automation.
2. Place a *Text* object on dashboard; insert the narrative field.

**KPI Tiles:** Create calculated fields:

```tableau
ThreeYearReturn := ([Adj Close]/WINDOW_MIN(IF FIRST()==0 THEN [Adj Close] END)) - 1
```

Add formatted `%` KPI text boxes: *Ticker Return*, *S\&P Return*, *Linear R²*, *LSTM R²*.

### Add Ticker Parameter for Interactivity

1. Create Parameter `pTicker` (String, Allow All).
2. Create Filter Calc: `[Ticker] = [pTicker]` OR `[pTicker] = 'ALL'`.
3. Show Parameter Control; wire to all worksheets.

---
