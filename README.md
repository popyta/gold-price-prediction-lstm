# 🪙 Gold Price Prediction System using Deep Stacked LSTM

An end-to-end, high-precision predictive analytics platform engineered to forecast gold market trends using a **Deep Stacked Long Short-Term Memory (LSTM)** neural network. The platform features automated data cleaning, financial feature engineering, interactive web interfaces, and comprehensive diagnostic performance visualizations.

---

## 📌 Executive Overview

Financial time-series data, specifically gold spot prices, pose severe challenges to traditional mathematical models due to non-linearity, stochastic market noise, regime shifts, and multi-scale volatility. 

To resolve this issue, this platform implements a **2-Layer Stacked LSTM Architecture** with specialized memory gates configured to capture long-term sequence dependencies and subtle micro-trends. Deployed via a modern **Python Flask** micro-framework, the system delivers real-time **7-day recursive look-ahead predictions ($D+1$ to $D+7$)** alongside an interactive diagnostic dashboard rendered using **Chart.js**.

---

## ✨ Key Features

* **Automated Data Sanitization:** Dynamic pipeline that cleans string boundaries, safely parses datetime sequences, handles missing data, and standardizes column mappings automatically.
* **Hand-Crafted Financial Features:** Calculates multi-scale moving averages (**MA5**, **MA20**) and annualized asset volatility metrics ($\text{Volatility} = \sigma_{10}(\text{Return}) \times \sqrt{252}$).
* **Deep Stacked Architecture:** Utilizes a stacked LSTM structure (64 units ➔ 32 units) integrated with Dropout layers (0.2) to prevent overfitting.
* **Interactive Visualization Hub:** Web dashboard allowing users to upload transaction histories and instantly view real-time recursive 7-day trend projections and diagnostic charts.
* **Comprehensive Model Diagnostics:** Incorporates back-testing analysis including loss decay curves, residual error distribution, SMA structural filtering, and monthly volatility boxplots.

---

## 🛠️ System Architecture & Execution Flow

```text
Raw CSV Input File
       │
       ▼
Dynamic Column Mapping & Cleaning ───────── (Standardizes dates & prices)
       │
       ▼
Feature Engineering Layer ───────────────── (Calculates MA5, MA20, & Volatility)
       │
       ▼
MinMaxScaler Transformation ─────────────── (Normalizes arrays into [0, 1] range)
       │
       ▼
3D Tensor Generation ────────────────────── (Shape: [Batch, 15, 4])
       │
       ▼
Stacked LSTM Deep Network ───────────────── (Layer 1: 64 units ➔ Layer 2: 32 units ➔ Dense)
       │
       ▼
Flask Micro-Server Engine ────────────────── (Sliding buffer processing & Chart.js rendering)
       │
       ▼
Interactive Web Dashboard ───────────────── (7-Day Recursive Horizon Forecast)
