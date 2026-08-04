<div align="center">

# 🪙 Gold Price Prediction System
### 📈 Deep Stacked LSTM Neural Network for Financial Forecasting

<img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow">
<img src="https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras">
<img src="https://img.shields.io/badge/Flask-Web%20Application-black?style=for-the-badge&logo=flask">
<img src="https://img.shields.io/badge/Chart.js-Visualization-ff6384?style=for-the-badge&logo=chartdotjs">
<img src="https://img.shields.io/badge/License-MIT-success?style=for-the-badge">

---

### 🚀 AI Powered Gold Price Forecasting Platform

Predict future gold prices using **Deep Stacked LSTM**, advanced financial indicators, automated preprocessing, and an interactive Flask dashboard.

</div>

---

# 🌟 Project Overview

Gold prices fluctuate due to inflation, interest rates, global conflicts, currency exchange, and market sentiment. Traditional statistical forecasting techniques often struggle to capture these highly nonlinear patterns.

This project introduces a **Deep Stacked Long Short-Term Memory (LSTM)** network capable of learning long-term dependencies from historical gold prices.

The system automatically:

✅ Cleans raw financial datasets

✅ Generates technical indicators

✅ Trains Deep Learning models

✅ Predicts the next **7 Days**

✅ Displays beautiful charts inside a Flask web application

---

# 🎯 Objectives

- 📈 Predict future gold prices accurately
- 🧠 Learn long-term sequential market patterns
- ⚡ Reduce forecasting error
- 🌐 Provide an interactive prediction dashboard
- 📊 Visualize model performance

---

# ✨ Key Features

| 🚀 Feature | Description |
|------------|-------------|
| 📂 Smart Data Cleaning | Automatically cleans missing values and formats datasets |
| 📊 Feature Engineering | MA5, MA20, Daily Return & Volatility Calculation |
| 🤖 Deep Stacked LSTM | 2 Hidden LSTM Layers (64 → 32 Units) |
| 🎯 High Prediction Accuracy | Learns nonlinear market behavior |
| 🌐 Flask Dashboard | Interactive prediction interface |
| 📉 Performance Metrics | MAE, RMSE, R² Score |
| 📈 Beautiful Charts | Chart.js + Matplotlib |
| 🔄 Recursive Forecast | Future 7-Day Gold Price Prediction |

---

# 🏗 System Architecture

```text
                    📂 Raw CSV Dataset
                           │
                           ▼
                🧹 Data Cleaning Pipeline
                           │
                           ▼
              📊 Feature Engineering Layer
        (MA5 • MA20 • Returns • Volatility)
                           │
                           ▼
               ⚖ MinMaxScaler Normalization
                           │
                           ▼
             🧠 Sequence Generation (15 Days)
                           │
                           ▼
          🤖 Deep Stacked LSTM Neural Network
             ├── LSTM (64 Units)
             ├── Dropout (0.2)
             ├── LSTM (32 Units)
             ├── Dropout (0.2)
             └── Dense Output Layer
                           │
                           ▼
                🌐 Flask Prediction Server
                           │
                           ▼
            📈 Interactive Web Dashboard
                           │
                           ▼
              🪙 Future Gold Price Forecast
                    (Next 7 Days)
```

---

# 🧠 Deep Learning Model

| Layer | Configuration |
|---------|--------------|
| Input | Sequence Length = 15 |
| LSTM Layer 1 | 64 Units |
| Dropout | 0.2 |
| LSTM Layer 2 | 32 Units |
| Dropout | 0.2 |
| Dense | 1 Neuron |
| Optimizer | Adam |
| Loss Function | Mean Squared Error |

---

# 📊 Financial Indicators

The model automatically generates:

- 📈 Moving Average (MA5)
- 📉 Moving Average (MA20)
- 💹 Daily Returns
- 📊 Historical Volatility
- 📅 Time-based Features

---

# 📈 Model Performance

| Metric | Score |
|---------|--------|
| MAE | ⭐ Excellent |
| RMSE | ⭐ Low Error |
| R² Score | ⭐ High Accuracy |
| Forecast Horizon | 7 Days |

---

# 🛠 Technology Stack

| Technology | Purpose |
|------------|----------|
| 🐍 Python | Programming |
| 🤖 TensorFlow | Deep Learning |
| 🧠 Keras | Neural Network |
| 🌐 Flask | Web Framework |
| 📊 Pandas | Data Processing |
| 🔢 NumPy | Numerical Computing |
| 📉 Matplotlib | Visualization |
| 📈 Chart.js | Interactive Charts |
| 💾 Scikit-learn | Scaling & Metrics |

---

# ⚙️ Installation Guide

Follow these steps to set up and run the project on your local machine.

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/popyta/gold-price-prediction-lstm.git
```

## 2️⃣ Navigate to the Project Folder

```bash
cd gold-price-prediction-lstm
```

## 3️⃣ Install Required Dependencies

```bash
pip install -r requirements.txt
```

## 4️⃣ Run the Flask Application

```bash
python app.py
```

## 5️⃣ Open the Application

```text
http://127.0.0.1:5000
```

The Gold Price Prediction dashboard will be displayed, where you can enter input data, generate predictions, and visualize the forecasting results.

---

# 📂 Project Resources

This repository contains all the required resources for the **Gold Price Prediction System** project, including the proposal, dataset, source code, final report, and deployment demonstration video.

| 📄 Resource | 🔗 Link |
|-------------|----------|
| 📊 Dataset | [📂 View Dataset](./data/) |
| 📓 Jupyter Notebook | [📂 View Notebooks](./notebooks/) |
| 💻 Source Code | [📂 Browse Source Code](./) |
| 📘 Final Report | [📄 View Final Report](./docs/report_final_fixed.pdf) |
| 🎥 Deployment Video | [🎬 Watch / Download Demo](./Demo/Gold%20Price%20Prediction%20Using%20Deep%20Learning.mp4) |

---

# 🎥 Deployment Video

The deployment video demonstrates the complete workflow of the project, including:

- ▶️ Running the Flask application
- 📥 Providing input data
- 🤖 Generating gold price predictions
- 📊 Displaying prediction results and charts
- 🌐 Using the interactive dashboard

---

# 👨‍💻 Author

### ❤️ Developed by

**Popy Talukdar**

Department of Computer Science & Engineering

North East University Bangladesh

---

<div align="center">

## ⭐ If you like this project, give it a Star ⭐

Made with ❤️ using **Python • TensorFlow • Flask**

</div>
