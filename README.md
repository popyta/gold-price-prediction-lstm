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

## 🚀 AI Powered Gold Price Forecasting Platform

Predict future gold prices using **Deep Stacked LSTM**, advanced financial indicators, automated preprocessing, and an interactive Flask dashboard.

</div>

---

# 🌟 Project Overview

Gold prices fluctuate due to inflation, interest rates, currency exchange rates, geopolitical events, and overall market sentiment. Traditional statistical forecasting techniques often struggle to capture these highly nonlinear and time-dependent patterns.

This project presents a **Deep Stacked Long Short-Term Memory (LSTM)** based forecasting system capable of learning long-term dependencies from historical gold price data. The model is trained using multiple financial indicators and integrated into a Flask web application for interactive forecasting and visualization.

### The system automatically:

- ✅ Cleans raw financial datasets
- ✅ Performs feature engineering
- ✅ Trains a Deep Stacked LSTM model
- ✅ Predicts future gold prices for the next **7 days**
- ✅ Displays prediction results with interactive charts

---

# 🎯 Project Objectives

- 📈 Predict future gold prices with high accuracy
- 🧠 Learn long-term sequential market behavior
- ⚡ Reduce forecasting error
- 🌐 Provide an easy-to-use web application
- 📊 Visualize predictions and model performance

---

# ✨ Key Features

| 🚀 Feature | Description |
|------------|-------------|
| 📂 Smart Data Cleaning | Automatically handles missing values and data formatting |
| 📊 Feature Engineering | Moving Average (MA5, MA20), Daily Returns & Volatility |
| 🤖 Deep Stacked LSTM | Two Hidden LSTM Layers (64 → 32 Units) |
| 🎯 Accurate Prediction | Learns nonlinear financial patterns |
| 🌐 Flask Dashboard | Interactive web-based prediction system |
| 📉 Performance Evaluation | MAE, RMSE & R² Score |
| 📈 Data Visualization | Matplotlib & Chart.js Graphs |
| 🔄 Recursive Forecasting | Predicts Gold Prices for the Next 7 Days |

---

# 🏗️ System Architecture

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
| Input Layer | Sequence Length = 15 |
| LSTM Layer 1 | 64 Units |
| Dropout | 0.20 |
| LSTM Layer 2 | 32 Units |
| Dropout | 0.20 |
| Dense Output | 1 Neuron |
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |

---

# 📊 Financial Indicators

The system automatically generates the following technical indicators:

- 📈 Moving Average (MA5)
- 📉 Moving Average (MA20)
- 💹 Daily Return
- 📊 Historical Volatility
- 📅 Time-based Features

---

# 📈 Model Performance

| Metric | Result |
|---------|---------|
| MAE | ⭐ Excellent |
| RMSE | ⭐ Low Error |
| R² Score | ⭐ High Accuracy |
| Forecast Horizon | 7 Days |

---

# 🛠 Technology Stack

| Technology | Purpose |
|------------|----------|
| 🐍 Python | Programming Language |
| 🤖 TensorFlow | Deep Learning Framework |
| 🧠 Keras | Neural Network API |
| 🌐 Flask | Web Application Framework |
| 📊 Pandas | Data Processing |
| 🔢 NumPy | Numerical Computing |
| 📉 Matplotlib | Visualization |
| 📈 Chart.js | Interactive Dashboard |
| 💾 Scikit-learn | Data Scaling & Evaluation |

---

# 📁 Project Structure

```text
Gold-Price-Prediction-System/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── gold_price_dataset.csv
│
├── notebooks/
│   └── Gold_Price_Prediction.ipynb
│
├── model/
│   ├── lstm_model.keras
│   └── scaler.pkl
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── docs/
│   └── report_final_fixed.pdf
│
└── Demo/
    └── Gold Price Prediction Using Deep Learning.mp4
```

---

# ⚙️ Installation Guide

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/popyta/gold-price-prediction-lstm.git
```

## 2️⃣ Navigate to the Project Directory

```bash
cd gold-price-prediction-lstm
```

## 3️⃣ Install Required Packages

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

You can now access the **Gold Price Prediction Dashboard**, provide input data, generate predictions, and visualize future gold prices.

---

# 📂 Project Resources

| 📄 Resource | 📎 Description |
|-------------|----------------|
| 📊 Dataset | Historical Gold Price Dataset |
| 📓 Jupyter Notebook | Model Development & Training |
| 💻 Source Code | Flask Application & Deep Learning Model |
| 📘 Final Report | Complete Project Documentation |
| 🎥 Deployment Video | Project Demonstration |

---

# 🔮 Future Improvements

- 📱 Mobile Responsive Interface
- ☁ Cloud Deployment
- 📈 Live Gold Price API Integration
- 🔔 Email Notification System
- 📡 Automatic Dataset Updates
- 🤖 Transformer-Based Forecasting Models
- 📊 Advanced Interactive Dashboard

---

# 👨‍💻 Author

### ❤️ Developed by

**Popy Talukdar**

Department of Computer Science & Engineering

North East University Bangladesh

---

# 📜 License

This project is released under the **MIT License**.

---

<div align="center">

## ⭐ If you like this project, don't forget to give it a Star! ⭐

Made with ❤️ using **Python • TensorFlow • Keras • Flask**

</div>
