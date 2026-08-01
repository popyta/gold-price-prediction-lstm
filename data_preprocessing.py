import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# SEQ_LEN 15 বা 30 আপনার মডেলের সাথে সামঞ্জস্যপূর্ণ রাখুন
SEQ_LEN = 15  
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_y.pkl"

def clean_raw_data(file_storage):
    """কলামের ছোট-বড় হাতের অক্ষরের বৈষম্য এবং ইনডেক্সিং বিচ্যুতি দূর করার পারফেক্ট লজিক"""
    # Flask এর file_storage অবজেক্ট অথবা নরমাল পাথ হ্যান্ডেল করার জন্য
    if hasattr(file_storage, 'read'):
        file_storage.seek(0)
    
    df_raw = pd.read_csv(file_storage)
    
    if 'Ticker' in str(df_raw.columns) or 'Price' in str(df_raw.columns):
        if hasattr(file_storage, 'seek'):
            file_storage.seek(0)
        df = pd.read_csv(file_storage, header=1)
    else:
        df = df_raw.copy()
        
    # সব কলামের নামকে ছোট হাতের অক্ষরে কনভার্ট করা হচ্ছে
    df.columns = [str(col).strip().lower() for col in df.columns]
    
    # ডাইনামিক কলাম ম্যাচিং লজিক
    if 'date' in df.columns and 'price' in df.columns:
        df = df[['date', 'price']]
    elif 'date' in df.columns and 'close' in df.columns:
        df = df[['date', 'close']]
    else:
        # কোনো নাম না মিললে প্রথম ২টি কলাম (০ এবং ১) নেওয়া হবে নিরাপদ উপায়ে
        df = df.iloc[:, [0, 1]]
        
    df.columns = ['date', 'price']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.sort_values('date')
    df = df.set_index('date')
    df = df.ffill()
    
    return df

def advanced_feature_engineering(df):
    """টেকনিক্যাল ইন্ডিকেটর ও স্ট্যান্ডার্ড অ্যানুয়ালাইজড ভোলাটিলিটি মেকিং"""
    try:
        if df.empty or 'price' not in df.columns:
            raise ValueError("Missing required price column or file is corrupted.")

        df['Return'] = df['price'].pct_change()
        df['MA_5'] = df['price'].rolling(5).mean()
        df['MA_20'] = df['price'].rolling(20).mean()
        df['Volatility'] = df['Return'].rolling(10).std() * np.sqrt(252)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        raise RuntimeError(f"Critical Data Feature Engineering Failure: {e}")

def prepare_tensors(df, feature_cols=['price', 'MA_5', 'MA_20', 'Volatility'], is_training=False):
    """ডাটা স্প্লিট ও ৩D ইনপুট টেনসর ক্রিয়েশন (ফিক্সড লজিক)"""
    if len(df) <= SEQ_LEN:
        raise ValueError(f"ফাইলে পর্যাপ্ত ডাটা নেই (মাত্র {len(df)} দিন)। উইন্ডো সাইজ {SEQ_LEN} দিন। অনুগ্রহ করে কমপক্ষে ৩৫+ দিনের ডাটা ব্যবহার করুন।")

    def make_seq(X, y):
        Xs, ys = [], []
        for i in range(SEQ_LEN, len(X)):
            Xs.append(X[i-SEQ_LEN:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    if is_training:
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))

        train_end = int(len(df) * 0.7)
        val_end = int(len(df) * 0.85)

        train = df.iloc[:train_end][feature_cols]
        val = df.iloc[train_end:val_end][feature_cols]
        test = df.iloc[val_end:][feature_cols]

        train_X = scaler_X.fit_transform(train)
        val_X = scaler_X.transform(val)
        test_X = scaler_X.transform(test)

        train_y = scaler_y.fit_transform(train[['price']])
        val_y = scaler_y.transform(val[['price']])
        test_y = scaler_y.transform(test[['price']])

        X_train, y_train = make_seq(train_X, train_y)
        X_val, y_val = make_seq(val_X, val_y)
        X_test, y_test = make_seq(test_X, test_y)

        # ট্রেইনিং এর সময় স্কেলার দুটিকে লোকাল ডিরেক্টরিতে সেভ করে রাখা হচ্ছে
        with open(SCALER_X_PATH, "wb") as f:
            pickle.dump(scaler_X, f)
        with open(SCALER_Y_PATH, "wb") as f:
            pickle.dump(scaler_y, f)

        return X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y

    else:
        # প্রেডিকশনের সময় পূর্বে সংরক্ষিত ট্রেইনিং স্কেলার লোড করা হচ্ছে (Staleness/Data Leakage রোধে)
        if os.path.exists(SCALER_X_PATH) and os.path.exists(SCALER_Y_PATH):
            with open(SCALER_X_PATH, "rb") as f:
                scaler_X = pickle.load(f)
            with open(SCALER_Y_PATH, "rb") as f:
                scaler_y = pickle.load(f)
        else:
            # ফলব্যাক লজিক: যদি কোনো কারণে ফাইল না থাকে তবেই কেবল নতুন করে ফিট হবে
            scaler_X = MinMaxScaler(feature_range=(0, 1))
            scaler_y = MinMaxScaler(feature_range=(0, 1))
            scaler_X.fit(df[feature_cols])
            scaler_y.fit(df[['price']])

        # শুধু transform হবে, fit_transform নয়
        scaled_features = scaler_X.transform(df[feature_cols])
        scaled_prices = scaler_y.transform(df[['price']])

        Xs, ys = make_seq(scaled_features, scaled_prices)
            
        return Xs, ys, scaler_X, scaler_y