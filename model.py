import os

# TensorFlow-এর ইন্টারনাল ওয়ার্নিং এবং লগ মেসেজ মিনিমাইজ করা
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf

# Keras এর স্ট্যান্ডার্ড ইম্পোর্ট
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input
from keras.callbacks import EarlyStopping

# ডিটারমিনিজম এবং রিপ্রোডুসিবিলিটি সেটআপ (প্রতিবার যাতে একই রেজাল্ট আসে)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(42)
tf.random.set_seed(42)

def build_stacked_lstm(input_shape):
    """ডুয়াল লেয়ার মেমোরি সেল সম্পন্ন Stacked LSTM আর্কিটেকচার"""
    # মেমোরি লিক এড়াতে ব্যাকএন্ড সেশন ক্লিয়ার করা
    keras.backend.clear_session()
    
    model = Sequential([
        Input(shape=input_shape),
        # প্রথম LSTM লেয়ার (পরবর্তী LSTM লেয়ারের জন্য return_sequences=True রাখা হয়েছে)
        LSTM(128, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        # দ্বিতীয় LSTM লেয়ার (ডেন্স লেয়ারে পাঠানোর জন্য return_sequences=False)
        LSTM(64, return_sequences=False, activation='tanh'),
        Dropout(0.2),
        # ডিপ ডেন্স রিগ্রেশন হেড
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    # Loss Function (MSE) এবং ইভ্যালুয়েশন মেট্রিক কম্পাইল করা
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def get_early_stopping():
    """ওভারফিটিং রোধে আর্লি স্টপিং কলব্যাক"""
    return EarlyStopping(
        monitor='val_loss',
        patience=7,          # ৭টি ইপক পর্যন্ত লস না কমলে ট্রেইনিং ব্রেক করবে
        min_delta=1e-5,      # নূন্যতম এই পরিমাণ ইমপ্রুভমেন্ট কাউন্ট হবে
        restore_best_weights=True # বেস্ট ওয়েটস ফিরিয়ে আনবে
    )