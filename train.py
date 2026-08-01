import os
import sys
import numpy as np
import pandas as pd

# আপনার নিজস্ব মডিউল থেকে ফাংশনগুলো লোড করা হচ্ছে
from data_preprocessing import advanced_feature_engineering, prepare_tensors
from model import build_stacked_lstm, get_early_stopping

def run_training_pipeline(csv_path="cleaned_gold_price.csv"):
    """
    সম্পূর্ণ ট্রেইনিং পাইপলাইন এক্সিকিউশন ফাংশন।
    এটি ডাটা লোড করবে, ফিচার ইঞ্জিনিয়ারিং করবে, ৩D টেনসর তৈরি করে মডেল ট্রেইন ও সেভ করবে।
    """
    # ফাইল পাথ চেক (যদি রুট ডিরেক্টরিতে থাকে তবে সরাসরি 'cleaned_gold_price.csv' ব্যবহার করবে)
    if not os.path.exists(csv_path):
        # যদি কোনো কারণে ডাটা ফোল্ডারের ভেতর থাকে, তার জন্য ফলব্যাক চেক
        alternative_path = os.path.join("data", csv_path)
        if os.path.exists(alternative_path):
            csv_path = alternative_path
        else:
            print(f"❌ Error: '{csv_path}' ফাইলটি বর্তমান ডিরেক্টরি বা data/ ফোল্ডারে পাওয়া যায়নি।")
            print("👉 অনুগ্রহ করে প্রথমে 'Data_Analysis_and_Cleaning.ipynb' রান করে ফাইলটি তৈরি করুন।")
            return None
        
    print(f"📖 Loading dataset from: {csv_path}...")
    try:
        df = pd.read_csv(csv_path, index_col='date', parse_dates=True)
    except Exception as e:
        print(f"❌ ফাইলটি রিড করতে সমস্যা হয়েছে: {e}")
        return None
    
    print("⚙️ Engineering technical indicators (Returns, MA_5, MA_20, Volatility)...")
    df = advanced_feature_engineering(df)
    
    print("📊 Constructing 3D input tensors for Stacked LSTM...")
    # is_training=True দেওয়ার কারণে এটি একই সাথে স্কেলার ২টিকেও লোকাল ড্রাইভে .pkl আকারে সেভ করে দেবে
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y = prepare_tensors(df, is_training=True)
    
    print(f"📐 Train Shape: {X_train.shape} | Val Shape: {X_val.shape} | Test Shape: {X_test.shape}")
    
    # ইনপুট শেপ ডাইনামিকালি পাস করা হচ্ছে (TIMESTEPS, FEATURES)
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_stacked_lstm(input_shape)
    early_stop = get_early_stopping()
    
    print("\n🚀 Training Deep Learning Network in Progress...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,          # মডেল আর্কিটেকচারের জন্য ৮০টি ইপক পর্যন্ত ট্রেইনিং এলাউড, আর্লি স্টপিং বাকিটা হ্যান্ডেল করবে
        batch_size=16,      # ২৫২টি রো এর ডাটার জন্য ১৬ ব্যাচ সাইজ বেশি অপ্টিমাইজড রেজাল্ট দেয়
        callbacks=[early_stop],
        verbose=1
    )
    
    # মডেল এক্সপোর্ট
    model.save("gold_model.keras")
    print("\n" + "="*50)
    print("✔ Model successfully trained and saved as 'gold_model.keras'")
    print(f"✔ Target Matrix Scalers preserved in current workspace.")
    print("="*50)
    
    return history

if __name__ == "__main__":
    # কোড রান করার মেইন এন্ট্রি পয়েন্ট
    run_training_pipeline()