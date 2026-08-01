import os
import io
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file

# Keras মডেল লোড করার জন্য সরাসরি আমদানি
import keras
from keras.models import load_model

# আপনার নিজস্ব প্রসেসিং মডিউল
from data_preprocessing import clean_raw_data, advanced_feature_engineering, prepare_tensors, SEQ_LEN

# ReportLab PDF লাইব্রেরি
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

app = Flask(__name__)

# পিডিএফ ডাউনলোডের জন্য গ্লোবাল ক্যাশ মেমোরি
latest_forecast = None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', forecast_data=None, eval_data=None)

@app.route('/predict', methods=['POST'])
def predict():
    global latest_forecast
    if 'file' not in request.files:
        return render_template('index.html', error_msg="কোনো ফাইল সিলেক্ট করা হয়নি।")
        
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error_msg="ফাইলটি খালি বা অবৈধ।")

    try:
        # ১. ডাটা লোড ও ক্লিনিং পাইপলাইন
        df_cleaned = clean_raw_data(file)
        df_features = advanced_feature_engineering(df_cleaned)
        
        # ২. টেস্ট ডাটা টেনসর প্রিপারেশন
        X_test, y_test, scaler_X, scaler_y = prepare_tensors(df_features, is_training=False)
        
        # ৩. মডেল ভ্যালিডেশন ও লোডিং
        if not os.path.exists("gold_model.keras"):
            return render_template('index.html', error_msg="gold_model.keras সার্ভারে পাওয়া যায়নি! আগে train.py রান করুন।")
            
        model = load_model("gold_model.keras")
        
        # ৪. ব্যাক-টেস্টিং ইভ্যালুয়েশন জেনারেশন
        pred = model.predict(X_test, verbose=0)
        y_pred = scaler_y.inverse_transform(pred).flatten()
        y_true = scaler_y.inverse_transform(y_test).flatten()
        
        # রেসিডুয়াল এরর এবং হিস্টোগ্রাম ডিস্ট্রিবিউশন
        residuals = y_true - y_pred
        counts, bins = np.histogram(residuals, bins=20)
        
        eval_data = {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "res_counts": counts.tolist(),
            "res_bins": [f"{b:.1f}" for b in bins[:-1]],
            "scatter": [{"x": float(t), "y": float(p)} for t, p in zip(y_true, y_pred)]
        }
        
        # ৫. অটোরিগ্রেসিভ রিকার্সিভ ফোরকাস্ট (৭ দিন)
        # ডাটার শেষ অংশ থেকে ফিচার ম্যাট্রিক্স তৈরি
        last_features = df_features.iloc[-SEQ_LEN:][['price', 'MA_5', 'MA_20', 'Volatility']]
        test_features_scaled = scaler_X.transform(last_features)
        current_buffer = test_features_scaled.copy().reshape(1, SEQ_LEN, test_features_scaled.shape[1])
        
        future_scaled = []
        for _ in range(7):
            nxt_scaled = model.predict(current_buffer, verbose=0)[0, 0]
            future_scaled.append(nxt_scaled)
            
            # রিকার্সিভলি বাফার স্লাইড ও আপডেট করা
            new_row = current_buffer[0, -1].copy()
            new_row[0] = nxt_scaled  # প্রাইস কলাম আপডেট
            current_buffer = np.concatenate([current_buffer[:, 1:, :], new_row.reshape(1, 1, -1)], axis=1)
            
        future_prices = scaler_y.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
        
        # ফ্রন্ট-এন্ড ও পিডিএফের জন্য ডেটা স্ট্রাকচার রেডি করা
        forecast_table = [{"day": f"Day {i+1} (D+{i+1})", "price": float(p)} for i, p in enumerate(future_prices)]
        latest_forecast = forecast_table
        
        last_15_prices = y_true[-15:].tolist()
        forecast_days_labels = [f"D+{i+1}" for i in range(7)]
        
        return render_template(
            'index.html',
            forecast_data=forecast_table,
            eval_data=eval_data,
            forecast_days=forecast_days_labels,
            forecast_prices=future_prices.tolist(),
            last_15_prices=last_15_prices
        )
        
    except Exception as e:
        return render_template('index.html', error_msg=f"প্রসেসিং বিচ্যুতি: {str(e)}")


@app.route('/download_pdf')
def download_pdf():
    global latest_forecast
    try:
        buffer = io.BytesIO()
        
        # ReportLab দিয়ে লেটার সাইজ ডকুমেন্ট সেটআপ
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        story = []
        styles = getSampleStyleSheet()
        
        # কাস্টম টেক্সট স্টাইল
        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor('#1A252C'), spaceAfter=15)
        meta_style = ParagraphStyle('MetaStyle', parent=styles['Normal'], fontSize=11, leading=16, textColor=colors.HexColor('#4A5568'))
        
        # ১. রিপোর্টের মেটাডাটা ও হেডার
        story.append(Paragraph("Gold Price Prediction Report", title_style))
        story.append(Spacer(1, 10))
        
        meta_text = """
        <b>Status:</b> System successfully configured and operational.<br/>
        <b>Model Architecture:</b> Deep Learning Stacked LSTM Model.<br/>
        <b>Forecast Horizon:</b> 7 Days Autoregressive Recursive Prediction.<br/>
        <b>Report Status:</b> Generated Successfully.<br/>
        """
        story.append(Paragraph(meta_text, meta_style))
        story.append(Spacer(1, 25)) 
        
        # ২. ফোরকাস্ট টেবিল হেডার
        story.append(Paragraph("<b>📊 Next 7 Days Price Forecast:</b>", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        # টেবিল ডাটা স্ট্রাকচার
        table_data = [["📅 Horizon", "💰 Predicted Price"]]
        
        if latest_forecast:
            for item in latest_forecast:
                table_data.append([item['day'], f"${item['price']:.2f}"])
        else:
            table_data.append(["No Data", "দয়া করে আগে ফাইল আপলোড করে Predict করুন।"])
            
        # টেবিল স্টাইলিং ও কালার প্যালেট
        forecast_table = Table(table_data, colWidths=[200, 200])
        forecast_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#2C3E50')),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8F9FA'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
        ]))
        
        story.append(forecast_table)
        
        # পিডিএফ কম্পাইল ও রিটার্ন
        doc.build(story)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name="Gold_Price_Forecast_Report.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        return f"পিডিএফ তৈরি করতে সমস্যা হয়েছে: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=8080)