from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import plotly.graph_objects as go
import os
import uuid

app = Flask(__name__)

df = pd.read_csv("D:\College\stock\google.csv")
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df['MA100'] = df['Close'].rolling(window=100).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()
df['Daily Return'] = df['Close'].pct_change()

model = joblib.load("stock.joblib")
scaler = joblib.load("train.pkl")

def predict_future_prices(days):
    last_100_days = df['Close'][-100:].values.reshape(-1, 1)
    last_100_scaled = scaler.transform(last_100_days)

    predictions = []
    for _ in range(days):
        X_input = last_100_scaled.reshape(1, -1)
        pred_scaled = model.predict(X_input)[0]
        pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
        predictions.append(pred_price)
        last_100_scaled = np.append(last_100_scaled, [[pred_scaled]], axis=0)[1:]
    return predictions

def save_plot(fig, name):
    path = f"static/images/{name}.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = []
    pred_plot = ma_plot = return_plot = candle_plot = None
    days = 5

    if request.method == "POST":
        days = int(request.form["days"])
        predictions = predict_future_prices(days)

        
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, days + 1), predictions, marker='o', color='green')
        ax1.set_title("Predicted Prices")
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Price (USD)")
        pred_plot = save_plot(fig1, f"predicted_{uuid.uuid4().hex}")

       
        fig2, ax2 = plt.subplots()
        df[['Close', 'MA100', 'MA200']].plot(ax=ax2)
        ax2.set_title("Moving Averages (MA100 & MA200)")
        ma_plot = save_plot(fig2, f"ma_{uuid.uuid4().hex}")

        
        fig3, ax3 = plt.subplots()
        df['Daily Return'].plot(kind='hist', bins=100, ax=ax3)
        ax3.set_title("Daily Return Distribution")
        return_plot = save_plot(fig3, f"return_{uuid.uuid4().hex}")

        fig = go.Figure(data=[go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])
        fig.update_layout(title='Google Stock Candlestick Chart', xaxis_rangeslider_visible=True)
        candle_path = f"static/images/candle_{uuid.uuid4().hex}.html"
        fig.write_html(candle_path)
        candle_plot = candle_path

    return render_template("index.html",
                           predictions=predictions,
                           days=days,
                           pred_plot=pred_plot,
                           ma_plot=ma_plot,
                           return_plot=return_plot,
                           candle_plot=candle_plot)
