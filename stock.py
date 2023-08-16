from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from io import BytesIO
import base64

app = Flask(__name__)
model=pickle.load(open("stock_price.pkl","rb"))
@app.route('/',methods=['GET'])
def Home():
    return render_template('stock.html')




@app.route("/", methods=["POST"])
def stock_prediction():
    if request.method == "POST":
        start_date = request.form["start"]
        end_date = request.form["end"]
        stock_symbol = request.form["symbol"]

        file = yf.download(stock_symbol, start=start_date, end=end_date)
        close = pd.DataFrame(file["Close"])
        length = int(len(close) * 0.9)

        training_data = list(close[:length]["Close"])
        testing_data = list(close[length:]["Close"])

        n_test = len(testing_data)
        model_predictions = []

        for i in range(n_test):
            model = sm.tsa.ARIMA(training_data, order=(4, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            model_predictions.append(yhat)
            actual = testing_data[i]
            training_data.append(actual)

        mse = np.mean(np.abs(np.array(model_predictions) - np.array(testing_data)) / np.abs(testing_data))
        accuracy = 100 - mse

        return render_template("stock.html", accuracy=accuracy)

    return render_template("stock.html")

if __name__ == "__main__":
    app.run(debug=True)
