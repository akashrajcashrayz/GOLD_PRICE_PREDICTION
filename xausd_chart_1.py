import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import time
from tensorflow.keras.models import load_model
import joblib

import requests
import csv
from datetime import datetime
st.set_page_config(
    page_title="Real-Time XAU-USD Dashboard",
    page_icon="✅",
    layout="wide",
)
st.title("Real-Time XAU-USD Price Candlestick Chart with KPIs")

left_column, right_column = st.columns([2, 3])
#left_column.title("Real-Time XAU-USD KPIs and News")

kpi_table = left_column.empty()
news_table = left_column.empty()

#right_column.title("Real-Time XAU-USD Price Chart")
graph_placeholder = right_column.empty()
# Set the stock ticker symbol for Yahoo Finance
stock_symbol = "XAU-USD"


# Function to fetch live stock data
def create_sequences(data, seq_length):
    sequences = []
    for i in range(seq_length, len(data)):
        sequences.append(data[i - seq_length:i, :])
    return np.array(sequences)


def predict_next_close(new_data_1):
    new_data = new_data_1[['Open', 'High', 'Low', 'Close']].pct_change()
    sequence_length = 15
    # Load the model
    model = load_model('next_close_prediction_model.h5')

    # Load the scaler object
    scaler = joblib.load('scaler.pkl')

    # Assume new_data is your new DataFrame with 'Open', 'High', 'Low', 'Close' columns
    new_values = new_data[['Open', 'High', 'Low', 'Close']].values
    new_values_1 = new_data_1[['Open', 'High', 'Low', 'Close']].values

    # Scale the new data
    scaled_new_data = scaler.transform(new_values)

    # Create sequences for the new data
    new_sequences = create_sequences(scaled_new_data, sequence_length)

    # Extract the last sequence as the input for prediction
    new_input = new_sequences[-1, :-1].reshape(1, sequence_length - 1, 4)

    # Perform prediction
    predicted_scaled_change = model.predict(new_input)

    # Inverse transform the predicted values to get percentage change
    predicted_change = scaler.inverse_transform(predicted_scaled_change)[0][-1]

    # Fetch the last 'Close' price from your data
    last_close_price = new_values_1[-1][-1]

    # Apply the predicted percentage change to get the predicted next 'Close' price
    predicted_close = last_close_price * (1 + predicted_change)

    return predicted_close


def give_minute_data():
    url = 'https://www.goldapi.io/api/XAU/USD'
    headers = {
        'x-access-token': 'goldapi-2zx4418lpl6rk8b-io'
    }

    csv_filename = 'gold_prices.csv'

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        price = data.get('price')
        timestamp = data.get('timestamp')

        # Convert epoch timestamp to a readable date/time format
        readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([readable_time, price])

    df = pd.read_csv(csv_filename, names=['Timestamp', 'Price'])  # Read the CSV file
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # Convert 'Timestamp' to datetime

    # Filter data for today's date
    today_date = datetime.now().date()
    today_data = df[df['Timestamp'].dt.date == today_date]

    # Set 'Timestamp' as the DataFrame index
    today_data.set_index('Timestamp', inplace=True)

    # Resample data into minute candlesticks
    minute_candlesticks = today_data['Price'].resample('1Min').ohlc()

    # Create a new DataFrame including Timestamp and OHLC data
    stock_data_with_timestamp = pd.DataFrame({
        'Timestamp': minute_candlesticks.index,  # Include the Timestamp
        'Open': minute_candlesticks['open'],
        'High': minute_candlesticks['high'],
        'Low': minute_candlesticks['low'],
        'Close': minute_candlesticks['close']
    })

    return stock_data_with_timestamp


def get_xausd_news():
    url = "https://forexnewsapi.com/api/v1"
    params = {
        "currencypair": "XAU-USD",
        "items": 5,
        "token": "ap9uqgnowl9ogx6tnt2kqwomzhiahpy3n72o7xhs"
    }

    response = requests.get(url, params=params)
    print(response)

    if response.status_code == 200:
        data = response.json()
        data_dataframe =  pd.DataFrame(data['data'])

        return data_dataframe[['date','title','sentiment']]

# Create an empty figure
fig = go.Figure()

# Create a placeholder container
placeholder = st.empty()

# Loop to continuously update the candlestick chart with real-time stock data
last_time = 0
news = 'no_news'
sentiment = 'neutral'
while True:
    # Fetch live stock data
    stock_data = give_minute_data().dropna()
    if stock_data.index[-1] != last_time:
        last_time = stock_data.index[-1]
        news_df = get_xausd_news()

        pred_value = 0
        if len(stock_data) >= 15:
            pred_value = predict_next_close(stock_data)

    # Update the candlestick chart
    trace = go.Candlestick(x=stock_data.index,
                           open=stock_data['Open'],
                           high=stock_data['High'],
                           low=stock_data['Low'],
                           close=stock_data['Close'],
                           name=stock_symbol)

    fig = go.Figure(data=[trace])

    # Update the layout of the figure
    fig.update_layout(title=f'{stock_symbol} Real-Time XAU-USD Price',
                      xaxis_title='Time',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)

    # Display the plot using Streamlit
    with graph_placeholder:
        st.plotly_chart(fig, use_container_width=True)

    # Create a table for KPIs and predicted value
    kpi_df = pd.DataFrame({
        'Open': [str(stock_data['Open'].iloc[-1])],
        'High': [str(stock_data['High'].iloc[-1])],
        'Low': [str(stock_data['Low'].iloc[-1])],
        'Close': [str(stock_data['Close'].iloc[-1])],
        'Predicted Value': [str(round(pred_value, 2))]
    })

    # Create a table for news and sentiment


    # Display the tables for KPIs and news
    with kpi_table:
        st.write("Real-Time XAU-USD KPIs and Predicted Value")
        st.table(kpi_df)

    with news_table:
        st.write("News and Sentiment")
        st.table(news_df)

    time.sleep(1)