# Sample model
# use MACD... to predict prices in the future
# Currently working on strategy
import math
from datetime import date, timedelta
import numpy as np
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


def run_model(stock_symbol):
    symbols = web.data.get_nasdaq_symbols()

    # Get stock quotes
    print(stock_symbol)
    print(symbols["Security Name"][stock_symbol])

    current_time = date.today()
    current_time = str(current_time)
    df = web.DataReader(stock_symbol, data_source='yahoo', start='2013-01-01', end=current_time)

    # Create a new dataFrame with only close column

    data = df.filter(['Close'])
    # Convert data to a numpy array
    dataset = data.values

    # Get the number of rows to train the model
    train_data_len = math.ceil(len(dataset) * 0.8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training dataset
    # Create the scaled training dataset
    train_data = scaled_data[0:train_data_len, :]

    # Split the data into x_train and y_train datasets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data. The LSTM network expects the input to be three dimensional; in the form of number of samples,
    #  number of time steps and number of  features.
    x_train = np.reshape(x_train, (train_data_len - 60, 60, 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing dataset
    # Create a new array containing scaled values from index 1054 to 1317
    test_data = scaled_data[train_data_len - 60:, :]

    # Create x_test and y_test datasets
    x_test = []
    y_test = dataset[train_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert x_test to numpy array
    x_test = np.array(x_test)

    # Reshape x_test
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate RMSE.... (later)

    train = data[:train_data_len]
    valid = data[train_data_len:]
    valid['Predictions'] = predictions

    # Test the model with the current real data
    #
    yesterday = date.today() - timedelta(days=1)

    amazon_quote = web.DataReader('AMZN', data_source='yahoo', start='2013-01-01', end=yesterday)
    new_df = amazon_quote.filter(['Close'])
    last_60_days = new_df[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)

    x_test_last = []
    x_test_last.append(last_60_days_scaled)
    x_test_last = np.array(x_test_last)
    x_test_last = np.reshape(x_test_last, (x_test_last.shape[0], x_test_last.shape[1], 1))
    pred_price = model.predict(x_test_last)

    pred_price = scaler.inverse_transform(pred_price)
    print("Predicted price: ", pred_price)

    # REal price

    amazon_quote2 = web.DataReader('AMZN', data_source='yahoo', start=current_time, end=current_time)

    print("Real price: ", amazon_quote2['Close'])

    # Predict tomorrow's close price
    amazon_quote = web.DataReader('AMZN', data_source='yahoo', start='2020-01-01', end=current_time)
    new_df = amazon_quote.filter(['Close'])
    last_60_days = new_df[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)

    x_test_last = []
    x_test_last.append(last_60_days_scaled)
    x_test_last = np.array(x_test_last)
    x_test_last = np.reshape(x_test_last, (x_test_last.shape[0], x_test_last.shape[1], 1))
    pred_price = model.predict(x_test_last)

    pred_price = scaler.inverse_transform(pred_price)
    print("Predicted price for tomorrow: ", pred_price)

    # Visualize...
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'])
    plt.show()


''' 

# Visualize closing price history
    plt.figure(figsize=(16, 8))
    plt.title("Close Price History")
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.show()


'''
