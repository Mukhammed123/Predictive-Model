# Sample model
# Super indicator; EMA-short, EMA-long, MACD, RSI combined

# back-propagation neural network (BPNN) input variable;
'''A better
model than the old learning model is constructed using only the
technical analysis and a new research model to explore the market
logic and knowledge rules'''

# Use MACD as input for BPNN

# Add all these techniques/predictions, give them weights and sum them up to see if you should buy the stock or not.
import math, sklearn
from datetime import date, timedelta, datetime
import numpy as np
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

MACD = []
signal_line = []
SMA = []
EMA_12 = []
SMA_26 = []
EMA_26 = []


def make_MACD(given_data):
    # 12 !!!!
    av = 0
    for i in range(11):
        SMA.append(0)
        EMA_12.append(0)
    EMA_12.append(0)
    for i in range(11,len(given_data)):
        for j in range(12):
            av += given_data[i-j]
        av /= 12
        SMA.append(av)
        av = 0

    smoothing = 2
    # SMA is yesterday for the first EMA calculation
    ema_12 = (given_data[12]-SMA[11])*(smoothing/(1+12)) + SMA[11]
    EMA_12.append(ema_12)

    for i in range(12,len(given_data)-1):
        ema_12 = (given_data[i+1]-EMA_12[i])*(smoothing/(1+12)) + EMA_12[i]
        EMA_12.append(ema_12)


    # 26 !!!!
    av = 0
    for i in range(25):
        SMA_26.append(0)
        EMA_26.append(0)
    EMA_26.append(0)
    for i in range(25,len(given_data)):
        for j in range(26):
            av += given_data[i-j]
        av /= 26
        SMA_26.append(av)
        av = 0

    ema_26 = (given_data[26]-SMA[25])*(smoothing/(1+26)) + SMA[25]
    EMA_26.append(ema_26)

    for i in range(27,len(given_data)):
        ema_26 = (given_data[i]-EMA_26[i-1])*(smoothing/(1+26)) + EMA_26[i-1]
        EMA_26.append(ema_26)


    # MACD !!!!
    sub = 0
    for i in range(len(given_data)):
        sub = EMA_12[i] - EMA_26[i]
        MACD.append(sub)


def make_signal_line(given_data):
    # Signal Line of MACD
    av=0

    for i in range(8):
        signal_line.append(0)

    for i in range(8,len(given_data)):
        for j in range(9):
            av += given_data[i-j]
        av /= 9
        signal_line.append(av)
        av = 0

#================================================ RUN THE FUNCTION ==========================================================
def run_model(stock_symbol):
    symbols = web.data.get_nasdaq_symbols()

    # Get stock quotes
    print(stock_symbol)
    print(symbols["Security Name"][stock_symbol])

    yesterday_min1 = date.today() - timedelta(days=2)
    df = web.DataReader(stock_symbol, data_source='yahoo', start='2019-01-01', end=yesterday_min1)


    # ============================================== RSI Calculation =====================================================

    '''
    RSI = 100 – 100 / ( 1 + RS )
    RS = Relative Strength = AvgU / AvgD
    AvgU = average of all up moves in the last N price bars
    AvgD = average of all down moves in the last N price bars
    N = the period of RSI
    There are 3 different commonly used methods for the exact calculation of AvgU and AvgD
    '''

    '''
    First, calculate the bar-to-bar changes for each bar: Chng = Closet – Closet-1

    For each bar, up move (U) equals:

    Closet – Closet-1 if the price change is positive
    Zero if the price change is negative or zero
    Down move (D) equals:

    The absolute value of Closet – Closet-1 if the price change is negative
    Zero if the price change is positive or zero
    '''
    upper_changes = []
    lower_changes = []
    temp = 0
    for i in range(1,len(df["Close"])):
        temp = df["Close"][i] - df["Close"][i-1]
        if temp > 0:
            upper_changes.append(temp)
            lower_changes.append(0)
        elif temp < 0:
            upper_changes.append(0)
            lower_changes.append(-1*temp)
        else:
            upper_changes.append(0)
            lower_changes.append(0)

    av_upper = []
    av_lower = []
    av = 0
    av2 = 0

    for j in range(10):
        av += upper_changes[j]
        av2 += lower_changes[j]
    av/=10
    av2/=10
    av_upper.append(av)
    av_lower.append(av2)

    a = 1/(10)

    for i in range(10,len(upper_changes)):
        av_upper.append((a*upper_changes[i]) + ((1-a)*av_upper[i-10]))
        av_lower.append((a*lower_changes[i]) + ((1-a)*av_lower[i-10]))

    # Calculate Relative Strength (RS)
    RS = []

    for i in range(len(av_lower)):
        RS.append(av_upper[i]/av_lower[i])

    RSI = []
    for i in range(10):
        RSI.append(0)
    temp = 0
    for i in range(len(RS)):
        temp = 100 - (100/(1+RS[i]))
        RSI.append(temp)

    df["RSI"] = RSI

    RSI_70 = []
    RSI_30 = []

    for i in range(len(df["Close"])):
        RSI_70.append(70)
        RSI_30.append(30)


    df["RSI_70"] = RSI_70
    df["RSI_30"] = RSI_30


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
    yesterday_min1 = date.today() - timedelta(days=2)
    yesterday = date.today() - timedelta(days=1)

    amazon_quote = web.DataReader(stock_symbol, data_source='yahoo', start='2019-01-01', end=yesterday_min1)
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

    amazon_quote2 = web.DataReader(stock_symbol, data_source='yahoo', start='2019-01-01', end=yesterday_min1)

    print("Real price: ", amazon_quote2['Close'][-1])

    # Predict tomorrow's close price
    amazon_quote = web.DataReader(stock_symbol, data_source='yahoo', start='2019-01-01', end=yesterday_min1)
    new_df = amazon_quote.filter(['Close'])
    last_60_days = new_df[-60:].values

    future_week = []
    for i in range(9):
        last_60_days_scaled = scaler.transform(last_60_days)

        x_test_last = []
        x_test_last.append(last_60_days_scaled)
        x_test_last = np.array(x_test_last)
        x_test_last = np.reshape(x_test_last, (x_test_last.shape[0], x_test_last.shape[1], 1))
        pred_price = model.predict(x_test_last)

        pred_price = scaler.inverse_transform(pred_price)
        print("Predicted price: ", pred_price)

        last_60_days = last_60_days[1:60,:]
        last_60_days = np.append(last_60_days,[pred_price])
        last_60_days = np.reshape(last_60_days,(60,1))

        future_week.append(pred_price)

    future_week = np.array(future_week)
    print(future_week.shape)
    future_week = np.reshape(future_week,(9))


    # Visualize test price predictions
    plt.figure(figsize=(16, 8))
    plt.title('Model price prediction (train)')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'][34:])
    plt.plot(valid['Close'])
    plt.plot(valid['Predictions'])
    plt.legend(['Train', 'Val', 'Predictions'])
    #     plt.show()

    # =============================== MACD & EMA_12 & EMA_26 calculation =================================================
    global MACD, EMA_12, EMA_26, signal_line

    make_MACD(df["Close"])

    df["MACD"] = np.array(MACD)

    make_signal_line(df["MACD"])

    df["EMA_12"] = EMA_12
    df["EMA_26"] = EMA_26
    df["signal_line"] = signal_line



    # ====================================== MACD & Signal Line prediction ================================================

    macd_data = df.filter(["MACD"])[34:]
    # print(df["MACD"][28:])

    # training data length
    macd_dataset = macd_data.values
    macd_dataset = macd_dataset


    macd_train_len = math.ceil(len(macd_dataset)*0.8)
    # macd values scaler
    macd_scaler = MinMaxScaler(feature_range=(0,1))
    macd_scaled_data = macd_scaler.fit_transform(macd_dataset)


    # create scaled training dataset
    macd_train_data = macd_scaled_data[:macd_train_len,:]


    macd_x_train = []
    macd_y_train = []

    for i in range(60,macd_train_len):
        macd_x_train.append(macd_train_data[i-60:i,0])
        macd_y_train.append(macd_train_data[i,0])


    macd_x_train, macd_y_train = np.array(macd_x_train), np.array(macd_y_train)

    macd_x_train = np.reshape(macd_x_train,(macd_train_len-60,60,1))

    # Build LSTM model
    macd_model = Sequential()
    macd_model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    macd_model.add(LSTM(50, return_sequences=False))
    macd_model.add(Dense(25))
    macd_model.add(Dense(1))

    # Compile the model
    macd_model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    macd_model.fit(macd_x_train, macd_y_train, batch_size=1, epochs=1)


    # Create the testing dataset
    # Create a new array containing scaled values from index 1054 to 1317
    macd_test_data = macd_scaled_data[macd_train_len - 60:, :]

    # Create x_test and y_test datasets
    macd_x_test = []
    macd_y_test = macd_dataset[macd_train_len:, :]

    for i in range(60, len(macd_test_data)):
        macd_x_test.append(macd_test_data[i - 60:i, 0])

    # Convert x_test to numpy array
    macd_x_test = np.array(macd_x_test)

    # Reshape x_test
    macd_x_test = np.reshape(macd_x_test, (macd_x_test.shape[0], macd_x_test.shape[1], 1))

    # Get the models predicted price values
    macd_predictions = macd_model.predict(macd_x_test)
    macd_predictions = macd_scaler.inverse_transform(macd_predictions)

    # Calculate RMSE.... (later)

    # Troublesome part
    macd_train = macd_data[:macd_train_len]
    macd_valid = macd_data[macd_train_len:]
    macd_valid['Predictions'] = macd_predictions

    # Visualize MACD test prediction
    plt.figure(figsize=(16, 8))
    plt.title('MACD Prediction (train)')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('MACD values', fontsize=18)
    plt.plot(macd_train)
    plt.plot(df['signal_line'][34:])
    plt.plot(macd_valid['Predictions'])
    plt.plot(macd_valid["MACD"])
    plt.legend(['Train','Signal Line','Predictions','Val'])
    #     plt.show()

    # Test the model with the current real data


    # Test the model with the current real data
    #
    # yesterday_min1 = date.today() - timedelta(days=2)
    # yesterday = date.today() - timedelta(days=1)

    new_df = df.filter(['MACD'])
    last_60_days = new_df[-60:].values

    future_week_macd = []
    for i in range(9):
        last_60_days_scaled = macd_scaler.transform(last_60_days)

        x_test_last = []
        x_test_last.append(last_60_days_scaled)
        x_test_last = np.array(x_test_last)
        x_test_last = np.reshape(x_test_last, (x_test_last.shape[0], x_test_last.shape[1], 1))
        pred_macd = macd_model.predict(x_test_last)

        pred_macd = macd_scaler.inverse_transform(pred_macd)
        print("Predicted macd: ", pred_macd)

        last_60_days = last_60_days[1:60,:]
        last_60_days = np.append(last_60_days,[pred_macd])
        last_60_days = np.reshape(last_60_days,(60,1))

        future_week_macd.append(pred_macd)

    future_week_macd = np.array(future_week_macd)
    print(future_week_macd.shape)
    future_week_macd = np.reshape(future_week_macd,(9))

    for i in range(9):
        a = df["Close"].keys()[-1]
        if type(a) == str:
            a = datetime.strptime(a,'%Y-%m-%d %H:%M:%S')
        day = a + timedelta(days=1)
        df["Close"][str(day)] = future_week[i]
        df["MACD"][str(day)] = future_week_macd[i]

    signal_line.clear()
    make_signal_line(df["MACD"])
    for i in range(8,-1):
        day = yesterday + timedelta(days=i+1)
        df["signal_line"][str(day) + ' 00:00:00'] = signal_line[len(signal_line) - i]

    # Visualize MACD prediction for a week
    plt.figure(figsize=(16, 8))
    plt.title('MACD Prediction (MACD history based)')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(df["MACD"][34:])
    plt.plot(df["MACD"][34:-9])
    plt.plot(df['signal_line'][34:])
    plt.legend(['Next Week(pred_price based) MACD','MACD','Signal Line'])
    # plt.show()



    # Visualize price prediction for a week
    plt.figure(figsize=(16, 8))
    plt.title("Close Price Prediction")
    plt.plot(df['Close'][34:],'b-')
    plt.plot(df['Close'][34:-9],'g-')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.legend(['Price Prediction for a week','Price'])

    plt.figure(figsize=(16, 8))
    plt.title('RSI ' + stock_symbol)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(df["RSI"][34:])
    plt.plot(df["RSI_70"][34:])
    plt.plot(df["RSI_30"][34:])
    # plt.plot(copy["MACD"])
    plt.legend(['RSI','RSI_70','RSI_30'])
    plt.show()


    print("Done")
# EMA 26 zeros + signal line 8 zeros = 34 zeros, df graph should start from 34th index
