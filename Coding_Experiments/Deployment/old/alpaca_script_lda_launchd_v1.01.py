import alpaca_trade_api as tradeapi
from finta import TA
import pickle
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Set Alpaca API key and secret
alpaca_api_key = os.getenv('ALPACA_API_KEY')
alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
model_path = os.getenv('ALPHAJET_MODEL_PATH')

# Import saved LDA model
#model_path = "/Users/adriencaudron/BOOTCAMP/Playground/AlphaJet2.0/ALGOTRADING_Research/Coding_Experiments/Deployment/"
papertrading_model = pickle.load(open(model_path + 'lda_classifier.pkl', 'rb'))

# Load scaler model
scaler = pickle.load(open(model_path + 'scaler_model.pkl', 'rb'))

# Define list of features
feats = ['SMA_5', 'RSI_5', 'VAMA_7', 'RSI_14', 'SMA_20', 'RSI_20', 'SMA_50',
       'HMA_50', 'ATR_50', 'EMA_100', 'VAMA_100', 'ATR_100', 'SMA_150',
       'ATR_200', 'BB_MED']

# Initialize Alpaca API
alpaca_api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, 'https://paper-api.alpaca.markets')

# Check the time
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Check if the API is running
try:
    account_info = alpaca_api.get_account()
    print(f"{current_time} - Alpaca REST API {account_info.status} - account #{account_info.account_number}")
    
except Exception as e:
    print(f"{current_time} - Error getting account info: {e}")

# Initialize previous_buy variable to store the previous BUY state
previous_buy = None

def execute_trade():
    # Declare 'previous_buy' as a global variable to update it within the function
    global previous_buy

    # Get latest data on 1HR timeframe
    data_trade = alpaca_api.get_crypto_bars(['ETH/USDT'], tradeapi.TimeFrame.Hour, "2023-03-01").df

    # Compute the SMA, RSI, VAMA, ATR technical features needed for the prediction
    for timeframe in [5, 20, 50, 150]:
        data_trade['SMA_'+str(timeframe)] = TA.SMA(data_trade, timeframe)
    for timeframe in [5, 14, 20]:
        data_trade['RSI_'+str(timeframe)] = TA.RSI(data_trade, timeframe)
    for timeframe in [7, 100]:
        data_trade['VAMA_'+str(timeframe)] = TA.VAMA(data_trade, timeframe)
    for timeframe in [50, 100, 200]:
        data_trade['ATR_'+str(timeframe)] = TA.ATR(data_trade, timeframe)

    # Adding Bollinger Bands, EMA_100 and HMA_50
    data_trade[['BB_UPPER','BB_MED','BB_LOWER']] =TA.BBANDS(data_trade)
    data_trade['EMA_100'] = TA.EMA(data_trade, 100)
    data_trade['HMA_50'] = TA.HMA(data_trade, 50)

    # Transform preprocessed data to numpy array
    data_to_predict = scaler.transform(data_trade[feats].dropna())

    # Make predictions using the model
    predictions = papertrading_model.predict(data_to_predict)

    # Check the trade direction: BUY (=1) or SELL (=0)
    BUY = predictions[-1]

    # Get the current price of BTC in USD
    eth_usd_price = alpaca_api.get_latest_crypto_orderbook(['ETH/USD'])['ETH/USD'].asks[0].p

    # Calculate the quantity of BTC to buy or sell
    usd_balance = float(alpaca_api.get_account().cash)
    quantity_to_trade = usd_balance / eth_usd_price

    # Define order parameters
    symbol = 'ETHUSD'
    order_type = 'limit'
    limit_price = eth_usd_price  # limit price for the order
    time_in_force = 'gtc'
    qty = quantity_to_trade  # quantity of ETH to trade

    # If previous_buy is None, it means this is the first run, and we need to set its value to the current BUY state
    if previous_buy is None:
        previous_buy = 0 #BUY using 1 for testing

    print("Prediction: BUY - " if BUY else "Prediction: SELL - ", end='')
    
    if BUY != previous_buy:
        if BUY:
            side = 'buy'
            limit_price += 0.0001
            # Place a buy order for all available USD capital
            try:
                order = alpaca_api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force,
                    limit_price=limit_price
                )
                print(f"{side.capitalize()} order for {qty:.4f} {symbol} at {limit_price:.4f} submitted successfully.")
            except Exception as e:
                print(f"Error submitting {side.capitalize()} order: ", e)

        else:
            side = 'sell'
            eth_qty = float(alpaca_api.get_position(symbol).qty)
            if eth_qty * eth_usd_price >= 10:
                qty_to_sell = eth_qty / 3
                limit_price -= 0.0001

                try:
                    order = alpaca_api.submit_order(
                        symbol=symbol,
                        qty=qty_to_sell,
                        side=side,
                        type=order_type,
                        time_in_force=time_in_force,
                        limit_price=limit_price
                    )
                    print(f"{side.capitalize()} order for {qty_to_sell:.4f} {symbol} at {limit_price:.4f} submitted successfully.")
                except Exception as e:
                    print(f"Error submitting {side.capitalize()} order: ", e)

        # Update previous_buy value
        previous_buy = BUY
    else:
        print('.', end='')

if __name__ == "__main__":
    execute_trade()