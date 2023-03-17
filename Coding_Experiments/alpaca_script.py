import xgboost as xgb
import alpaca_trade_api as tradeapi
from finta import TA
import pickle
import os
import sys
import time
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Set Alpaca API key and secret
alpaca_api_key = os.getenv('ALPACA_API_KEY')
alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')

# Import saved XGB model
papertrading_model = xgb.XGBClassifier()
papertrading_model.load_model('xgb_clf.bst')

# Load scaler model
scaler = pickle.load(open('scaler_model.pkl', 'rb'))

# Define list of features
feats = ['HMA_5', 'RSI_5', 'ATR_14', 'RSI_14', 'RSI_150', 'cci']

# Initialize Alpaca API
alpaca_api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, 'https://paper-api.alpaca.markets')

# Check if the API is running
try:
    account_info = alpaca_api.get_account()
    print(f"Alpaca REST API is running! Status is {account_info.status} for account #{account_info.account_number} ")
except Exception as e:
    print("Error getting account info:", e)

while True:
    # Get latest data on 1HR timeframe
    data_trade = alpaca_api.get_crypto_bars(['BTC/USDT'], tradeapi.TimeFrame.Hour, "2023-01-01").df

    # Compute the features needed for the prediction
    for timeframe in [5, 14, 150]:
        data_trade['RSI_'+str(timeframe)] = TA.RSI(data_trade, timeframe)
    data_trade['ATR_14'] = TA.ATR(data_trade, 14)
    data_trade['HMA_5'] = TA.HMA(data_trade, 5)
    data_trade['cci'] = TA.CCI(data_trade)

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
    quantity_to_trade = usd_balance / eth_usd_price * 0.1

    # Define order parameters
    symbol = 'ETHUSD'
    order_type = 'limit'
    limit_price = eth_usd_price  # limit price for the order
    time_in_force = 'gtc'
    qty = quantity_to_trade  # quantity of ETH to trade

    # check if BUY value changed from previous time
    if 'previous_buy' not in locals():
        previous_buy = BUY

    if BUY != previous_buy:
        if BUY:
            side = 'buy'
            limit_price += 0.0001
        else:
            side = 'sell'
            qty = float(alpaca_api.get_position(symbol).qty)  # quantity of ETH to sell (same as the current position)
            limit_price -= 0.0001

        # Place order
        try:
            order = alpaca_api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price
            )
            print(f"{side.capitalize()} order for {qty} {symbol} at {limit_price} submitted successfully.")

        except Exception as e:
            print(f"Error submitting {side.capitalize()} order: ", e)

        # Update previous_buy value
        previous_buy = BUY
    else :
        print('.', end='')

    # Redirect output to log file
    with open('alpaca_script.log', 'a') as f:
        sys.stdout = f
        sys.stderr = f

    # Wait 1 hour then check again
    time.sleep(3600)



