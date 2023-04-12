####################### IMPORTS #######################
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
papertrading_model = pickle.load(open(model_path + 'lda_classifier.pkl', 'rb'))

# Load scaler model
scaler = pickle.load(open(model_path + 'scaler_model.pkl', 'rb'))

####################### FUNCTIONS ##########################
def wait_for_order_execution(alpaca_api, order, symbol, timeout=180, check_interval=10):
    start_time = time.time()
    side = order.side
    
    while True:
        current_time = time.time()
        if current_time - start_time >= timeout:
            # printing the timeout message, using the flush option to avoid buffering and displaying the message immediately on the console output
            print(f"{displayed_time} - Order execution timed out. Resubmitting the order with an updated price.", flush=True)

            # Cancel the previous order
            alpaca_api.cancel_order(order.id)

            # Update the limit price
            eth_usd_price = alpaca_api.get_latest_crypto_orderbook(['ETH/USD'])['ETH/USD'].asks[0].p
            if side == 'buy':
                new_limit_price = eth_usd_price + 0.0001
            else:  # 'sell'
                new_limit_price = eth_usd_price - 0.0001

            # Resubmit the order with the new price
            try:
                new_order = alpaca_api.submit_order(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    type=order.type,
                    time_in_force=order.time_in_force,
                    limit_price=new_limit_price
                )
                print(f"{displayed_time} - {side.capitalize()} order for {order.qty} {order.symbol} at {new_limit_price} submitted successfully.", flush=True)
                order = new_order
                start_time = current_time
            except Exception as e:
                print(f"{displayed_time} - Error resubmitting {side.capitalize()} order: ", e, flush=True)

        order_status = alpaca_api.get_order(order.id).status

        if order_status == 'filled':
            available_usd_cash = float(alpaca_api.get_account().cash)
            eth_qty = float(alpaca_api.get_position(symbol).qty)
            portfolio_value = available_usd_cash + eth_qty * eth_usd_price
            print(f"{displayed_time} - Order executed successfully | Portfolio Value = {portfolio_value:.4f} USD", flush=True)
            break
        elif order_status in ('canceled', 'rejected'):
            print(f"{displayed_time} - Order {order_status}.", flush=True)
            break

        time.sleep(check_interval)

def do_nothing(alpaca_api, symbol, eth_usd_price, BUY):
    # obtaining portfolio content and value
    available_usd_cash = float(alpaca_api.get_account().cash)
    eth_qty = float(alpaca_api.get_position(symbol).qty)
    portfolio_value = available_usd_cash + eth_qty * eth_usd_price
    
    # displaying portfolio content, value and current status
    print(f"{symbol} = {eth_usd_price:.4f} | ETH owned = {eth_qty:.4f} | Portfolio Value = {portfolio_value:.4f} USD|", end='')
    print("HODLing ETH." if BUY else "Waiting for right time to buy.")


def execute_trade():
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
    quantity_to_trade = usd_balance / eth_usd_price *.95 #using 95% of the avail. balance.

    # Define order parameters
    symbol = 'ETHUSD'
    order_type = 'limit'
    limit_price = eth_usd_price  # limit price for the order
    time_in_force = 'gtc'
    qty = quantity_to_trade  # quantity of ETH to trade

    print("Prediction = BUY | " if BUY else "Prediction = SELL | ", end='')

    # Buy logic 
    if BUY : 
        available_usd_cash = float(alpaca_api.get_account().cash)
        eth_usd_price = alpaca_api.get_latest_crypto_orderbook(['ETH/USD'])['ETH/USD'].asks[0].p

        if available_usd_cash >= 50:
            amount_to_buy = available_usd_cash * (1 - 0.03) # Multiply by (1 - fee_percentage) to account for the 0.25% (TAKER) fee
            qty_to_buy = amount_to_buy / eth_usd_price
            limit_price = eth_usd_price + 0.0001

            try:
                order = alpaca_api.submit_order(
                    symbol=symbol,
                    qty=qty_to_buy,
                    side='buy',
                    type=order_type,
                    time_in_force=time_in_force,
                    limit_price=limit_price
                )
                print(f"Buy order for {qty_to_buy:.4f} {symbol} at {limit_price:.4f} submitted successfully.")
                wait_for_order_execution(alpaca_api, order, symbol)



            except Exception as e:
                print(f"Error submitting Buy order: ", e)

        else: 
            do_nothing(alpaca_api, symbol, eth_usd_price,BUY)       


    # Sell logic
    else : # if BUY !=1, then we sell
        eth_usd_price = alpaca_api.get_latest_crypto_orderbook(['ETH/USD'])['ETH/USD'].asks[0].p
        eth_qty = float(alpaca_api.get_position(symbol).qty)
        
        if eth_qty * eth_usd_price >= 50:
            qty_to_sell = eth_qty / 3 * (1 - 0.01) # Multiply by (1 - fee_percentage) to account for the 0.25% (TAKER) fee
            limit_price = eth_usd_price - 0.0001

            try:
                order = alpaca_api.submit_order(
                    symbol=symbol,
                    qty=qty_to_sell,
                    side='sell',
                    type=order_type,
                    time_in_force=time_in_force,
                    limit_price=limit_price
                )
                print(f"Sell order for {qty_to_sell:.4f} {symbol} at {limit_price:.4f} submitted successfully.")
                wait_for_order_execution(alpaca_api, order, symbol)

            except Exception as e:
                print(f"Error submitting Sell order: ", e)

        else: 
            do_nothing(alpaca_api, symbol, eth_usd_price,BUY)   




####################### MAIN #######################

# Define list of features
feats = ['SMA_5', 'RSI_5', 'VAMA_7', 'RSI_14', 'SMA_20', 'RSI_20', 'SMA_50',
       'HMA_50', 'ATR_50', 'EMA_100', 'VAMA_100', 'ATR_100', 'SMA_150',
       'ATR_200', 'BB_MED']

# Initialize Alpaca API
alpaca_api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, 'https://paper-api.alpaca.markets')

# Check the time
current_time = displayed_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Check if the API is running
try:
    account_info = alpaca_api.get_account()
    print(f"{displayed_time} | #{account_info.account_number} | ", end='')
    
except Exception as e:
    print(f"{displayed_time} - Error getting account info: {e}")

# Get the alpaca script's directory
script_directory = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    execute_trade()