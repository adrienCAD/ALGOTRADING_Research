# Import Libraries
import os
import sys
import time
import logging
import pickle
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from finta import TA


# Load environment variables: Alpaca Keys and Path
load_dotenv()
alpaca_api_key = os.getenv('ALPACA_API_KEY')
alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
model_path = os.getenv('ALPHAJET_MODEL_PATH')

# Define Logging preferences
log_filename = model_path + 'logfile.log'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_filename)]
)

# Create a class for logging immediately the outcomes of the code execution
class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

# Create a class EthTrader
# This class will get the latest ETHUSD candlestick, prepare and pre-process the data, predict the ETHUSD movemtn, handle the buy and sell orders, and exectute the trades
class EthTrader:
    def __init__(self, api_key, secret_key, base_url):
        self.alpaca_api = tradeapi.REST(api_key, secret_key, base_url)
        self.model = pickle.load(open(model_path + 'lda_classifier.pkl', 'rb'))
        self.scaler = pickle.load(open(model_path + 'scaler_model.pkl', 'rb'))
        self.fee_percentage = 0.0025
        self.feats = ['SMA_5', 'RSI_5', 'VAMA_7', 'RSI_14', 'SMA_20', 'RSI_20', 'SMA_50',
                      'HMA_50', 'ATR_50', 'EMA_100', 'VAMA_100', 'ATR_100', 'SMA_150',
                      'ATR_200', 'BB_MED']

    @staticmethod
    def print_flush(message):
        logging.info(message)

    def get_crypto_bars(self):
        end_date = datetime.now()
        end_date_str = end_date.strftime('%Y-%m-%d')
        return self.alpaca_api.get_crypto_bars(['ETHUSD'], tradeapi.TimeFrame.Hour, start="2023-03-01", end=end_date_str).df



    def prepare_data(self):
        data_trade = self.alpaca_api.get_crypto_bars(['ETH/USDT'], tradeapi.TimeFrame.Hour, "2023-03-01").df

        for timeframe in [5, 20, 50, 150]:
            data_trade['SMA_' + str(timeframe)] = TA.SMA(data_trade, timeframe)
        for timeframe in [5, 14, 20]:
            data_trade['RSI_' + str(timeframe)] = TA.RSI(data_trade, timeframe)
        for timeframe in [7, 100]:
            data_trade['VAMA_' + str(timeframe)] = TA.VAMA(data_trade, timeframe)
        for timeframe in [50, 100, 200]:
            data_trade['ATR_' + str(timeframe)] = TA.ATR(data_trade, timeframe)

        data_trade[['BB_UPPER', 'BB_MED', 'BB_LOWER']] = TA.BBANDS(data_trade)
        data_trade['EMA_100'] = TA.EMA(data_trade, 100)
        data_trade['HMA_50'] = TA.HMA(data_trade, 50)

        return data_trade

    def preprocess_data(self, data_trade):
        return self.scaler.transform(data_trade[self.feats].dropna())

    def get_eth_usd_price(self):
        return self.alpaca_api.get_latest_crypto_orderbook(['ETH/USD'])['ETH/USD'].asks[0].p

    def handle_buy(self, eth_usd_price):
        available_usd_cash = float(self.alpaca_api.get_account().cash)
        order = self.pass_buy_order(available_usd_cash, eth_usd_price)

        if order:
            self.wait_for_order_execution(order)
        else:
            self.do_nothing(eth_usd_price, buy=True)

    def handle_sell(self, eth_usd_price):
        symbol = 'ETHUSD'
        eth_qty = float(self.alpaca_api.get_position(symbol).qty)

        if eth_qty * eth_usd_price >= 50:
            qty_to_sell = eth_qty / 3 * (1 - 0.01)
            limit_price = eth_usd_price - 0.0001

            try:
                order = self.alpaca_api.submit_order(
                    symbol=symbol,
                    qty=qty_to_sell,
                    side='sell',
                    type='limit',
                    time_in_force='gtc',
                    limit_price=limit_price
                )
                self.print_flush(f"Sell order for {qty_to_sell:.4f} {symbol} at {limit_price:.4f} submitted successfully.")
                self.wait_for_order_execution(order)
            except Exception as e:
                self.print_flush(f"Error submitting SELL order: {e}")
        else:
            self.do_nothing(eth_usd_price, buy=False)

    def pass_buy_order(self, available_usd_cash, eth_usd_price):
        symbol = 'ETHUSD'
        order_type = 'limit'
        time_in_force = 'gtc'

        if available_usd_cash >= 50:
            adjusted_amount = available_usd_cash * (1 - self.fee_percentage)
            adjusted_qty = round(adjusted_amount / eth_usd_price, 4)

            while True:
                try:
                    limit_price = eth_usd_price + 0.0001
                    order = self.alpaca_api.submit_order(
                        symbol=symbol,
                        qty=adjusted_qty,
                        side='buy',
                        type=order_type,
                        time_in_force=time_in_force,
                        limit_price=limit_price
                    )
                    self.print_flush(f"Buy order for {adjusted_qty:.4f} {symbol} at {limit_price:.4f} submitted successfully.")
                    return order
                except Exception as e:
                    error_message = str(e)
                    if 'insufficient balance' in error_message.lower():
                        adjusted_amount *= 0.99
                        adjusted_qty = round(adjusted_amount / eth_usd_price, 4)
                    else:
                        self.print_flush(f"Error submitting Buy order: {e}")
                        return None
        else:
            return None

    def wait_for_order_execution(self, order, timeout=90, check_interval=5):
        symbol = order.symbol
        side = order.side
        start_time = time.time()

        while True:
            current_time = time.time()
            if current_time - start_time >= timeout:
                self.print_flush(f"Order execution timed out. Resubmitting the order with an updated price.")
                self.alpaca_api.cancel_order(order.id)

                eth_usd_price = self.get_eth_usd_price()
                new_limit_price = eth_usd_price + 0.0001 if side == 'buy' else eth_usd_price - 0.0001
                try:
                    new_order = self.alpaca_api.submit_order(
                        symbol=order.symbol,
                        qty=order.qty,
                        side=side,
                        type=order.type,
                        time_in_force=order.time_in_force,
                        limit_price=new_limit_price
                    )
                    self.print_flush(f"{side.capitalize()} order for {order.qty} {order.symbol} at {new_limit_price} submitted successfully.")
                    order = new_order
                    start_time = current_time
                except Exception as e:
                    self.print_flush(f"Error resubmitting {side.capitalize()} order: {e}")

            order_status = self.alpaca_api.get_order(order.id).status

            if order_status == 'filled':
                available_usd_cash = float(self.alpaca_api.get_account().cash)
                eth_qty = float(self.alpaca_api.get_position(symbol).qty)
                eth_usd_price = self.get_eth_usd_price()
                portfolio_value = available_usd_cash + eth_qty * eth_usd_price
                self.print_flush(f"Order executed successfully | Portfolio Value = {portfolio_value:.4f} USD")
                break
            elif order_status in ('canceled', 'rejected'):
                self.print_flush(f"Order {order_status}.")
                break

            time.sleep(check_interval)

    def do_nothing(self, eth_usd_price, buy):
        symbol = 'ETHUSD'
        available_usd_cash = float(self.alpaca_api.get_account().cash)
        eth_qty = float(self.alpaca_api.get_position(symbol).qty)
        portfolio_value = available_usd_cash + eth_qty * eth_usd_price

        self.print_flush(f"{symbol} = {eth_usd_price:.4f} | ETH owned = {eth_qty:.4f} | Portfolio Value = {portfolio_value:.4f} USD|")
        self.print_flush("HODLing ETH." if buy else "Waiting for right time to buy.")

    def execute_trade(self):
        data_trade = self.get_crypto_bars()
        self.compute_technical_indicators(data_trade)
        data_to_predict = self.preprocess_data(data_trade)
        predictions = self.model.predict(data_to_predict)
        buy = predictions[-1]

        eth_usd_price = self.get_eth_usd_price()
        self.print_flush("↑↑ Prediction = BUY ↑↑ " if buy else "↓↓ Prediction = SELL ↓↓")

        if buy:
            self.handle_buy(eth_usd_price)
        else:
            self.handle_sell(eth_usd_price)

if __name__ == "__main__":
    base_url = 'https://paper-api.alpaca.markets'  # Add your desired base URL
    trader = EthTrader(alpaca_api_key, alpaca_secret_key, base_url)
    trader.execute_trade()