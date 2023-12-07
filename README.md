# ALGOTRADING_Research

## To-Do:
### ML Model :
- try an three-gate approach using clustering or simple cutoffs to define strong sell / sell / hold / buy/ strong buy, then run a ML multi-class classification on top of it.
    - find a way to fix the class imbalance problem with time series for the ML training
- possibly use fear and greed index for sentiment analysis
- save 1st models to JSON

### REST-API:
- find the easiest way to create a REST API connecting to Alpaca


## Plan:

- use a web framework like Flask or FastAPI to build a REST API that connects to the Alpaca API. This will allow to get the new prices and send buy/sell orders.

- use the Alpaca API to retrieve the latest OHLCV data for ETHUSD every hour.  get_last_trade function from the alpaca_trade_api module to get the latest trade data.

- load pre-trained CatBoost model from the JSON file and use it to predict the buy/sell signal based on the latest OHLCV data.

- use the submit_order function from the alpaca_trade_api module to place a buy or sell order on Alpaca papertrading based on the predicted signal.
    -   set the order parameters such as quantity, limit price
    -   stop loss price as needed.

- use the papertrading results to calculate the Sortino and Sharpe ratios to measure the performance of the trading strategy.
