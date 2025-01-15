from autoencoder import *
from dataGetter import *
from tradingBot import TradingBot


bot = TradingBot("./marketData/XRPUSDT-5m-2020-23", "300-100-50-5_4.33e-6")

candles = load_data("./marketData/XRPUSDT-5m-2024")
bot.predict(candles)
