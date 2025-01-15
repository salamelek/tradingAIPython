from dataGetter import *
from tradingBot import TradingBot

# TODO re-train the autoencoder, since i added tanh to the normalisation of candles

bot = TradingBot(
    "./marketData/XRPUSDT-5m-2020-23",
    "300-100-50-5_4.33e-6",
)

candles = load_data("./marketData/XRPUSDT-5m-2024")
predictedPos = bot.predict(candles)

print(predictedPos)
