import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

candles = pd.read_csv("backtested-candles-SMA.csv")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

# Plot Close price
ax1.plot(candles["Close"], label="Close Price", color="blue")
ax1.set_ylabel("Price")
ax1.set_title("Price Chart")

# Plot strategy positions
ax2.plot(candles["strategy_position"], label="Strategy Position", color="orange")
ax2.set_ylabel("Position")
ax2.set_title("Strategy Signal")

# Add grid and legends
ax1.grid(True)
ax2.grid(True)
ax1.legend()
ax2.legend()

plt.tight_layout()
plt.show()
