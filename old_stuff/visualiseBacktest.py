import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
candles = pd.read_csv("backtest-knn-2023.csv", parse_dates=['open_time'], index_col='open_time')  # adjust date column name


def plot_trades(ax, candles, color, direction):
    # Convert to numpy arrays for boolean operations
    prev_pos = candles['strategy_position'].shift(1).fillna(0).to_numpy()
    current_pos = candles['strategy_position'].to_numpy()

    # Find proper entries and exits
    if direction == 1:
        # Long entries: 0 -> 1 transitions
        entries = np.where((prev_pos == 0) & (current_pos == 1))[0]
        # Long exits: 1 -> 0 transitions
        exits = np.where((prev_pos == 1) & (current_pos == 0))[0]
    else:
        # Short entries: 0 -> -1 transitions
        entries = np.where((prev_pos == 0) & (current_pos == -1))[0]
        # Short exits: -1 -> 0 transitions
        exits = np.where((prev_pos == -1) & (current_pos == 0))[0]

    # Pair entries with exits
    trades = []
    entry_ptr = 0
    exit_ptr = 0

    while entry_ptr < len(entries) and exit_ptr < len(exits):
        entry_idx = entries[entry_ptr]

        # Find first exit after entry
        while exit_ptr < len(exits) and exits[exit_ptr] <= entry_idx:
            exit_ptr += 1

        if exit_ptr >= len(exits):
            break  # No exit found for this entry

        exit_idx = exits[exit_ptr]
        trades.append((entry_idx, exit_idx))

        # Move to next entry/exit pair
        entry_ptr += 1
        exit_ptr += 1

    # Plot the trades
    for entry_idx, exit_idx in trades:
        entry_time = candles.index[entry_idx]
        exit_time = candles.index[exit_idx]
        entry_price = candles.iloc[entry_idx]['Close']
        exit_price = candles.iloc[exit_idx]['Close']

        # Plot connecting line
        ax.plot([entry_time, exit_time], [entry_price, exit_price],
                color=color, alpha=0.6, linewidth=1.5, linestyle=':')

        # Plot markers
        marker = '^' if direction == 1 else 'v'
        ax.scatter(entry_time, entry_price, marker=marker,
                   color=color, s=80, edgecolors='k', zorder=5)
        ax.scatter(exit_time, exit_price, marker='o',
                   color=color, s=40, edgecolors='k', zorder=5)


# Calculate returns and metrics
candles["return"] = np.log(candles["Close"]).diff().shift(-1)
candles["strategy_return"] = candles["strategy_position"] * candles["return"]
returns = candles["strategy_return"].dropna()
cum_returns = returns.cumsum()

# Calculate metrics
profit_factor = returns[returns > 0].sum() / returns[returns < 0].abs().sum()
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 288)  # annualized for 5m data
total_return = cum_returns.iloc[-1]

# Calculate moving averages (adjust window sizes as needed)
sma_fast = candles['Close'].rolling(window=5).mean()  # Fast MA (e.g., 20 periods)
sma_slow = candles['Close'].rolling(window=10).mean()  # Slow MA (e.g., 100 periods)

# Create plot
plt.figure(figsize=(14, 12))

# Price chart with MAs and positions
ax1 = plt.subplot(3, 1, (1, 2))
plt.plot(candles['Close'], label='Price', color='#1f77b4', alpha=0.8, linewidth=1.5)
# plt.plot(sma_fast, label='Fast MA (5)', color='#ff7f0e', alpha=0.8, linewidth=1.2)
# plt.plot(sma_slow, label='Slow MA (10)', color='#9467bd', alpha=0.8, linewidth=1.2)

# Plot trades
plot_trades(ax1, candles, 'green', 1)  # Long trades
plot_trades(ax1, candles, 'red', -1)   # Short trades

plt.title('Price with Moving Averages and Trade Signals')
plt.grid(True, alpha=0.3)
plt.legend()


# Cumulative returns plot
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(cum_returns, color='#17becf', linewidth=1.5)
plt.fill_between(cum_returns.index, cum_returns, alpha=0.2, color='#17becf')
plt.title('Cumulative Returns')
plt.grid(True, alpha=0.3)

# Add metrics box
metrics_text = f"""
Profit Factor: {profit_factor:.2f}
Sharpe Ratio: {sharpe_ratio:.2f}
Total Return: {(total_return * 100):.2f}%"""
plt.gcf().text(0.85, 0.7, metrics_text, 
              bbox=dict(facecolor='white', alpha=0.5), 
              fontfamily='monospace')

plt.tight_layout()
plt.show()


"""
normal
pf: 1.04
sr: 2.08
tr: 75.48

overfit
pf: 1.04
sr: 2.08
tr: 75.48
"""
