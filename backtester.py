import numpy as np
import heapq
from tradingBot import *
from joblib import Parallel, delayed
from tqdm import tqdm


def vectorized_backtest(candles, entry_signals, sl, tp, max_positions=1):
    high = candles['High'].values
    low = candles['Low'].values
    close = candles['Close'].values
    n = len(candles)
    positions = np.zeros(n, dtype=np.int8)

    # Precompute all entry-exit pairs
    entries = []
    for i in range(len(entry_signals)):
        if entry_signals[i] != 0:
            entry_idx = 100 + i  # Adjust based on your window offset
            direction = entry_signals[i]
            entry_price = close[entry_idx]

            # Calculate exit index
            start = entry_idx + 1
            if start >= n:
                exit_idx = n
            else:
                if direction == 1:
                    tp_price = entry_price * (1 + tp)
                    sl_price = entry_price * (1 - sl)
                    exit_mask = (high[start:] >= tp_price) | (low[start:] <= sl_price)
                else:
                    tp_price = entry_price * (1 - tp)
                    sl_price = entry_price * (1 + sl)
                    exit_mask = (low[start:] <= tp_price) | (high[start:] >= sl_price)

                exit_points = np.where(exit_mask)[0]
                exit_idx = start + exit_points[0] if exit_points.size > 0 else n

            entries.append((entry_idx, exit_idx, direction))

    # Event-driven processing
    entries.sort(key=lambda x: x[0])
    exit_heap = []
    current_positions = 0

    for entry in entries:
        entry_idx, exit_idx, direction = entry

        # Process exits that happen before this entry
        while exit_heap and exit_heap[0][0] <= entry_idx:
            _ = heapq.heappop(exit_heap)
            current_positions -= 1

        if current_positions < max_positions:
            heapq.heappush(exit_heap, (exit_idx, entry_idx, direction))
            current_positions += 1
            positions[entry_idx:exit_idx] += direction

    return positions


botSMA = TradingBotSMA(5, 10)
botKNN = TradingBotKNN(
    ["./marketData/XRPUSDT-5m-2020-23"],#, "./marketData/ETHUSDT-5m-2020-24", "./marketData/BTCUSDT-5m-2020-24", "./marketData/DOGEUSDT-5m-2024"],
    "ae_noShuffle_beta-03",
    [300, 100, 50, 25],
    sl=0.01,
    tp=0.02,
    minDistThreshold=0.12,
    k=2,
    posMaxLen=24,
    dimNum=25
)

sl = 0.01
tp = 0.01

candles = getCandles("./marketData/XRPUSDT-5m-2024")
# candles = candles[:50000]


def compute_signal_wrapper(candles, max_positions):
    entry_signals = np.zeros(len(candles), dtype=np.int8)
    position_open = False
    current_position_end = 0

    def compute_signal(i):
        nonlocal position_open, current_position_end

        # Skip if we're in the middle of a position
        if i < current_position_end:
            return 0

        # Calculate signal only if no position is open
        window = candles.iloc[i:i + 100]
        # predictedPos, _ = botSMA.predict(window)
        predictedPos, _ = botKNN.predict(window)

        # If we open a new position, set the skip window
        if predictedPos != 0:
            position_open = True
            # Find exit index (next 100 candles or end of dataset)
            current_position_end = min(i + 100, len(candles))

        return predictedPos

    # Process in parallel chunks but with position awareness
    chunk_size = 1000  # Adjust based on typical position duration
    for chunk_start in tqdm(range(0, len(candles), chunk_size), desc="Processing"):
        chunk_end = min(chunk_start + chunk_size, len(candles))

        # Only process indices >= current_position_end
        start = max(chunk_start, current_position_end)
        if start >= chunk_end:
            continue

        indices = range(start, chunk_end)
        chunk_results = Parallel(n_jobs=1)(
            delayed(compute_signal)(i) for i in indices
        )

        # Update entry signals and position status
        for idx, res in zip(indices, chunk_results):
            entry_signals[idx] = res
            if res != 0:
                position_open = True
                current_position_end = min(idx + 100, len(candles))

    return entry_signals


# Usage
entry_signals = compute_signal_wrapper(candles, max_positions=1)

candles["strategy_position"] = vectorized_backtest(candles, entry_signals, sl, tp, max_positions=1)
candles.to_csv("backtest.csv")
