from strategies import *
from dataGetter import *
from backtester import *
from performanceMetrics import *

from rich.progress import track
from numpy.lib.stride_tricks import sliding_window_view as swv


faiss_train_data = getCandles(
    "./marketData/BTCUSDT-5m-2022",
)#[:100]

D1 = getCandles(
    "./marketData/BTCUSDT-5m-2023",
)#[:100]


S = KnnIndicatorsStrategy
P = PFMetric()


tmp_s = S()
index = faiss.IndexFlatL2(3)

try:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
except Exception as e:
    print(e)
    print("No GPU for faiss, using CPU instead.")
    
norm_indicators, faiss_train_data = tmp_s.get_norm_indicators(faiss_train_data)
index.add(norm_indicators)


default_params = {
    "index": index,
    "faiss_data": faiss_train_data,
    "k": 1,
}

params, p1 = {'sma_window': 5, 'atr_window': 5, 'rsi_window': 5, 'max_pos_len': 10, 'tp': 0.01, 'sl': 0.01}, 1.6957450784473715
So = S(**params, **default_params)



# ============ GENERATE SIGNALS ============


k = 3
sl = 0.01
tp = 0.02
max_dist = 0.1
max_pos_len = 25


# Step 1: Get nearest neighbors
indicators, data = So.get_norm_indicators(D1)
print("Knn... ", end="", flush=True)
distances, indices = So.index.search(indicators, k)
print("Done!", flush=True)

# Filter rows where max neighbor distance < threshold
valid_rows = distances[:, k-1] < max_dist
distances = distances[valid_rows]
indices = indices[valid_rows]

# Step 2: Prepare price data
high = faiss_train_data["High"].to_numpy().astype(np.float32)  # Save memory
low = faiss_train_data["Low"].to_numpy().astype(np.float32)
close = faiss_train_data["Close"].to_numpy().astype(np.float32)


# FIXME from here its all AI slop


# Create sliding window views (memory-efficient)
high_sw = swv(high, max_pos_len)  # shape: [n_windows, max_pos_len]
low_sw = swv(low, max_pos_len)

# Calculate max/min for all possible windows in one pass
global_max_high = high_sw.max(axis=1)  # shape: [n_windows]
global_min_low = low_sw.min(axis=1)

# Get valid indices and entry prices
max_valid_idx = len(close) - max_pos_len - 1
valid_idx_mask = (indices <= max_valid_idx).all(axis=1)
valid_indices = indices[valid_idx_mask]
entry_prices = close[valid_indices]

# Vectorized lookup of pre-computed extremes
max_highs = global_max_high[valid_indices]  # shape: [n_valid, k]
min_lows = global_min_low[valid_indices]

# TP/SL conditions
buy_tp = (entry_prices + tp <= max_highs).any(axis=1)
buy_sl = (entry_prices - sl >= min_lows).any(axis=1)

sell_tp = (entry_prices - tp >= min_lows).any(axis=1)
sell_sl = (entry_prices + sl <= max_highs).any(axis=1)

# Results
results = {
    'long_tp_ratio': buy_tp.mean(),
    'long_sl_ratio': buy_sl.mean(),
    'short_tp_ratio': sell_tp.mean(),
    'short_sl_ratio': sell_sl.mean(),
    'signals': {
        'long': np.select([buy_tp, buy_sl], [1, -1], default=0),
        'short': np.select([sell_tp, sell_sl], [1, -1], default=0)
    }
}

print(results)
