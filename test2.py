from strategies import *
from dataGetter import *
from backtester import *
from performanceMetrics import *

from rich.progress import track


faiss_train_data = getCandles(
    "./marketData/BTCUSDT-5m-2022",
)[:100]

D1 = getCandles(
    "./marketData/BTCUSDT-5m-2023",
)[:100]


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


signals = So.generate_signals(D1)

print(f"Signals:\n{signals}")

print(signals["strategy_signal"].value_counts())
