"""
1) Create a strategy S in strategies.py
2) Optimise its parameters using training data
    In other words; we want to find the maximum of P(S(params), D1) -> So, p0
    P in this case is the performance function, that calculates the return or any other metric
    modifying the strategy parameters (probably using gradient descent or something like tha).
    p0 is the best performance achieved.
3) Once we have our optimised strategy So, we can test the performance of the strategy on n random permutations of D1.
    Note that n should be at the very least 100. We will denote the permutations of D1 as D1I.
    Once we have a list of performances for every permutation [P(S, D1i)], we can count how many of them
    are greater than the performance on D1. So count([P(S, D1i)] > P(So, D1)) = m. Now calculating the performance
    ratio pr = m / n, we have to aim at a pr < 0.01. Lower is, of course, better.
4) Now that our strategy So outperforms random permutations of D1, we can test it on some validation data D2.
    The performance P(So, D2) is expected to be lower than P(So, D1), so we have to decide if it's worth trading.
5) If we decide that the strategy is still good, we can run it on some permutations of D2.
    Just as before, we create a list of permutations of D2i and then run the performance function on them.
    Now the performance ratio pr is expected to be higher, but as before, we should aim at something at the very least
    below 0.5 and lower is better.
6) If we are satisfied with all the above criteria, we can proceed with trading :>
"""

from strategies import *
from dataGetter import *
from backtester import *
from performanceMetrics import *

from rich.progress import track


# Get the train and validation candles
# DURING TESTING, DON'T PUT SAME YEARS OF DATA IN THE TRAIN AND TRAIN_FAISS, SINCE THERE IS SOME CLEAR CORELATION
faiss_train_data = getCandlesFromFolders([
    # "./marketData/ETHUSDT-5m-2020",
    # "./marketData/ETHUSDT-5m-2021",
    "./marketData/ETHUSDT-5m-2022",
])

D1 = getCandlesFromFolders([
    "./marketData/BTCUSDT-5m-2023",
])

D2 = getCandles(
    "./marketData/BTCUSDT-5m-2024"
)


# Choose a strategy and a performance metric
S = KnnIndicatorsStrategy
P = PFMetric()


# create the faiss index
tmp_s = S()
index = faiss.IndexFlatL2(tmp_s.k)
norm_indicators, faiss_train_data = tmp_s.get_norm_indicators(faiss_train_data)
index.add(norm_indicators)


# define default strategy params
default_params = {
    "index": index,
    "faiss_data": faiss_train_data,
}


# Optimise the strategy S on D1
# params, p1 = optimise_strategy(P, S, D1, n_trials=100, **default_params)
params, p1 = {"sma_window": 5, "atr_window": 15, "rsi_window": 15, "max_pos_len": 57}, 1.7239907243622206
print(f"The best {P.name} achieved is {p1} with the parameters {params}")


# Create the optimised strategy So
So = S(**params, **default_params)


# Check the performance of So on DI
n1 = 1
performances1 = np.empty(n1, dtype=np.float64)


# Create permutations of D1 and evaluate them
for i in track(range(n1), description="Permuting D1"):
    D1i = create_permutation(D1)
    performances1[i] = P(So, D1i)

# Check the performance ratio
pr1 = np.sum(performances1 > p1) / n1
print(f"P-value of test data with n={n1} is {pr1} ({np.sum(performances1 > p1)} / n1)")
print(f"The performances were: {performances1}")


input("Press enter to continue [THIS WILL TEST THE STRATEGY ON THE VALIDATION DATA]")


# Test on validation data
p2 = P(So, D2)
print(f"The {P.name} for the validation dataset is {p2}")


# Permutate the validation data
n2 = 1
performances2 = np.empty(n2, dtype=np.float64)

for i in track(range(n2), description="Permuting D2"):
    D2i = create_permutation(D2)
    performances2[i] = P(So, D2i)

# Performance ratio
pr2 = np.sum(performances2 > p2) / n2
print(f"P-value of validation data with n={n2} is {pr2}")
