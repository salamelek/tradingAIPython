import numpy as np
import pandas as pd
from numba import njit  # For CPU acceleration


class MatrixBacktester:
    def __init__(self, ohlc: np.ndarray):
        """
        ohlc: 2D numpy array [timestamp, open, high, low, close, volume]
        """
        self.ohlc = ohlc
        self.signals = None
        self.positions = None

    @njit
    def _vectorized_indicators(self, windows: np.ndarray):
        """Calculate indicators in matrix form"""
        n = len(self.ohlc)
        close = self.ohlc[:, 3]  # Close prices

        # Matrix of rolling windows (shape: [n_windows, n_periods])
        indicator_matrix = np.empty((n, len(windows)))

        for i, window in enumerate(windows):
            # Vectorized rolling calculations
            for j in range(window, n):
                indicator_matrix[j, i] = np.mean(close[j - window:j])

        return indicator_matrix

    @njit
    def _matrix_signals(self, indicator_matrix: np.ndarray, rules: np.ndarray):
        """Signal generation using matrix operations"""
        # rules: 2D array of conditions [indicator_idx, operator, threshold]
        signals = np.zeros(len(self.ohlc))

        for i in range(len(rules)):
            idx, op, val = rules[i]
            if op == 0:  # Greater than
                signals += (indicator_matrix[:, idx] > val).astype(np.float32)
            elif op == 1:  # Less than
                signals += (indicator_matrix[:, idx] < val).astype(np.float32)

        return np.sign(signals)

    @njit
    def _vectorized_position(self, signals: np.ndarray, max_pos: int):
        """Position management using vector ops"""
        positions = np.zeros_like(signals)
        current_pos = 0

        for i in range(1, len(signals)):
            if current_pos < max_pos and signals[i] != 0:
                positions[i] = signals[i]
                current_pos += 1
            elif positions[i - 1] != 0 and signals[i] == 0:
                current_pos -= 1

        return positions

    def run(self, windows: np.ndarray, rules: np.ndarray, max_pos: int = 1):
        """End-to-end vectorized backtest"""
        # Step 1: Matrix indicator calculation
        indicators = self._vectorized_indicators(windows)

        # Step 2: Signal generation
        self.signals = self._matrix_signals(indicators, rules)

        # Step 3: Position management
        self.positions = self._vectorized_position(self.signals, max_pos)

        # Step 4: Vectorized PnL calculation
        returns = np.zeros(len(self.ohlc))
        entry_prices = np.zeros(len(self.ohlc))

        in_trade = False
        for i in range(1, len(self.ohlc)):
            if not in_trade and self.positions[i] != 0:
                entry_prices[i] = self.ohlc[i, 3]  # Entry at close
                in_trade = True
            elif in_trade and self.positions[i] == 0:
                returns[i] = self.positions[i - 1] * (
                        self.ohlc[i, 3] / entry_prices[i - 1] - 1)
                in_trade = False

        return returns.cumsum()