import pandas as pd


def backtest_strategy(data: pd.DataFrame) -> None:
    """
    Computes the positions taken by the strategy
    """

    # check that the data has ohlc data + strategy_signal
    required_cols = {"Open", "High", "Low", "Close", "strategy_signal"}

    if not required_cols.issubset(data.columns):
        raise Exception(f"The given data does not contain some required columns:\nGiven:    {set(data.columns)}\nRequired: {required_cols}")


def backtest_in_sample_permutation(data: pd.DataFrame) -> None:
    pass
