"""
The definitions of the strategies
"""


class Strategy:
    pass


class SMACrossoverStrategy(Strategy):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    # somehow define the strategy so it can be called
