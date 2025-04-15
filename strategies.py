"""
The definitions of the strategies
"""


class Strategy:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class SMACrossoverStrategy(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # somehow define the strategy so it can be called
