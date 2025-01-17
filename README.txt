TRADING AI PYTHON


I need more data:
    The autoencoder seems to learn well with "just" those few xrp years,
    but maybe adding another few pairs will help it generalize more

    The KNN is in dire need of a shit ton of data. It performs very well
    when the parameters are strict, but only a few trades are made.
    To make more good trades, the parameters have to remain strict, but
    if we get more data, there will be more neighbours.

    Would it make sense to introduce other pairs in the KNN algorythm?
    Having the candles so normalised, trend is trend. Maybe it would work...

    This would mean that each dataPoint has to store not only the index,
    but also the pair? Or i can just remember te range of indexes for each pair

    1st testing:
        I added ETH data in the mix. At a first glance, this will be useless.
        When using the same normalization, it is clear that ETH is much more
        stable than XRP. This means that the abs value of the numbers in the
        vector are much lower.

        Actual results:
            eth+xrp:  [47689] Wins: 39, Losses: 53, Profit factor: 1.47
            xrp only: [47589] Wins: 22, Losses: 51, Profit factor: 0.86

        So it seems that the autoencoder with more pairs encodes trends better.

    Now let's take a look at the knn dataPoints:
        XRP: [-0.2017,  0.1775, -0.0058,  ...,  0.0023, -0.0592, -0.0839]
        ETH: [-0.1996,  0.1827,  0.0055,  ..., -0.0069, -0.0475, -0.0795]

        They seem surprisingly similar...

    Now let's take a look with the autoencoder trained only on xrp:
        XRP: [ 9.2949e-03, -4.0895e-03, -1.7379e-01,  ..., -2.5334e-02, -4.5922e-03, -3.5404e-02]
        ETH: [ 5.1194e-03, -2.1501e-03, -1.5941e-01,  ..., -2.6484e-02, 5.2337e-03, -3.2262e-02]

        tbh they seem similar as well...

    Well now this means that I can probably use other pairs in the knn search without problem :)

    2nd testing:
        I tested putting eth data in the knn, using the autoencoder trained on both.
        The profit factor is higher, but interestingly, the number of trades is lower.
        Maybe it's a result of having more negative confirmation? idk.

        100K candles test (5 dims, posMaxLen=48):
            only xrp:  [99968] Wins: 244, Losses: 440, Profit factor: 1.11
            xrp + eth: [99763] Wins: 158, Losses: 322, Profit factor: 0.98
        100K candles test (10 dims, posMaxLen=24):
            only xrp:  [99917] Wins: 65, Losses: 117, Profit factor: 1.11
            xrp + eth: [99903] Wins: 35, Losses: 63, Profit factor: 1.11
        100K candles test (10 dims, posMaxLen=48):
            only xrp:  [99982] Wins: 252, Losses: 411, Profit factor: 1.23
            xrp + eth:
        100K candles test (20 dims, posMaxLen=48):
            only xrp:  [99958] Wins: 242, Losses: 436, Profit factor: 1.11
            xrp + eth: [99627] Wins: 184, Losses: 310, Profit factor: 1.19