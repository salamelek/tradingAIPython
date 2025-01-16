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