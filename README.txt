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

        100K candles, 5 dims, posMaxLen=48:
            only xrp:  [99968] Wins: 244, Losses: 440, Profit factor: 1.11
            xrp + eth: [99763] Wins: 158, Losses: 322, Profit factor: 0.98
        100K candles, 10 dims, posMaxLen=48:
            only xrp:  [99982] Wins: 252, Losses: 411, Profit factor: 1.23
            xrp + eth: [99903] Wins: 186, Losses: 318, Profit factor: 1.17
        100K candles, 20 dims, posMaxLen=48:
            only xrp:  [99958] Wins: 242, Losses: 436, Profit factor: 1.11
            xrp + eth: [99627] Wins: 184, Losses: 310, Profit factor: 1.19

        100K candles, 5 dims, posMaxLen=24:
            only xrp:  [99968] Wins: 73, Losses: 128, Profit factor: 1.14
            xrp + eth: [95180] Wins: 34, Losses: 68, Profit factor: 1.0
        100K candles, 10 dims, posMaxLen=24:
            only xrp:  [99917] Wins: 65, Losses: 117, Profit factor: 1.11
            xrp + eth: [99903] Wins: 35, Losses: 63, Profit factor: 1.11
        100K candles, 20 dims, posMaxLen=24:
            only xrp:  [98959] Wins: 61, Losses: 116, Profit factor: 1.05
            xrp + eth: [99554] Wins: 41, Losses: 64, Profit factor: 1.28

    Bonus test:
        100K candles, 20 dims, posMaxLen=48, no autoencoder:
            only xrp:  [99713] Wins: 273, Losses: 567, Profit factor: 0.96

    Now, what do these tests mean?
        ... no idea :(
        Lets ask chatGPT :>
        ChatGPT confirms what I thought... I should add more data and see what happens

    Added even more data (btc):
        Memory usage skyrocketed, so I had to make some changes on the data.
        Now, instead of pre-making the data with a sliding window, the idea is
        that we say to torch to use one on the data. This is the only way I can
        train this on my pc. A little problem is that there are bugs everywhere, probably.
        Let's take a deep dive in how to train the autoencoder:


AUTOENCODER TRAINING:
    After some initial testings, it is probable that a general autoencoder
    is way better than a specialised one. This raises a demand for lots of data.
    The method I was using to prepare the data increased it's size 100-fold (numOfCandles-fold).
    The much better approach is to put the normalised candles in a 1D tensor and say to
    the torch module to apply a sliding window with a step of 3 and length of 100 to it.

    On paper, it is pretty simple, now that I wrote it down. Then where did I fail?

    Let's write the training steps in pseudocode:
    1) get the data:
        This step is trivial, I already have the functions for it.
        The function is getNormCandles() from dataGetter.py
        The returned candles are given as a pd.DataFrame.

    2) Prepare the candles:
        The goal is to turn them into a data structure that is accepted by torch
        and can be seen with a sliding window. It seems that the appropriate structure
        is a torch tensor. I basically need a really long vector that contains
        all the normalised candles, flattened. I then want torch to read it using a
        sliding window, as mentioned.
        Do I really need to use a tensor?
        Which dimensions should it have?

    3) Train iterate through the data using the sliding window:
        It seems that the "correct" way of implementing a sliding window is
        to make a child of the class Dataset. Did I implement it right? Does it really work?
        Using the custom dataset, the model should correctly train with the candles window

    4) Use the encoder:
        Since we also have to use this, here are the steps.
        We have some candles dataFrame as an input, which we need to:
        - get only the last 100 candles
        - normalise them
        - flatten them
        - turn them into a tensor
        And then we can feed it to the autoencoder. The result is a dataPoint of the
        candle window.

    5) KNN:
        Since I mentioned dataPoints, here is a breakdown of the knn that I want to
        set up.
        As mentioned above, we get a dataPoint through the autoencoder. When fitting
        the KNN model, we have to keep track which range of indexes belongs to which
        dataset, to then correctly simulate the position.


So now that I put this down in words, here is how I should implement this:
    1) Autoencoder training:
        1.1) Data preparation
            1.1.1) Get normalised candles
            1.1.2) Flatten them
            1.1.3) Convert them to a tensor
        1.2) Data feeding
            1.2.1) Create the sliding window dataset
            1.2.2) initialise it and feed it the tensor

    2) KNN model:
        1) Fit the model:
            1.1) Get the train candles
            1.2)

    Who am I kidding... I don't know enough about torch to write this
    Let's go on a study trip :>
    I'll start with this: https://www.youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1
    Also, here is the documentation: https://pytorch.org/docs/stable/index.html

        2) Use the model