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

        2) Use the model

    Who am I kidding... I don't know enough about torch to write this
    Let's go on a study trip :>
    I'll start with this: https://www.youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1
    Also, here is the documentation: https://pytorch.org/docs/stable/index.html


PyTorch study:
    Tensors:
        https://pytorch.org/docs/stable/tensors.html
        Just like arrays, but more powerful


DeepSeek autoencoder analysis:
    - Deleted that [0] in the validation loop of the autoencoder
    - Tried a bit modifying the beta parameter
    - Tried to not shuffle (it works it seems?)
    - Tried cosine similarity (Does it really works so well?)


Cosine similarity:
    seems to yield better results, but I still have to fine tune it.


I've hit a plateau:
    With the new changes, it seems that the profit factor is stuck around 1.3
    Granted, I didn't implement those fancy profit factor amplification methods,
    but I didn't implement fees either.

    So what now?
    I really have no idea. I think that I want a plain profit factor of at least 1.5
    before committing to other improvements.

    I downloaded a year of doge data to test on, but the results are always the same:
    Somewhat positive, but waaay too close to neutral. Which is a kind of positive result.
    It means that I was right in thinking that the kline has prediction power, but
    it seems not nearly enough.. or maybe I'm just not extracting it right?
    

Potential issues:
	1) Normalization of candles:
		The current normalization (tanh(h/o-1), ...) does normalize well,
		but the normalized candles lose the measure of volatility.
	
	2) Autoencoder:
		Is it needed? What is the optimal bottleneck?
		If dimensionality reduction is needed, consider also PCA or t-SNE
	
	3) KNN:
		Could use a probabilistic threshold for how many neighbours go in the same direction
		instead of using only 100% match. Expanding on that, the function predict()
		could return a float instead of an int, still in the range [-1, 1], but 0.1 would mean
		that the prediction is really weak.
	
	4) Temporal awareness:
		Throwing everything in a flattened vector, then autoencoder and then KNN, 
		absolutely destroys the temporal dependencies of the data. Consider using a
		CNN, RNN, LSTM or Transformer, to maintain the temporal dependencies.
		Look into "Multivariate time series forecasting"
		Look also into how to incorporate values that are not dependent on time (volume)

	5) Other data:
		Instead of using only the kline, try using the orderbook or other things.    


LSTM model:
    The main difference here is that I have to label the data.
    I guess that I have to set a time range and then mark the price at the end of it.
    Or another method would be to set a tp and sl and then count how many candles does it need to reach it.

    Data labeling:
        - Next candle?
        - Price in n candles?
        - sl/tp - -1 1
        

What data to use?
	The data is the first stone on which this church will be built. With trash data we can expect trash results.
	

	OHLC:
		The standard data for any trader
		
	Technical indicators:
		They give signals based on price action movement
		Might be a pain in the ass to calculate
	
	Orderbook:
		Could be helpful to get some insight in the market depth,
		but it has lots of data that needs to be interpreted just right.
	
	Volume:
		Could be useful, maybe look into VWAP


New idea while running:
    It's not new, but I think I could revisit using various indicators in a KNN model.
    Instead of only using RSI, ADX and whatever, I could put in like 10 different indicators
    and maybe also the autoencoder output.
    The big problem here is to normalise the data such that one feature does not overshadow everything else.

    Examples for scaling:
        Ranged (from x to y): minmax scaling
        Unbounded (SMA, EMA): z-score normalization
        Boolean: keep it 0 1
        Vectors (autoencoder): L2 normalization
        Volatility like ATR: log + z-score
        

How to truly backtest:
	https://www.youtube.com/watch?v=NLBXgSmRBgU

	To get an optimal strategy, we have to have:
		1) A strategy idea
			Example: KNN model
		2) Development data
			Example: 3 years of XRP 5min
		3) An optimiser:
			Example: K parameter, distance function, max position len, ...
	
	1) In-sample excellence:
		We first have to create the stratregy's signal AT EACH BAR: -1 sell, 0 hold, 1 buy
			I'm guessing that I could use non integer values in case of different tp/sl or position
			sizes. Then we compute the strategy's return (close to close) FOR EACH BAR. With these granular results we
			can then really get a nicer approximation of the results. This is usually done by checking
			the profits at each trade, but doing it this way brings up much more information.
			By cumulative summing up all of the granular results, we get a crude backtest.
			
			Programming tip:
				we have a pandas DF with all the candles.
				We then run our strategy and add a column for position (assuming only 1 position at a time, mor columns for more positions at a time)
				We add a column "strategy_signal" that will contain -1 (short) 1 (long) 0 (hold) values
				Then to get the returns for each candle is as simple as running this:
					df["return"] = np.log(df["close"]).diff().shift(-1)		# logged difference of close to close candle
					df["strategy_return"] = df["return"] * strategy_signal 	# *0 will nullify, -1 sell and 1 buy
			
			Looking at this crude backtest we have to ask ourselves:
				IS THIS EXCELLENT?
					no -  polish the strategy further
					yes - nice
				IS THIS OBVIOUSLY OVERFIT?
					yes - 			simplify the strategy
					not obviously - continue with testing
			
			Now that we have a seemingly excellent strategy, we have to ask ourselves this:
				Was this excellent performance found due to patterns intrinsic in the data
				or just because we optimised the strategy a bit too much?
				
				Any strategy will have a nice performance if we optimise it enough, but a good
				strategy will be good even with little optimisation.
				To test further, we have to assume that our strategy is trash. To disprove this
				belief, we will run the In-sample permutation test. If it performs worse there,
				there is a good chance that our strategy was not worthless.
				
	2) In-sample permutation test
		We first create permutations of the data that the strategy was optimised on (about 1000).
		Then we run our crude backtest on each of them. If our original optimisation outperforms the
		vast majority of the permutations, we could start to assume that the strategy is not really overfit,
		but it actually seeks patterns in the price action.
		
		But of course, this permutation test destroys any data involving volume, long memory stuff and serial correlation.
		So any strategy that uses those parameters will be obviously better in non-permutations. Anyhow, this is not a bad
		test, since if it performs average as the permutations, it is obviously overfitted.
		
		A good strategy should outperform AT LEAST 99% of the permutated tests.
		IMPORTANT: that 99% is only indicative, if you fiddle enough with any strategy, it will pass it. DON'T OVERUSE IT
		
		This step will take lots of time, since you should aim at like 1000 permutations, which
			means optimising the strategy 1000 times. We shall see how slow the KNN stuff is :')
			
		But why not use validation data?
			Well, because once used, you can't use it anymore (from personal experience :')
			So instead of wasting that precious validation data, we first run the algo on permutations.
			Then MAYBE if it performs well, we will test it on the val data.
		
	So now that we have an excellent algorythm, that performs better on real data than on permutations, lets continue.
		
	3) Walk forward test:
		Once we know that our algo outperforms random permutations, we can test it on some future validation data.
		The algorythm is expected to perform worse on this data, since there is no bias (assuming it's 100% fresh data, never tested before)
		After this test we can ask ourselves if this algorythm's performance is worth trading. 
		
	4) Walk forward permutation test
		Just to make sure of the results, we can permutate the validation data.
		Our algo should outperform the vast majority of the permutations of the validation data.
		The difference in strategies will be much smaller than before, since this is data that has not
		been optimised on. Of course, the better it outperforms the permutations, the best the strategy is.
		
	So if the walk forward test gives a good profit factor AND the walk forward permutation test is promising, the strategy can be traded.


Books to read:
	https://www.youtube.com/watch?v=ftFptCxm5ZU

	- Systematic trading - Robert Carver
	- Permutation and randomization tests for trading system development algorythms in C++ - Timothy Masters
	- Testing and Tuning Market Trading Systems - Timothy Masters
