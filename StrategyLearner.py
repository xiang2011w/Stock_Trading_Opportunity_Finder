""""""  		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		   	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			 	 	 		 		 	
or edited.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Student Name: Tucker Balch (replace with your name)  		  	   		   	 			  		 			 	 	 		 		 	
GT User ID: tb34 (replace with your User ID)  		  	   		   	 			  		 			 	 	 		 		 	
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
import datetime as dt
import numpy as np
import pandas as pd  		  	   		   	 			  		 			 	 	 		 		 	
import util as ut
import RTLearner as rtl
import BagLearner as bgl
import indicators as indicators


  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
class StrategyLearner(object):  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 			  		 			 	 	 		 		 	
        If verbose = False your code should not generate ANY output.  		  	   		   	 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		   	 			  		 			 	 	 		 		 	
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 			  		 			 	 	 		 		 	
    :type impact: float  		  	   		   	 			  		 			 	 	 		 		 	
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 			  		 			 	 	 		 		 	
    :type commission: float  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    # constructor  		  	   		   	 			  		 			 	 	 		 		 	
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Constructor method  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        self.verbose = verbose  		  	   		   	 			  		 			 	 	 		 		 	
        self.impact = impact
        self.commission = commission
        # ref cs7646 classification trader hints: set N=5 to study 5 day return for Y data
        self.N_day = 5
        # ref cs7646 classification trader hints: set leaf_size >=5 to avoid degenerate overfit solution
        self.learner = bgl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size": 10, "verbose": False}, bags=10, boost=False, verbose=False)
        # set windowsize to be 20 for indicators calculation
        self.window_size = 20


    # this method is to create a RTlearner, and train it for trading
    # converting regression learner into classification learner
    def add_evidence(  		  	   		   	 			  		 			 	 	 		 		 	
        self,  		  	   		   	 			  		 			 	 	 		 		 	
        symbol="AAPL",
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 			  		 			 	 	 		 		 	
        ed=dt.datetime(2009, 12, 31),
        sv=100000,
    ):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Trains your strategy learner over a given time frame.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
        :param symbol: The stock symbol to train on  		  	   		   	 			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		   	 			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		   	 			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		   	 			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		   	 			  		 			 	 	 		 		 	
        :type sv: int  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
        # add your code to do learning here  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
        # example usage of the old backward compatible util function  		  	   		   	 			  		 			 	 	 		 		 	
        syms = [symbol]  		  	   		   	 			  		 			 	 	 		 		 	
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices_df = prices_all.drop(['SPY'], axis=1) # remove SPY from df
        prices_df = prices_df.ffill().bfill()  # fill nan values
        prices_df = prices_df / prices_df.ix[0]   # normalize the prices

        #calculate technical indicators with indicators.py
        # 1st indicator: Exponential moving average(EMA)
        ema = indicators.cal_ema(prices_df, self.window_size)

        # 2nd indicator: bollinger bands percentage, bb_percent
        bb_percent = indicators.cal_bb_percent(prices_df, self.window_size)

        # 3rd indicator:  momentum
        momentum = indicators.cal_momentum(prices_df, self.window_size)

        # create x_data_train to group together three indicators into one df
        rows = len(prices_df) - self.N_day
        x_data_train = np.zeros((rows, 3))
        for i in range(0, rows):
            x_data_train[i][0] = ema.iloc[i]
            x_data_train[i][1] = bb_percent.iloc[i]
            x_data_train[i][2] = momentum.iloc[i]

        #print('strategy learner x_data_train ', x_data_train,'x_data_train.shape', x_data_train.shape)

        # calculate Y values, ref: cs7646 classification trader hints
        y_data_train=[]  # create a new list for data y

        for i in range(0, rows):
            ret = prices_df.ix[i+self.N_day]/prices_df.ix[i] - 1.0  # calculate return ratio
            y_buy = 0.002 + self.impact  # use 0.2% as threshold
            y_sell = -0.002-self.impact

            if (ret.iloc[0] > y_buy):
                y_data_train.append(1)  # LONG action
            elif (ret.iloc[0] < y_sell):
                y_data_train.append(-1)  # SHORT action
            else:
                y_data_train.append(0)   # do nothing

        y_data_train = np.array(y_data_train)  #list to array

        #use X, y data to train
        self.learner.add_evidence(x_data_train, y_data_train)

    # this method should use the existing policy and test it against new data
    def testPolicy(  		  	   		   	 			  		 			 	 	 		 		 	
        self,  		  	   		   	 			  		 			 	 	 		 		 	
        symbol="AAPL",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=100000,
    ):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Tests your learner using data outside of the training data  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
        :param symbol: The stock symbol that you trained on on  		  	   		   	 			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		   	 			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		   	 			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		   	 			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		   	 			  		 			 	 	 		 		 	
        :type sv: int  		  	   		   	 			  		 			 	 	 		 		 	
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		   	 			  		 			 	 	 		 		 	
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		   	 			  		 			 	 	 		 		 	
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		   	 			  		 			 	 	 		 		 	
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		   	 			  		 			 	 	 		 		 	
        :rtype: pandas.DataFrame  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
        # here we build a fake set of trades  		  	   		   	 			  		 			 	 	 		 		 	
        # your code should return the same sort of data
        ################ copy from above addEvidence function to generate new test data X with 3 indicators
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices_df = prices_all.drop(['SPY'], axis=1)  # remove SPY from df
        prices_df = prices_df.ffill().bfill()  # fill nan values
        prices_df = prices_df / prices_df.ix[0]  # normalize the prices

        # calculate technical indicators with indicators.py
        # 1st indicator: Exponential moving average(EMA)
        ema = indicators.cal_ema(prices_df, self.window_size)
        ema.bfill()

        # 2nd indicator: bollinger bands percentage, bb_percent
        bb_percent = indicators.cal_bb_percent(prices_df, self.window_size)
        bb_percent.bfill()

        # 3rd indicator:  momentum
        momentum = indicators.cal_momentum(prices_df, self.window_size)
        momentum.bfill()

        # create x_data_test to group together three indicators into one df
        rows = len(prices_df) - self.N_day
        x_data_test = np.zeros((rows, 3))
        for i in range(0, rows):
            x_data_test[i][0] = ema.iloc[i]
            x_data_test[i][1] = bb_percent.iloc[i]
            x_data_test[i][2] = momentum.iloc[i]

        ################## above is the same logic as add_evidence function above

        # use x test data to query the learner model and get the results of actions
        y_data_test = self.learner.query(x_data_test)

        #convert query results (action list) into actual trades dataframe
        # ref: youtube CS 7646: QLearning Trader Project Overview
        # create new df for trades and initialize with zero
        trades_df = prices_df[syms].copy()
        trades_df.iloc[:, :] = 0

        #loop through query results to update trades_df
        #set initial position as 0
        cur_pos = 0 # current position

        for i in range(0, rows):
            if y_data_test[i] > 0:   #long action, make cur_Pos to be maximum +1000 shares
                trades_df.iloc[i, :] = 1000 - cur_pos
                cur_pos = 1000

            elif y_data_test[i] < 0:   #short action, make cur_Pos to be minimum -1000 shares
                trades_df.iloc[i, :] = -1000 - cur_pos
                cur_pos = -1000

            # else y_data_test[i] == 0 do nothing
        return trades_df


    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "xzhang934"  # replace tb34 with your Georgia Tech username

    def gtid(self):
        return 903541088  # replace with your GT ID number