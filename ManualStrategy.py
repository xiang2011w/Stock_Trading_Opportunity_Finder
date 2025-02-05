import datetime as dt
import random
import pandas as pd
import util as ut
import indicators as indicators


class ManualStrategy(object):

    def __init__(self):
        self.window_size = 20
        pass

    def testPolicy(
        self,
        symbol="AAPL",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=100000,
    ):
        # get normalized stock prices df, same as strategylearner.py
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices_df = prices_all.drop(['SPY'], axis=1)  # remove SPY from df
        prices_df = prices_df.ffill().bfill()  # fill nan values
        prices_df = prices_df / prices_df.ix[0]  # normalize the prices

        # calculate technical indicators with indicators.py, same as strategylearner.py
        # 1st indicator: Exponential moving average(EMA)
        ema = indicators.cal_ema(prices_df, self.window_size)

        # 2nd indicator: bollinger bands percentage, bb_percent
        bb_percent = indicators.cal_bb_percent(prices_df, self.window_size)

        # 3rd indicator:  momentum
        momentum = indicators.cal_momentum(prices_df, self.window_size)

        # create new df for trades and initialize with zero (no action)
        trades_df = prices_df[syms].copy()
        trades_df.ix[:, :] = 0

        # set initial position as 0
        cur_pos = 0  # current position

        # manual trading policy based on indicators observations:
        for i in range(0, len(prices_df)):

            # if current stock price is too high relative to ema or bb_percent, and momentum is negative, SHORT, reach minimum -1000 position
            if (bb_percent.iloc[i,0] > 0.6) and (momentum.iloc[i,0] < -0.1):
                trades_df.ix[i] = -1000 - cur_pos
                cur_pos = -1000

            if (prices_df.iloc[i,0] > 1.05 * ema.iloc[i,0]) and (momentum.iloc[i,0] < -0.1):
                trades_df.ix[i] = -1000 - cur_pos
                cur_pos = -1000

            # if current stock price is too low relative to ema or bb_percent, and momentum is positive, LONG, reach maximum +1000 position
            #elif (bb_percent.iloc[i,0] < 0.2) and (momentum.iloc[i,0] > 0.1):
                #trades_df.iloc[i] = 1000 - cur_pos
                #cur_pos = 1000

            elif (prices_df.iloc[i,0] < 0.95 * ema.ix[i,0]) and (momentum.iloc[i,0] > 0.1):
                trades_df.iloc[i] = 1000 - cur_pos
                cur_pos = 1000

        return trades_df

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "xzhang934"  # replace tb34 with your Georgia Tech username

    def gtid(self):
        return 903541088  # replace with your GT ID number