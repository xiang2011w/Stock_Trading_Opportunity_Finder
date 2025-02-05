import math
import pandas as pd
from util import get_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# 1st indicator SMA(simple moving average)
# ref CS7646 vectorize me
def cal_sma (prices_df, lookback):
    sma = prices_df.rolling(window=lookback, min_periods=lookback).mean() # ref CS7646 vectorize me
    return sma

# 2nd indicator EMA(Exponential moving average)
# ref https://www.learndatasci.com/tutorials/python-finance-part-3-moving-average-trading-strategy/
def cal_ema (prices_df, lookback):
    ema = prices_df.ewm(span=lookback, adjust=False).mean()
    return ema



# 3rd indicator: calculate bollinger bands
# ref CS7646 vectorize me
def cal_bb(prices_df, lookback):
    rolling_std = prices_df.rolling(window=lookback, min_periods=lookback).std()
    sma = prices_df.rolling(window=lookback, min_periods=lookback).mean()
    btm_band = sma - (2*rolling_std)
    top_band = sma + (2 * rolling_std)
    return top_band, btm_band


# 3rd indicator:calculate bollinger bands percentage, bb_percent
# ref CS7646 vectorize me
def cal_bb_percent(prices_df, lookback):
    rolling_std = prices_df.rolling(window=lookback, min_periods=lookback).std()
    sma = prices_df.rolling(window=lookback, min_periods=lookback).mean()
    btm_band = sma - (2*rolling_std)
    top_band = sma + (2 * rolling_std)
    bb_percent = (prices_df - btm_band) / (4*rolling_std)
    return bb_percent


# 4th indicator: calculate momentum
# ref CS7646 02-06
def cal_momentum(prices_df, lookback):
    return (prices_df/prices_df.shift(lookback))-1

# 5th indicator: calculate volatility
# ref https://www.datacamp.com/community/tutorials/finance-python-trading
def cal_volatility(prices_df, lookback):
    volatility = prices_df.rolling(window=lookback, min_periods=lookback, center=False).std() * math.sqrt(lookback)
    return volatility


##################################################################################################
##########################################plot 5 indicators below
# 1st indicator:plot SMA(simple moving average)
def gen_plot_sma(prices_df, lookback):
    sma = cal_sma(prices_df, lookback)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sma, label="SMA", color = "blue")
    ax.plot(prices_df, label="Stock price", color="purple")
    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Stock price')
    plt.grid(True)
    plt.legend()
    plt.title('Simple Moving Average(SMA) of Stock Price')
    plt.savefig('SMA.png')
    plt.close()

def gen_plot_ema(prices_df, lookback):
    ema = cal_ema(prices_df, lookback)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ema, label="EMA", color = "blue")
    ax.plot(prices_df, label="Stock price", color="purple")
    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Stock price')
    plt.grid(True)
    plt.legend()
    plt.title('Exponential Moving Average(EMA) of Stock Price')
    plt.savefig('EMA.png')
    plt.close()



# 3rd indicator: plot bollinger bands
def gen_plot_bb(prices_df, lookback):
    top_band, btm_band = cal_bb(prices_df, lookback)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(top_band, label="top_band", color="blue")
    ax.plot(btm_band, label="btm_band", color="brown")
    ax.plot(prices_df, label="Stock price", color="purple")

    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Stock price')
    plt.grid(True)
    plt.legend()
    plt.title('Bollinger Bands')
    plt.savefig('bb.png')
    plt.close()

# 3rd indicator: plot bb_percent
def gen_plot_bbp(prices_df, lookback):
    bbp = cal_bb_percent(prices_df, lookback)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(bbp, label="bb_percentage", color="orange")
    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Percentage')
    plt.grid(True)
    plt.legend()
    plt.title('Bollinger Bands percentage')
    plt.savefig('bbp.png')
    plt.close()

# 4th indicator: plot momentum
def gen_plot_momentum(prices_df, lookback):
    momentum = cal_momentum(prices_df, lookback)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(momentum, label="momentum", color="blue")
    ax.plot(prices_df, label="Stock price", color="purple")

    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Momentum')
    plt.grid(True)
    plt.legend()
    plt.title('Stock price and Momentum')
    plt.savefig('momentum.png')
    plt.close()


# 5th indicator: plot volatility, let's set lookback to be 10
def gen_plot_volatility(prices_df, lookback):
    volatility = cal_volatility(prices_df, lookback)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(volatility, label="volatility", color="blue")
    ax.plot(prices_df, label="Stock price", color="purple")

    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Volatility')

    plt.grid(True)
    plt.legend()
    plt.title('Stock price and Volatility')
    plt.savefig('volatility.png')
    plt.close()

def author():
    return "xzhang934"  # replace tb34 with your Georgia Tech username.

def gtid():
    return 903541088  # replace with your GT ID number

if __name__ == "__main__":
    sd = '01-01-2008'
    ed = '12-31-2009'
    dates = pd.date_range(sd, ed)
    prices_df = get_data(['JPM'],dates).drop(['SPY'], axis=1) # remove SPY from df
    prices_df = prices_df / prices_df.ix[0]
    gen_plot_sma(prices_df, 20)
    gen_plot_ema(prices_df, 20)
    gen_plot_bb(prices_df, 20)
    gen_plot_bbp(prices_df, 20)
    gen_plot_momentum(prices_df, 20)
    gen_plot_volatility(prices_df, 20)