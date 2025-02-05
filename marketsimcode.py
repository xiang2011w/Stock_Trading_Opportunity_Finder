""""""  		  	   		   	 			  		 			 	 	 		 		 	
"""Project 6: Market simulator.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
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
  		  	   		   	 			  		 			 	 	 		 		 	
Student Name: Xiang Zhang (replace with your name)  		  	   		   	 			  		 			 	 	 		 		 	
GT User ID: xzhang934 (replace with your User ID)  		  	   		   	 			  		 			 	 	 		 		 	
GT ID: 903541088 (replace with your GT ID)  		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
import datetime as dt  		  	   		   	 			  		 			 	 	 		 		 	
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data  		  	   		   	 			  		 			 	 	 		 		 	

def author():
    return "xzhang934"  # replace tb34 with your Georgia Tech username.

def gtid():
    return 903541088  # replace with your GT ID number


def compute_portvals(  		  	   		   	 			  		 			 	 	 		 		 	
    trades,                     # modified for project 6
    symbol = 'JPM',
    start_val=1000000,
    commission=9.95,  		  	   		   	 			  		 			 	 	 		 		 	
    impact=0.005,  		  	   		   	 			  		 			 	 	 		 		 	
):  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    Computes the portfolio values.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    :param orders_file: Path of the order file or the file object  		  	   		   	 			  		 			 	 	 		 		 	
    :type orders_file: str or file object  		  	   		   	 			  		 			 	 	 		 		 	
    :param start_val: The starting value of the portfolio  		  	   		   	 			  		 			 	 	 		 		 	
    :type start_val: int  		  	   		   	 			  		 			 	 	 		 		 	
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		   	 			  		 			 	 	 		 		 	
    :type commission: float  		  	   		   	 			  		 			 	 	 		 		 	
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		   	 			  		 			 	 	 		 		 	
    :type impact: float  		  	   		   	 			  		 			 	 	 		 		 	
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		   	 			  		 			 	 	 		 		 	
    :rtype: pandas.DataFrame  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    # this is the function the autograder will call to test your code  		  	   		   	 			  		 			 	 	 		 		 	
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		   	 			  		 			 	 	 		 		 	
    # code should work correctly with either input  		  	   		   	 			  		 			 	 	 		 		 	
    # TODO: Your code here

    # modify trades to create orders for maketsim.py to compute_portvals,
    # Ref: copy from my own code in theoreticallyOptimalStrategy.py testPolicy
    orders = trades.copy()
    orders.ix[:, :] = 0

    # add in Order and shares columns
    orders['Order'] = 0
    orders['Shares'] = 0
    # select Symbol, Order, Shares columns, ref: https://medium.com/dunder-data/selecting-subsets-of-data-in-pandas-6fcd0170be9c
    orders.columns = ['Symbol', 'Order', 'Shares']

    # loop through the df and update BUY or SELL actions based on value of trade
    size2 = trades.shape[0]
    for i in range(size2):
        trade = trades.ix[i, 0]
        if trade == 0:
            orders.ix[i, 'Order'] = 'BUY'
        if trade < 0:
            orders.ix[i, 'Order'] = 'SELL'
        if trade > 0:
            orders.ix[i, 'Order'] = 'BUY'

        orders.ix[i, 'Symbol'] = symbol   # update all rows to be 'JPM'
        orders.ix[i, 'Shares'] = abs(trade)  # ref https://www.w3schools.com/python/ref_func_abs.asp


    # modified for project 6
    # get starting and end date in the csv file
    start_date = orders.index.min()
    end_date = orders.index.max()


    # get the symbols for different stocks
    sym = orders.Symbol.unique().tolist()   # ref: https://www.dezyre.com/recipes/list-unique-values-in-pandas-dataframe

    # create data frame for prices
    prices = get_data(sym, pd.date_range(start_date, end_date))
    prices = prices[sym]    # remove SPY
    prices['Cash'] = 1  #add a cash column as the last column with all values to be 1

    # create trades dataframe, copy prices df and fill up with zeros
    prices_x = prices.shape[0]
    prices_y = prices.shape[1]
    trades = prices.copy()
    trades.ix[:, :] = np.zeros((prices_x, prices_y)) # fill up with zeros

    # update values in trades df
    for date, row in orders.iterrows():

        num_shares = row['Shares']
        traded_symbol = row['Symbol']
        share_price = prices.ix[date, traded_symbol]
        trade_value = num_shares * share_price

        if row['Order'] == 'BUY':
            trades.loc[date, traded_symbol] += num_shares
            trades.loc[date, 'Cash'] -= trade_value + (commission + impact * trade_value)

        else:
            trades.loc[date, traded_symbol] -= num_shares
            trades.loc[date, 'Cash'] += trade_value - (commission + impact * trade_value)

    # creat holding df, copy trades df
    holdings = trades.copy()
    holdings.ix[:, :] = np.zeros((trades.shape[0], trades.shape[1]))  # fill up with zeros
    holdings.ix[0, 'Cash'] = start_val

    for row in range(0, holdings.shape[0]):
        if row == 0:
            holdings.ix[0, :] += trades.ix[0, :] # update the first row in holdings df, as starting point of for loop
        else:
            holdings.ix[row, :] += holdings.ix[row-1, :] + trades.ix[row, :]

    # create value df
    value = holdings * prices

    #create portfolio df
    portfolio = value.sum(axis=1)
    return portfolio


def test_code():  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    Helper function to test code  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    # this is a helper function you can use to test your code  		  	   		   	 			  		 			 	 	 		 		 	
    # note that during autograding his function will not be called.  		  	   		   	 			  		 			 	 	 		 		 	
    # Define input parameters  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    of = "./orders/orders-02.csv"
    sv = 1000000  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # Process orders  		  	   		   	 			  		 			 	 	 		 		 	
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		   	 			  		 			 	 	 		 		 	
    if isinstance(portvals, pd.DataFrame):  		  	   		   	 			  		 			 	 	 		 		 	
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		   	 			  		 			 	 	 		 		 	
    else:  		  	   		   	 			  		 			 	 	 		 		 	
        "warning, code did not return a DataFrame"  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # Get portfolio stats  		  	   		   	 			  		 			 	 	 		 		 	
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		   	 			  		 			 	 	 		 		 	
    start_date = portvals.index.min()
    end_date = portvals.index.max()

    cum_ret = portvals[-1] / portvals[0] - 1
    daily_rets = (portvals[1:] / portvals[:-1].values) - 1
    daily_rets = daily_rets[1:]  # day 0 has no return
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()

    # Sharpe Ratio yearly
    sr_daily = (avg_daily_ret - 0) / std_daily_ret  # daily sr
    k = 252 ** 0.5  # k is adjustment factor
    sharpe_ratio = k * sr_daily  # annual sr

    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [  		  	   		   	 			  		 			 	 	 		 		 	
        0.2,  		  	   		   	 			  		 			 	 	 		 		 	
        0.01,  		  	   		   	 			  		 			 	 	 		 		 	
        0.02,  		  	   		   	 			  		 			 	 	 		 		 	
        1.5,  		  	   		   	 			  		 			 	 	 		 		 	
    ]  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # Compare portfolio against $SPX  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Date Range: {start_date} to {end_date}")  		  	   		   	 			  		 			 	 	 		 		 	
    print()  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		   	 			  		 			 	 	 		 		 	
    print()  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		   	 			  		 			 	 	 		 		 	
    print()  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		   	 			  		 			 	 	 		 		 	
    print()  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		   	 			  		 			 	 	 		 		 	
    print()  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		   	 			  		 			 	 	 		 		 	
    test_code()  		  	   		   	 			  		 			 	 	 		 		 	
