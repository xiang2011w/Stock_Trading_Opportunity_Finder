import datetime as dt
import experiment1 as ex1
import experiment2 as ex2
import ManualStrategy as man
import matplotlib.pyplot as plt
import marketsimcode as marketsim


def author(self):
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "xzhang934"  # replace tb34 with your Georgia Tech username

def gtid():
    return 903541088  # replace with your GT ID number



def separate_trades(trades_df):
    index = trades_df.index.tolist()
    long_index = []
    short_index = []
    for i in range (len(trades_df)):
        if trades_df.ix[i,0] > 0:
            long_index.append(index[i])
        elif trades_df.ix[i,0] < 0:
            short_index.append(index[i])
    return long_index, short_index

def generate_manualStrategy_graph(sym, normed_manual, normed_benchmark, long_index, short_index):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(normed_manual, label="normed_portfolio_manual", color="red")
    ax.plot(normed_benchmark, label="normed_portfolio_benchmark", color="green")
    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Portfolio value')
    plt.grid(True)
    plt.legend()
    plt.title('Manual Strategy (in-sample) vs Benchmark')
    for long in long_index:
        plt.axvline(x=long, color='blue')
    for short in short_index:
        plt.axvline(x=short, color='black')
    plt.savefig('ManualStrategy.png')
    plt.close()



#### to create out-of-sample graph, only change title
def generate_manualStrategy_graph_out (sym, normed_manual, normed_benchmark, long_index, short_index):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(normed_manual, label="normed_portfolio_manual", color="red")
    ax.plot(normed_benchmark, label="normed_portfolio_benchmark", color="green")
    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Portfolio value')
    plt.grid(True)
    plt.legend()
    plt.title('Manual Strategy (out-of-sample) vs Benchmark')
    for long in long_index:
        plt.axvline(x=long, color='blue')
    for short in short_index:
        plt.axvline(x=short, color='black')
    plt.savefig('ManualStrategy_out.png')
    plt.close()

if __name__ == "__main__":

    # Compare Manual strategy and Strategy learner In-sample trading JPM
    sym = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    start_val = 100000

    experiment1 = ex1.do_experiment1(sym, start_date, end_date, start_val)
    experiment2 = ex2.do_experiment2(sym, start_date, end_date, start_val)



##############################################################
    ## generate manual strategy report chart:
    # manual portfolio
    manual_strategy = man.ManualStrategy()
    orders_manual = manual_strategy.testPolicy(symbol=sym, sd=start_date, ed=end_date, sv=start_val)
    portfolio_manual = marketsim.compute_portvals(orders_manual)
    normed_portfolio_manual = portfolio_manual / portfolio_manual.iloc[0]


    # benchmark portfolio
    orders_benchmark = orders_manual.copy()
    orders_benchmark.iloc[:, :] = 0
    orders_benchmark.iloc[0] = 1000
    portfolio_benchmark = marketsim.compute_portvals(orders_benchmark)
    normed_portfolio_benchmark = portfolio_benchmark / portfolio_benchmark.iloc[0]

    long_index, short_index = separate_trades(orders_manual)

    generate_manualStrategy_graph(sym, normed_portfolio_manual, normed_portfolio_benchmark, long_index, short_index)


    #######################################################
    # out of sample manual strategy VS benchmark
    start_date_out = dt.datetime(2010, 1, 1)
    end_date_out = dt.datetime(2011, 12, 31)

    manual_strategy_out = man.ManualStrategy()
    orders_manual_out = manual_strategy_out.testPolicy(symbol=sym, sd=start_date_out, ed=end_date_out, sv=start_val)
    portfolio_manual_out = marketsim.compute_portvals(orders_manual_out)
    normed_portfolio_manual_out = portfolio_manual_out / portfolio_manual_out.iloc[0]


    # benchmark portfolio
    orders_benchmark_out = orders_manual_out.copy()
    orders_benchmark_out.iloc[:, :] = 0
    orders_benchmark_out.iloc[0] = 1000
    portfolio_benchmark_out = marketsim.compute_portvals(orders_benchmark_out)
    normed_portfolio_benchmark_out = portfolio_benchmark_out / portfolio_benchmark_out.iloc[0]

    long_index_out, short_index_out = separate_trades(orders_manual_out)
    generate_manualStrategy_graph_out(sym, normed_portfolio_manual_out, normed_portfolio_benchmark_out, long_index_out, short_index_out)

    ############################################################
    ## calculate performance statistics

    #### in sample: manual strategy
    cum_ret_portfolio_manual  = portfolio_manual[-1] / portfolio_manual[0] - 1
    daily_rets_portfolio_manual = (normed_portfolio_manual[1:] / normed_portfolio_manual[:-1].values) - 1
    daily_rets_portfolio_manual = daily_rets_portfolio_manual[1:]  # day 0 has no return
    avg_daily_ret_portfolio_manual = daily_rets_portfolio_manual.mean()
    std_daily_ret_portfolio_manual = daily_rets_portfolio_manual.std()

    #print('*******************   manual strategy on in-sample   *****************')
    #print('cumulative return of manual strategy on in sample', cum_ret_portfolio_manual)
    #print('standard deviation of daily return manual strategy on in sample', std_daily_ret_portfolio_manual)
    #print('mean of daily return manual strategy on in sample', avg_daily_ret_portfolio_manual)



    #### out of sample: manual strategy
    cum_ret_portfolio_manual_out = portfolio_manual_out [-1] / portfolio_manual_out [0] - 1
    daily_rets_portfolio_manual_out  = (normed_portfolio_manual_out [1:] / normed_portfolio_manual_out [:-1].values)-1
    daily_rets_portfolio_manual_out  = daily_rets_portfolio_manual_out [1:]  # day 0 has no return
    avg_daily_ret_portfolio_manual_out  = daily_rets_portfolio_manual_out.mean()
    std_daily_ret_portfolio_manual_out  = daily_rets_portfolio_manual_out.std()

    #print('*******************   manual strategy on out-of-sample   *****************')
    #print('cumulative return of manual strategy on out-of-sample', cum_ret_portfolio_manual_out)
    #print('standard deviation of daily return manual strategy on out-of-sample', std_daily_ret_portfolio_manual_out)
    #print('mean of daily return manual strategy on out-of-sample', avg_daily_ret_portfolio_manual_out)

    #### in sample: benchmark
    cum_ret_portfolio_benchmark = portfolio_benchmark[-1] / portfolio_benchmark[0] - 1
    daily_rets_portfolio_benchmark = (normed_portfolio_benchmark[1:] / normed_portfolio_benchmark[:-1].values) - 1
    daily_rets_portfolio_benchmark = daily_rets_portfolio_benchmark[1:]  # day 0 has no return
    avg_daily_ret_portfolio_benchmark = daily_rets_portfolio_benchmark.mean()
    std_daily_ret_portfolio_benchmark = daily_rets_portfolio_benchmark.std()

    #print('*******************   benchmark on in-sample   *****************')
    #print('cumulative return of benchmark on in sample', cum_ret_portfolio_benchmark)
    #print('standard deviation of daily return benchmark on in sample', std_daily_ret_portfolio_benchmark)
    #print('mean of daily return benchmark on in sample', avg_daily_ret_portfolio_benchmark)

    #### out of sample: benchmark
    cum_ret_portfolio_benchmark_out = portfolio_benchmark_out[-1] / portfolio_benchmark_out[0] - 1
    daily_rets_portfolio_benchmark_out = (normed_portfolio_benchmark_out[1:] / normed_portfolio_benchmark_out[:-1].values) - 1
    daily_rets_portfolio_benchmark_out = daily_rets_portfolio_benchmark_out[1:]  # day 0 has no return
    avg_daily_ret_portfolio_benchmark_out = daily_rets_portfolio_benchmark_out.mean()
    std_daily_ret_portfolio_benchmark_out = daily_rets_portfolio_benchmark_out.std()

    #print('*******************   benchmark on out-of-sample   *****************')
    #print('cumulative return of benchmark on in sample', cum_ret_portfolio_benchmark_out)
    #print('standard deviation of daily return benchmark on in sample', std_daily_ret_portfolio_benchmark_out)
    #print('mean of daily return benchmark on in sample', avg_daily_ret_portfolio_benchmark_out)