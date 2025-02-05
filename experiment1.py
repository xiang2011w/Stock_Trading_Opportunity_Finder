import datetime as dt
import StrategyLearner as str
import ManualStrategy as man
import matplotlib.pyplot as plt
import marketsimcode as marketsim

def author(self):
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "xzhang934"  # replace tb34 with your Georgia Tech username


def gtid(self):
    return 903541088  # replace with your GT ID number


def generate_portfolio_graph(sym, manual, strategy, benchmark):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(manual, label="normed_portfolio_manual", color="green")
    ax.plot(strategy, label="normed_portfolio_strategy", color="red")
    ax.plot(benchmark, label="normed_portfolio_benchmark", color="blue")
    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Portfolio value')
    plt.grid(True)
    plt.legend()
    plt.title('JPM In-sample: ManualStrategy VS StrategyLearner')
    plt.savefig('Experiment1.png')
    plt.close()






def do_experiment1(sym, start_date, end_date, start_val):
    # 1. manual strategy
    manual_strategy = man.ManualStrategy()
    orders_manual = manual_strategy.testPolicy(symbol=sym, sd=start_date, ed=end_date, sv=start_val)
    #print("manual orders", orders_manual)

    # 2. strategy learner
    strategy_learner = str.StrategyLearner(impact=0.005)
    # use in-sample data to build model and also use in-sample data to test
    strategy_learner.add_evidence(symbol=sym, sd=start_date,ed=end_date,sv=start_val)
    orders_strategy = strategy_learner.testPolicy(symbol=sym, sd=start_date,ed=end_date,sv=start_val)
    #print(orders_strategy)


    #3. benchmark portfolio
    #benchmark = manual_strategy.testPolicy(symbol=sym, sd=start_date, ed=end_date, sv=start_val)
    orders_benchmark = orders_manual.copy()
    orders_benchmark.iloc[:,:] = 0
    orders_benchmark.iloc[0] = 1000
    #print('bench mark trades', benchmark)



    #calculate portfolio
    portfolio_manual = marketsim.compute_portvals(orders_manual)
    normed_portfolio_manual = portfolio_manual / portfolio_manual.iloc[0]

    portfolio_strategy = marketsim.compute_portvals(orders_strategy)
    normed_portfolio_strategy = portfolio_strategy / portfolio_strategy.iloc[0]

    portfolio_benchmark = marketsim.compute_portvals(orders_benchmark)
    normed_portfolio_benchmark = portfolio_benchmark / portfolio_benchmark.iloc[0]

    #generate portfolio graph
    generate_portfolio_graph(sym, normed_portfolio_manual, normed_portfolio_strategy,normed_portfolio_benchmark)


    ######## generate statistics

    # 1. strategy learner
    strategy_val = portfolio_strategy * 100000
    strategy_cum_return = strategy_val.ix[-1] / strategy_val.ix[0] - 1
    strategy_daily_return = strategy_val / strategy_val.shift(1) - 1
    strategy_mean_daily = strategy_daily_return.mean()
    strategy_std_daily = strategy_daily_return.std()

    #print('strategy learner: \n')
    #print('cumulative return: %.4f \n' % strategy_cum_return)
    #print('standard deviation of daily return: %.4f \n' % strategy_std_daily)
    #print('mean of daily return: %.4f \n' % strategy_mean_daily)


    # 2. manual

    manual_val = portfolio_manual * 100000
    manual_cum_return = manual_val.ix[-1] / manual_val.ix[0] - 1
    manual_daily_return = manual_val / manual_val.shift(1) - 1
    manual_mean_daily = manual_daily_return.mean()
    manual_std_daily = manual_daily_return.std()

    #print('manual : \n')
    #print('cumulative return: %.4f \n' % manual_cum_return)
    #print('standard deviation of daily return: %.4f \n' % manual_std_daily)
    #print('mean of daily return: %.4f \n' % manual_mean_daily)




    # 3. benchmark

    benchmark_val = portfolio_benchmark * 100000
    benchmark_cum_return = benchmark_val.ix[-1] / benchmark_val.ix[0] - 1
    benchmark_daily_return = benchmark_val / benchmark_val.shift(1) - 1
    benchmark_mean_daily = benchmark_daily_return.mean()
    benchmark_std_daily = benchmark_daily_return.std()

    #print('Benchmark: \n')
    #print('cumulative return: %.4f \n' % benchmark_cum_return)
    #print('standard deviation of daily return: %.4f \n' % benchmark_std_daily)
    #print('mean of daily return: %.4f \n' % benchmark_mean_daily)