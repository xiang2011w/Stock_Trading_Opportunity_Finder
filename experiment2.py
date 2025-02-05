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

def generate_portfolio_graph(sym, port1, port2, port3):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(port1, label="normed_portfolio (impact = 0.00)", color="black")
    ax.plot(port2, label="normed_portfolio (impact = 0.20)", color="red")
    ax.plot(port3, label="normed_portfolio (impact = 0.30)", color="blue")
    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Portfolio value')
    plt.grid(True)
    plt.legend()
    plt.title('JPM In-sample: StrategyLearner with different impact factors')
    plt.savefig('Experiment2.png')
    plt.close()


def cal_num_trades(sym, trades):
    num = 0
    for i in range(0, trades.shape[0]):
        if trades.ix[i, sym] != 0:
            num += 1
    return num


def do_experiment2(sym, start_date, end_date, start_val):

    # generate trades df for different impact factors' portfolio
    strategy_learner1 = str.StrategyLearner(impact=0.0, commission=0.0)
    # use in-sample data to build model and also use in-sample data to test
    strategy_learner1.add_evidence(symbol=sym, sd=start_date, ed=end_date, sv=start_val)
    orders_strategy1 = strategy_learner1.testPolicy(symbol=sym, sd=start_date, ed=end_date, sv=start_val)

    strategy_learner2 = str.StrategyLearner(impact=0.20, commission=0.0)
    # use in-sample data to build model and also use in-sample data to test
    strategy_learner2.add_evidence(symbol=sym, sd=start_date, ed=end_date, sv=start_val)
    orders_strategy2 = strategy_learner2.testPolicy(symbol=sym, sd=start_date, ed=end_date, sv=start_val)

    strategy_learner3 = str.StrategyLearner(impact=0.30, commission=0.0)
    # use in-sample data to build model and also use in-sample data to test
    strategy_learner3.add_evidence(symbol=sym, sd=start_date, ed=end_date, sv=start_val)
    orders_strategy3 = strategy_learner3.testPolicy(symbol=sym, sd=start_date, ed=end_date, sv=start_val)


    # 1st metric: calculate portfolio for different impact factors' portfolio
    portfolio_strategy1 = marketsim.compute_portvals(orders_strategy1, commission=0.0)
    normed_portfolio_strategy1 = portfolio_strategy1 / portfolio_strategy1.iloc[0]

    portfolio_strategy2 = marketsim.compute_portvals(orders_strategy2, commission=0.0)
    normed_portfolio_strategy2 = portfolio_strategy2 / portfolio_strategy2.iloc[0]

    portfolio_strategy3 = marketsim.compute_portvals(orders_strategy3, commission=0.0)
    normed_portfolio_strategy3 = portfolio_strategy3 / portfolio_strategy3.iloc[0]


    generate_portfolio_graph(sym, normed_portfolio_strategy1, normed_portfolio_strategy2, normed_portfolio_strategy3)

    # 2nd metric: calculate total number of trades for different impact factors' portfolio
    trades_num1 = cal_num_trades(sym, orders_strategy1)
    trades_num2 = cal_num_trades(sym, orders_strategy2)
    trades_num3 = cal_num_trades(sym, orders_strategy3)

    #print('impact factor = 0.0, the number of trades is ', trades_num1)
    #print('impact factor = 0.20, the number of trades is ', trades_num2)
    #print('impact factor = 0.30, the number of trades is ', trades_num3)

