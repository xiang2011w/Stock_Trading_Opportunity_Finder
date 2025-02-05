Indicator Evaluation
------------------------
Parameters (in-sample):
 
-sym = "JPM"
-start_date = dt.datetime(2008, 1, 1)
-end_date = dt.datetime(2009, 12, 31)
-start_val = 100000

---------------------------------------------------------------------
testproject.py

- PYTHONPATH=../:. python testproject.py
- output: 
	- generate chart for Experiment1: JPM In-sample: ManualStrategy VS StrategyLearner 
	- generate chart for Experiment2: JPM In-sample: StrategyLearner with different impact factors

------------------------------------------------------------------------

experiment1.py 
	- run from testproject.py to conduct Experiment1 with in-sample data as required by assignment description

------------------------------------------------------------------------
experiment2.py 
	- run from testproject.py to conduct Experiment2 with in-sample data as required by assignment description

------------------------------------------------------------------------
StrategyLearner.py 
	- run from experiment1.py and experiment2.py to build up learner model with baglearner(RTlearner), and return predictions to improve the portfolio
	
------------------------------------------------------------------------
ManualStrategy.py 
	- run from experiment1.py to return predictions based on technical indicators

------------------------------------------------------------------------
