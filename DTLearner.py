""""""  		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
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
"""  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
import numpy as np  		  	   		   	 			  		 			 	 	 		 		 	
import pandas as pd
  		  	   		   	 			  		 			 	 	 		 		 	
class DTLearner(object):  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    This is a Linear Regression Learner. It is implemented correctly.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 			  		 			 	 	 		 		 	
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		   	 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    def __init__(self, leaf_size=1, verbose=False):
        self.data = None
        self.tree = None
        self.leaf_size = leaf_size   #leaf size default value is 1
        self.verbose = verbose
        pass  # move along, these aren't the drones you're looking for

    def author(self):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        :return: The GT username of the student  		  	   		   	 			  		 			 	 	 		 		 	
        :rtype: str  		  	   		   	 			  		 			 	 	 		 		 	
        """
        return "xzhang934"  # replace tb34 with your Georgia Tech username  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    def add_evidence(self, data_x, data_y):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Add training data to learner  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
        :param data_x: A set of feature values used to train the learner  		  	   		   	 			  		 			 	 	 		 		 	
        :type data_x: numpy.ndarray  		  	   		   	 			  		 			 	 	 		 		 	
        :param data_y: The value we are attempting to predict given the X data  		  	   		   	 			  		 			 	 	 		 		 	
        :type data_y: numpy.ndarray  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	

        # create dataframe from input data_x and data_y
        new_data = pd.DataFrame(data_x)
        new_data['Y'] = data_y                     # ref https://www.geeksforgeeks.org/
        self.data = new_data
        self.tree = self.build_tree(new_data)


        """  		  	   		   	 			  		 			 	 	 		 		 	
        Estimate a set of test points given the model we built.  		  	   		   	 			  		 			 	 	 		 		 	

        :param points: A numpy array with each row corresponding to a specific query.  		  	   		   	 			  		 			 	 	 		 		 	
        :type points: numpy.ndarray  		  	   		   	 			  		 			 	 	 		 		 	
        :return: The predicted result of the input data according to the trained model  		  	   		   	 			  		 			 	 	 		 		 	
        :rtype: numpy.ndarray  		  	   		   	 			  		 			 	 	 		 		 	
        """

    def query(self, points):
        y_estimate = np.array([])
        num_points = int(points.shape[0])

        for i in range(0, num_points):
            index = 0  # start from node 0
            factor = int(self.tree[index, 0])
            while factor != -1:
                split = self.tree[index, 1]
                points_val = points[i, factor]
                if points_val <= split:
                    left_index = int(self.tree[index, 2])  # relative index
                    index += left_index
                else:
                    right_index = int(self.tree[index, 3])  # relative index
                    index += right_index

                factor = int(self.tree[index, 0])
				
            val = self.tree[index, 1]
            y_estimate = np.append(y_estimate, val)
        return y_estimate

    def build_tree(self, data):   # build up tree recursively, ref JR Quinlan code as shown in lecture

        df_x = data.iloc[:, 0:-1]  # x factors, except last column in dataframe
        df_y = data.iloc[:, -1]  # label y, last column in dataframe

        if data.shape[0] <= self.leaf_size:  # when there is less rows in df to build a leaf
            y_val = np.mean(df_y)
            return np.array([-1, y_val, -1, -1])  # return [leaf, data.y, NA,NA]

        if len(pd.unique(df_y)) == 1:                       # check last column if all same value: https://stackoverflow.com/questions/54405704/
            return np.array([-1, np.mean(df_y), -1, -1])   # return [leaf, data.y, NA,NA]
        else:
            i = self.find_factor_index(data)

            splitVal = np.median(data.iloc[:, i])
            left_data = data[data.iloc[:, i] <= splitVal]
            right_data = data[data.iloc[:, i] > splitVal]

            # special case: if only left right tree or only left tree use mean value as splitVal
            if (left_data.shape[0] ==  df_x.shape[0]) or (right_data.shape[0] ==  df_x.shape[0]):
                splitVal = np.mean(data.iloc[:, -1])
                return np.array([-1, splitVal, -1, -1])

            left = self.build_tree(left_data)
            right = self.build_tree(right_data)

            if left.ndim == 1:
                root = [i, splitVal, 1, 1+1]
            else:
                root = [i, splitVal, 1, left.shape[0]+1]
            cur_tree = np.vstack((root, left, right))  # ref numpy.org
            return cur_tree   # return current tree, not the final tree

    def find_factor_index(self, data): 	 # determine the best factor used to build the root/current node
        cor_table = np.array(data.corr())       # get correlation coefficient table
        lower_coefficients = np.tril(cor_table, k=-1)   # select lower half of the table, diagonal matrix ref geeksforgeeks.org/python-pandas-dataframe-corr
        factor = abs(np.nan_to_num(lower_coefficients[-1])).argmax()   # find the max number in last row
        return int(factor)
