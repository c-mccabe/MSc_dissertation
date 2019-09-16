# This is the main Python script. It loads the data and performs a Bayesian
# optimisation of the hyperparameters of the base learners. It stores the optimal
# parameters in the 'reports' folder.


import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from skmultilearn.problem_transform import LabelPowerset
import GPyOpt
from GPyOpt.experiment_design import initial_design
from GPyOpt import methods
from scipy.io import mmread
from scipy.sparse import csr_matrix, vstack, hstack
from scipy import stats
from functools import partial


from functions import LabelPowerset_Extra_predict, LabelPowerset_RF_predict
from functions import load_data, objective_function

# Here you can change the batch size of the Bayesian optimisation as well as
# the number of cores available to use.

batch_size = 1
num_cores  = 4

# Load the data

X_train, X_test, y_train, y_test = load_data()

# Define the domain

domain = [

          # The domain over which we will run the Bayesian optimisation.
          # We have seven base models each of which takes four hyperparameters
          # for a total of 28 hyperparameters to tune. For this reason, it's
          # necessary to restrict the domains somewhat to more likely values.
    
          # Three random forests
    
          {'name': 'n_estimators', 'type': 'discrete', 'domain': range(250)}, 
          {'name': 'max_depth', 'type': 'discrete', 'domain': range(25000)},         
          {'name': 'min_samples_split', 'type': 'discrete', 'domain': range(2, 4)},
          {'name': 'min_samples_leaf', 'type': 'discrete', 'domain': range(1, 3)},
    
          {'name': 'n_estimators', 'type': 'discrete', 'domain': range(250)}, 
          {'name': 'max_depth', 'type': 'discrete', 'domain': range(25000)},        
          {'name': 'min_samples_split', 'type': 'discrete', 'domain': range(2, 4)},
          {'name': 'min_samples_leaf', 'type': 'discrete', 'domain': range(1, 3)},
    
          {'name': 'n_estimators', 'type': 'discrete', 'domain': range(50, 250)},
          {'name': 'max_depth', 'type': 'discrete', 'domain': range(8000, 25000)},       
          {'name': 'min_samples_split', 'type': 'discrete', 'domain': range(2, 4)},
          {'name': 'min_samples_leaf', 'type': 'discrete', 'domain': range(1, 3)},
    
          #Â Four extremely randomised trees
    
          {'name': 'n_estimators', 'type': 'discrete', 'domain': range(50, 250)},
          {'name': 'max_depth', 'type': 'discrete', 'domain': range(8000, 25000)},   
          {'name': 'min_samples_split', 'type': 'discrete', 'domain': range(2, 4)},
          {'name': 'min_samples_leaf', 'type': 'discrete', 'domain': range(1, 3)},
    
          {'name': 'n_estimators', 'type': 'discrete', 'domain': range(250)}, 
          {'name': 'max_depth', 'type': 'discrete', 'domain': range(25000)},
          {'name': 'min_samples_split', 'type': 'discrete', 'domain': range(2, 4)},
          {'name': 'min_samples_leaf', 'type': 'discrete', 'domain': range(1, 3)},

          {'name': 'n_estimators', 'type': 'discrete', 'domain': range(50, 250)},
          {'name': 'max_depth', 'type': 'discrete', 'domain': range(25000)},    
          {'name': 'min_samples_split', 'type': 'discrete', 'domain': range(2, 4)},
          {'name': 'min_samples_leaf', 'type': 'discrete', 'domain': range(1, 3)},
    
          {'name': 'n_estimators', 'type': 'discrete', 'domain': range(250)}, 
          {'name': 'max_depth', 'type': 'discrete', 'domain': range(8000, 25000)},    
          {'name': 'min_samples_split', 'type': 'discrete', 'domain': range(2, 4)},
          {'name': 'min_samples_leaf', 'type': 'discrete', 'domain': range(1, 3)}
    
          ]

# f is the function we will pass to the Bayesian optimisation to maximise. We only
# want f to be a function of the model hyperparameters and not the data, so we specify
# the data now.

f = partial(objective_function, y_test = y_test, X_train = X_train,
            X_test = X_test, y_train = y_train)

# Here we initialise the Bayesian optimisation.

BO =  methods.BayesianOptimization(f = f,  # Objective function       
                                   domain = domain,          # Box-constraints of the problem
                                   acquisition_type='EI',        # Expected Improvement
                                   evaluator_type = 'local_penalization', # allows for parallel computation
                                   batch_size = batch_size, # batch size
                                   maximize = True, # we want to maximize our function
                                   num_cores = num_cores
                                  )

for i in range(10):
    
    # This loops iterates the Bayesian optimisation. It saves a report every
    # 10 iterations which will contain the optimal paramaters and the accuracy
    # achieved using those hyperparameters.
    
    max_iter = 10
                                            
    BO.run_optimization(max_iter)
    BO.save_report('reports/2saved_report_step_%d'%i)


