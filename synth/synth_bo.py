import sys
sys.path.append('../')

from bayesian_optimization_bo import BO
import pickle
import numpy as np
from scipy.stats import bernoulli

max_iter = int(1e4+500)

linear_bandit = False # set it to False for kernelized bandit algorithms; set it to True for linear bandit algorithms
binary = False # set it to True for Bernoulli noise; set it to False for Gaussian noise

ls = 0.1

obs_noise = 0.3**2
# obs_noise = 0.4**2

log_file_name = "saved_synth_funcs/synth_func_ls_" + str(ls) + "_noise_var_" + str(obs_noise) + ".pkl"
all_func_info = pickle.load(open(log_file_name, "rb"))
domain = all_func_info["domain"]
f = all_func_info["f"]

log_file_name = "saved_synth_funcs/random_features_ls_" + str(ls) + "_noise_var_" + str(obs_noise) + ".pkl"
random_features = pickle.load(open(log_file_name, "rb"))

def synth_func(param):
    x = param[0]
    ind = np.argmin(np.abs(domain - x))

    if not binary:
        obs = np.random.normal(f[ind], np.sqrt(obs_noise))
    else:
        obs = bernoulli.rvs(f[ind])

    return obs, f[ind, 0]

ts = np.arange(1, max_iter+10)
beta_t = np.sqrt(2) * np.ones(len(ts))

run_list = np.arange(10)
for itr in run_list:
    np.random.seed(itr)

    if not binary:
        log_file_name = "results_bo/res_noise_var_" + str(obs_noise) + "_iter_" + str(itr) + ".pkl"
    else:
        log_file_name = "results_bo/res_iter_" + str(itr) + "_binary.pkl"

    bo = BO(f=synth_func, pbounds={'x1':(0, 1)}, log_file=log_file_name, beta_t=beta_t, \
              random_features=random_features, linear_bandit=linear_bandit, domain=domain)
    bo.maximize(n_iter=max_iter, init_points=1)
