import sys
sys.path.append('../')

from bayesian_optimization_bo import BO
import pickle
import numpy as np
from scipy.stats import bernoulli
from multiprocessing.dummy import Pool as ThreadPool
from sklearn import svm
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split, KFold


max_iter = int(5000+10)

linear_bandit = False

ls = np.array([1, 1])
v_kernel = 0.5
M_target = 200
log_file_name = "saved_synth_funcs/random_features_ls_" + str(ls) + "_v_kernel_" + str(v_kernel) + \
        "_M_" + str(M_target) + ".pkl"
random_features = pickle.load(open(log_file_name, "rb"))
domain = random_features["domain"]


obs_noise = 0.01**2
# obs_noise = 0.05**2

diabetes_data = pd.read_csv("clinical_data/diabetes.csv")
label = np.array(diabetes_data["Outcome"])
features = np.array(diabetes_data.iloc[:, :-1])
X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size=0.3, stratify=label, random_state=0)
n_ft = X_train.shape[1]
n_classes = 2

def svm_reward_function(param):
    parameter_range = [[1e-4, 1.0], [1e-4, 1.0]]
    C_ = param[0]
    C = C_ * (parameter_range[0][1] - parameter_range[0][0]) + parameter_range[0][0]
    gam_ = param[1]
    gam = gam_ * (parameter_range[1][1] - parameter_range[1][0]) + parameter_range[1][0]

    clf = svm.SVC(kernel="rbf", C=C, gamma=gam, probability=True)
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)
    acc = np.count_nonzero(pred == Y_test) / len(Y_test)

    obs = np.random.normal(acc, np.sqrt(obs_noise))

    return obs, acc

dim = 2
pbounds = {}
for i in range(dim):
    pbounds["x" + str(i+1)] = (0, 1)

ts = np.arange(1, max_iter+5)
beta_t = np.ones(len(ts))

run_list = np.arange(5)
for itr in run_list:
    np.random.seed(itr)

    log_file_name = "results_bo/res_noise_var_" + str(obs_noise) + "_iter_" + str(itr) + ".pkl"
    if linear_bandit:
        log_file_name = log_file_name[:-4] + "_linear_bandit.pkl"

    bo = BO(f=svm_reward_function, pbounds=pbounds, log_file=log_file_name, beta_t=beta_t, \
              random_features=random_features, linear_bandit=linear_bandit, domain=domain, save_interval=100)
    bo.maximize(n_iter=max_iter, init_points=1)
