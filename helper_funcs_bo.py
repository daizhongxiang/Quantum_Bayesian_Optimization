import numpy as np
from datetime import datetime
from scipy.optimize import minimize
import pickle

def acq_max(ac, M, random_features, bounds, nu_t, Sigma_t_inv, beta, linear_bandit, domain):
    para_dict={"M":M, "random_features":random_features, "nu_t":nu_t, "Sigma_t_inv":Sigma_t_inv, \
              "beta":beta, "linear_bandit":linear_bandit}
    ys = []
    for i, x in enumerate(domain):
        ys.append(-ac(x.reshape(1, -1), para_dict))

    ys = np.squeeze(np.array(ys))
    argmin_ind = np.argmin(ys)
    x_max = domain[argmin_ind, :]

    return x_max


class UtilityFunction(object):
    def __init__(self):
        self.kind = "ucb"
    def utility(self, x, para_dict):
        M, random_features, nu_t, Sigma_t_inv, beta, linear_bandit = para_dict["M"], \
            para_dict["random_features"],\
            para_dict["nu_t"], para_dict["Sigma_t_inv"], para_dict["beta"], \
            para_dict["linear_bandit"]

        if self.kind == 'ucb':
            return self._ucb(x, random_features, nu_t, Sigma_t_inv, beta, linear_bandit)

    @staticmethod
    def _ucb(x, random_features, nu_t, Sigma_t_inv, beta, linear_bandit):
        d = x.shape[1]
        s = random_features["s"]
        b = random_features["b"]
        obs_noise = random_features["obs_noise"]
        v_kernel = random_features["v_kernel"]
        M = b.shape[0]

        if not linear_bandit:
            x = np.squeeze(x).reshape(1, -1)
            features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
            features = features.reshape(-1, 1)

            features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
            features = np.sqrt(v_kernel) * features
        else:
            features = x.transpose()

        mean = np.squeeze(np.dot(features.T, nu_t))
        
        lam = 1
        var = lam * np.squeeze(np.dot(np.dot(features.T, Sigma_t_inv), features))

        std = np.sqrt(var)

        return np.squeeze(mean + beta * std)

