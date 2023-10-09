# -*- coding: utf-8 -*-
import numpy as np
from helper_funcs_quantum import UtilityFunction, acq_max
import pickle
import itertools
import time

class QBO(object):
    def __init__(self, f, pbounds, \
                 log_file=None, beta_t=None, \
                 random_features=None, linear_bandit=False, domain=None):
        """
        """
        self.linear_bandit = linear_bandit
        self.domain = domain

        self.random_features = random_features

        self.log_file = log_file
        self.pbounds = pbounds
        self.incumbent = None
        self.beta_t = beta_t
        
        self.keys = list(pbounds.keys())
        self.dim = len(pbounds)

        self.bounds = []
        for key in self.pbounds.keys():
            self.bounds.append(self.pbounds[key])
        self.bounds = np.asarray(self.bounds)
        
        self.f = f

        self.initialized = False

        self.init_points = []
        self.x_init = []
        self.y_init = []

        self.X = np.array([]).reshape(-1, 1)
        self.Y = np.array([])
        
        self.i = 0

        self.util = None
        
        self.res = {}
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values':[], 'params':[], 'init_values':[], 'init_params':[], 'init':[], \
                          'f_values':[], 'init_f_values':[], 'noise_var_values':[], 'init_noise_var_values':[], \
                          'incumbent_x':[], \
                          'track_queries':[]}

        self.total_used_queries = 0
        self.eps_list = np.array([])
        

    def init(self, init_points):
        l = [np.random.uniform(x[0], x[1], size=init_points)
             for x in self.bounds]

        self.init_points += list(map(list, zip(*l)))
        y_init = []
        for x in self.init_points:
            y, f_value, num_oracle_queries = self.f(x, 1)

            self.total_used_queries += num_oracle_queries
            self.res['all']['track_queries'].append(num_oracle_queries)

            y_init.append(y)
            self.res['all']['init_values'].append(y)
            self.res['all']['init_f_values'].append(f_value)
            self.res['all']['f_values'].append(f_value)
            self.res['all']['init_params'].append(dict(zip(self.keys, x)))

        self.X = np.asarray(self.init_points)
        self.Y = np.asarray(y_init)        

        self.incumbent = np.max(y_init)
        self.initialized = True

        init = {"X":self.X, "Y":self.Y, "f_values":self.res['all']['init_f_values']}
        self.res['all']['init'] = init


    def maximize(self, n_iter=1000, init_points=1):
        self.util_ucb = UtilityFunction()

        if not self.initialized:
            self.init(init_points)

        self.eps_list = np.append(self.eps_list, 1)

        s = self.random_features["s"]
        b = self.random_features["b"]
        obs_noise = self.random_features["obs_noise"]
        v_kernel = self.random_features["v_kernel"]
        M_target = b.shape[0]
        
        if self.linear_bandit:
            M_target = self.dim

        Phi = np.zeros((self.X.shape[0], M_target))
        for i, x in enumerate(self.X):
            if not self.linear_bandit:
                x = np.squeeze(x).reshape(1, -1)
                features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)

                features = features / np.sqrt(np.inner(features, features))
                features = np.sqrt(v_kernel) * features

                features = features * (1 / self.eps_list[i])
            else:
                features = x

            Phi[i, :] = features

        lam = 1
        Sigma_t = np.dot(Phi.T, Phi) + lam * np.identity(M_target)

        Sigma_t_inv = np.linalg.inv(Sigma_t)
        Y_weighted = np.matmul(np.diag(1 / self.eps_list**2), self.Y.reshape(-1, 1))
        nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), Y_weighted)

        x_max = acq_max(ac=self.util_ucb.utility, M=M_target, random_features=self.random_features, \
                        bounds=self.bounds, nu_t=nu_t, Sigma_t_inv=Sigma_t_inv, beta=self.beta_t[len(self.X)-1], \
                       domain=self.domain, linear_bandit=self.linear_bandit)

        if not self.linear_bandit:
            x = np.squeeze(x_max).reshape(1, -1)
            features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
            features = features.reshape(-1, 1)
            features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
            features = np.sqrt(v_kernel) * features # v_kernel is set to be 1 here in the synthetic experiments
        else:
            features = x.reshape(-1, 1)

        lam = 1
        var = lam * np.squeeze(np.dot(np.dot(features.T, Sigma_t_inv), features))
        eps = np.sqrt(var) / np.sqrt(lam)
        
        self.eps_list = np.append(self.eps_list, eps)

        while self.total_used_queries < n_iter:
            y, f_value, num_oracle_queries = self.f(x_max, self.eps_list[-1])
    
            self.total_used_queries += num_oracle_queries
            self.res['all']['track_queries'].append(num_oracle_queries)
    
            self.res['all']['f_values'].append(f_value)

            self.Y = np.append(self.Y, y)
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))

            incumbent_x = self.X[np.argmax(self.Y)]
            self.res['all']['incumbent_x'].append(incumbent_x)
                
            s = self.random_features["s"]
            b = self.random_features["b"]
            obs_noise = self.random_features["obs_noise"]
            v_kernel = self.random_features["v_kernel"]
            M_target = b.shape[0]

            if self.linear_bandit:
                M_target = self.dim

            Phi = np.zeros((self.X.shape[0], M_target))
            for i, x in enumerate(self.X):
                if not self.linear_bandit:
                    x = np.squeeze(x).reshape(1, -1)
                    features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)

                    features = features / np.sqrt(np.inner(features, features))
                    features = np.sqrt(v_kernel) * features

                    features = features * (1 / self.eps_list[i])
                else:
                    features = x

                Phi[i, :] = features

            lam = 1
            Sigma_t = np.dot(Phi.T, Phi) + lam * np.identity(M_target)

            Sigma_t_inv = np.linalg.inv(Sigma_t)
            Y_weighted = np.matmul(np.diag(1 / self.eps_list**2), self.Y.reshape(-1, 1))
            nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), Y_weighted)

            x_max = acq_max(ac=self.util_ucb.utility, M=M_target, random_features=self.random_features, \
                            bounds=self.bounds, nu_t=nu_t, Sigma_t_inv=Sigma_t_inv, beta=self.beta_t[len(self.X)-1], \
                           domain=self.domain, linear_bandit=self.linear_bandit)

            if not self.linear_bandit:
                x = np.squeeze(x_max).reshape(1, -1)
                features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
                features = features.reshape(-1, 1)
                features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
                features = np.sqrt(v_kernel) * features # v_kernel is set to be 1 here in the synthetic experiments
            else:
                features = x.reshape(-1, 1)
            
            lam = 1
            var = lam * np.squeeze(np.dot(np.dot(features.T, Sigma_t_inv), features))
            eps = np.sqrt(var) / np.sqrt(lam)

            self.eps_list = np.append(self.eps_list, eps)

            print("iter {0} ------ x_t: {1}, y_t: {2}".format(self.i+1, x_max, y))

            self.i += 1

            x_max_param = self.X[self.Y.argmax(), :-1]

            self.res['max'] = {'max_val': self.Y.max(), 'max_params': dict(zip(self.keys, x_max_param))}
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(self.X[-1])

            if self.log_file is not None:
                pickle.dump(self.res, open(self.log_file, "wb"))

