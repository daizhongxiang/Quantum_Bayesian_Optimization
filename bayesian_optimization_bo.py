# -*- coding: utf-8 -*-

import numpy as np
from helper_funcs_bo import UtilityFunction, acq_max
import pickle
import itertools
import time

class BO(object):
    def __init__(self, f, pbounds, \
                 log_file=None, beta_t=None, \
                 random_features=None, linear_bandit=False, domain=None, save_interval=500):
        """
        """
        self.save_interval = save_interval
        self.domain = domain
        self.linear_bandit = linear_bandit
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
                          'incumbent_x':[]}

    def init(self, init_points):
        l = [np.random.uniform(x[0], x[1], size=init_points)
             for x in self.bounds]

        self.init_points += list(map(list, zip(*l)))
        y_init = []
        for x in self.init_points:
            y, f_value = self.f(x)
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
            else:
                features = x

            Phi[i, :] = features

        lam = 1
        Sigma_t = np.dot(Phi.T, Phi) + lam * np.identity(M_target)

        Sigma_t_inv = np.linalg.inv(Sigma_t)
        nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), self.Y.reshape(-1, 1))

        x_max = acq_max(ac=self.util_ucb.utility, M=M_target, random_features=self.random_features, \
                        bounds=self.bounds, nu_t=nu_t, Sigma_t_inv=Sigma_t_inv, beta=self.beta_t[len(self.X)-1], \
                       linear_bandit=self.linear_bandit, domain=self.domain)

        for _ in range(n_iter):
            y, f_value = self.f(x_max)
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

            if not self.linear_bandit:
                x = np.squeeze(x_max).reshape(1, -1)
                features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
                features = features / np.sqrt(np.inner(features, features))
                features = np.sqrt(v_kernel) * features
            else:
                features = x_max

            a = features.reshape(-1,1)
            Sigma_t = Sigma_t + np.dot(a, a.transpose())
            Phi = np.concatenate((Phi, features.reshape(1, -1)), axis=0)

            Sigma_t_inv = np.linalg.inv(Sigma_t)
            nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), self.Y.reshape(-1, 1))

            x_max = acq_max(ac=self.util_ucb.utility, M=M_target, random_features=self.random_features, \
                            bounds=self.bounds, nu_t=nu_t, Sigma_t_inv=Sigma_t_inv, beta=self.beta_t[len(self.X)-1], \
                           linear_bandit=self.linear_bandit, domain=self.domain)

            print("iter {0} ------ x_t: {1}, y_t: {2}, f_t: {3}".format(self.i+1, x_max, y, f_value))

            self.i += 1

            x_max_param = self.X[self.Y.argmax(), :-1]

            self.res['max'] = {'max_val': self.Y.max(), 'max_params': dict(zip(self.keys, x_max_param))}
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(self.X[-1])

            if self.log_file is not None and len(self.X) % self.save_interval == 0:
                pickle.dump(self.res, open(self.log_file, "wb"))
