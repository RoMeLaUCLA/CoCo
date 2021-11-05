import os
import cvxpy as cp
import pickle
import numpy as np
import pdb
import time
import random
import sys
import torch

from datetime import datetime

sys.path.insert(1, os.environ['PWD'])
sys.path.insert(1, os.path.join(os.environ['PWD'], 'pytorch'))
sys.path.insert(1, os.path.join(os.environ['PWD'], 'book_problem'))
sys.path.insert(1, os.path.join(os.path.dirname(os.environ['PWD']), 'bookshelf_generator'))
sys.path.insert(1, os.path.join(os.path.dirname(os.environ['PWD']), 'utils'))

from core import Problem, Solver 
from fcn_book_problem_solver_pinned import fcn_book_problem_clustered_solver_pinned
from book_problem_classes import ShelfGeometry


class KNN(Solver):
    def __init__(self, system, problem, prob_features, knn=5):
        """Constructor for KNN class.

        Args:
            system: string for system name e.g. 'cartpole'
            problem: Problem object for system
            prob_features: list of features to be used
        """
        super().__init__()
        self.system = system
        self.problem = problem
        self.prob_features = prob_features
        self.knn = knn

        self.num_train, self.num_test = 0, 0
        self.model, self.model_fn = None, None

    def train(self, n_features, train_data, test_data=None):
        """ Reads in training data and constructs strategy dictonary
            TODO(acauligi): be able to compute strategies for test-set too
        """
        self.n_features = n_features
        self.strategy_dict = {}

        params = train_data[0]
        self.Y = train_data[3]
        features = train_data[1]
        self.num_train = len(self.Y)
        num_probs = self.num_train

        self.n_y = self.Y[0].size
        self.y_shape = self.Y[0].shape
        self.features = np.zeros((num_probs, self.n_features))
        for iter_prob in range(num_probs):
            ff = features[iter_prob].flatten()
            assert self.n_features == len(ff), "From construct strategies: Inconsistent length of feature vector !!"
            self.features[iter_prob, :] = np.array(ff)

        self.labels = np.zeros((num_probs, 1+self.n_y))
        self.n_strategies = 0

        str_dict = {}
        for ii in range(num_probs):
            y_true = self.Y[ii, :]

            if tuple(y_true) not in self.strategy_dict.keys():
                self.strategy_dict[tuple(y_true)] = np.hstack((self.n_strategies, np.copy(y_true)))
                self.n_strategies += 1

            # strategy_dict is not repeated, self.labels is repeated (corresponding to problems)
            self.labels[ii] = self.strategy_dict[tuple(y_true)]

            idx = int(self.labels[ii, 0])
            if idx in str_dict.keys():
                str_dict[idx] += [ii]
            else:
                str_dict[idx] = [ii]

        self.centroids = np.zeros((self.n_strategies, self.features.shape[1]))
        for ii in range(self.n_strategies):
            self.centroids[ii] = np.mean(self.features[str_dict[ii]], axis=0)

        print("Number of unique strategies is {}".format(len(self.centroids)))

    def forward(self, features, solver=cp.GUROBI):  # I only have Gurobi solver
        print("Error: This should not be used !!")

    def forward_book(self, prob_params, features, iter_data, folder_name):

        feature_input = np.array(features.flatten())
        print("Feature is:")
        print(feature_input)

        t0 = time.time()

        feature_center = torch.from_numpy(feature_input).unsqueeze(0)
        ind_max = torch.argsort(torch.cdist(feature_center, torch.from_numpy(self.centroids))).numpy()[0]
        ind_max = ind_max[:self.knn]

        # For debug
        # ll = len(self.centroids)
        # dist_list = []
        # for iii in range(ll):
        #     dd = np.linalg.norm((feature_input-self.centroids[iii]), 2)
        #     print(dd)
        #     dist_list.append(dd)
        #
        # ret = np.argsort(dist_list)  # IDs of dist_list that would sort the array

        total_time = time.time()-t0

        y_guesses = np.zeros((self.knn, self.n_y), dtype=int)
        for ii, idx in enumerate(ind_max):
            jj = np.where(self.labels[:, 0] == idx)[0][0]
            y_guesses[ii] = self.labels[jj, 1:]

        bin_width = prob_params['bin_width']
        bin_height = prob_params['bin_height']
        num_of_item = prob_params['num_of_item']

        inst_shelf_geometry = ShelfGeometry(shelf_width=bin_width, shelf_height=bin_height)

        prob_success, cost, optvals = False, np.Inf, None
        y_guess_all = y_guesses

        prob_success, cost, solve_time, optvals = fcn_book_problem_clustered_solver_pinned(inst_shelf_geometry,
                                                                                           num_of_item,
                                                                                           features,
                                                                                           y_guess_all,
                                                                                           iter_data, folder_name)

        total_time += solve_time

        return prob_success, cost, total_time, optvals
