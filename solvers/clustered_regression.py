import os
import cvxpy as cp
import pickle
import numpy as np
import pdb
import time
import random
import sys
import torch

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sigmoid
from datetime import datetime

sys.path.insert(1, os.environ['PWD'])
sys.path.insert(1, os.path.join(os.environ['PWD'], 'pytorch'))
sys.path.insert(1, os.path.join(os.environ['PWD'], 'book_problem'))
sys.path.insert(1, os.path.join(os.path.dirname(os.environ['PWD']), 'bookshelf_generator'))
sys.path.insert(1, os.path.join(os.path.dirname(os.environ['PWD']), 'utils'))

from core import Problem, Solver
from pytorch.models import FFNet
from fcn_book_problem_solver_pinned import fcn_book_problem_clustered_solver_pinned
from book_problem_classes import ShelfGeometry


class Regression(Solver):
    def __init__(self, system, problem, prob_features):
        """Constructor for Regression class.

        Args:
            system: string for system name e.g. 'cartpole'
            problem: Problem object for system
            prob_features: list of features to be used
        """
        super().__init__()
        self.system = system
        self.problem = problem
        self.prob_features = prob_features

        self.num_train, self.num_test = 0, 0
        self.model, self.model_fn = None, None

        # training parameters
        self.training_params = {}
        self.training_params['TRAINING_ITERATIONS'] = int(1500)
        self.training_params['BATCH_SIZE'] = 100
        self.training_params['CHECKPOINT_AFTER'] = int(1000)
        self.training_params['SAVEPOINT_AFTER'] = int(30000)
        self.training_params['TEST_BATCH_SIZE'] = 320

    def construct_strategies(self, n_features, train_data, test_data=None):
        """ Reads in training data and constructs strategy dictonary
            TODO(acauligi): be able to compute strategies for test-set too
        """
        self.n_features = n_features  # Should be 17 for the book problem
        self.strategy_dict = {}

        params = train_data[0]
        self.Y = train_data[3]
        features = train_data[1]
        self.num_train = len(self.Y)

        num_probs = self.num_train

        self.n_y = self.Y[0].size

        self.features = np.zeros((num_probs, self.n_features))
        for iter_prob in range(num_probs):
            ff = features[iter_prob].flatten()
            assert self.n_features == len(ff), "From construct strategies: Inconsistent length of feature vector !!"
            self.features[iter_prob, :] = np.array(ff)

        self.labels = np.zeros((num_probs, self.n_y))  # Without policy ID

        for iter_prob in range(num_probs):
            self.labels[iter_prob, :] = self.Y[iter_prob, :]

        print("number of problems {}".format(num_probs))

    def setup_network(self, depth=2, neurons=1000, device_id=0):
        self.device = torch.device('cuda:{}'.format(device_id))

        ff_shape = [self.n_features]
        for ii in range(depth):
            ff_shape.append(neurons)
        ff_shape.append(self.n_y)  # Only output one single integer policy

        self.model = FFNet(ff_shape, activation=torch.nn.ReLU(), drop_p=0.15).to(device=self.device)

        # file names for PyTorch models
        now = datetime.now().strftime('%Y%m%d_%H%M')
        model_fn = 'clustered_regression_{}_{}.pt'
        model_fn = os.path.join(os.getcwd(), model_fn)
        self.model_fn = model_fn.format(self.system, now)

    def load_network(self, fn_regressor_model):
        if os.path.exists(fn_regressor_model):
            print('Loading presaved regression model from {}'.format(fn_regressor_model))
            self.model.load_state_dict(torch.load(fn_regressor_model))
            self.model_fn = fn_regressor_model

    def train(self, verbose=True):
        # grab training params
        TRAINING_ITERATIONS = self.training_params['TRAINING_ITERATIONS']
        BATCH_SIZE = self.training_params['BATCH_SIZE']
        CHECKPOINT_AFTER = self.training_params['CHECKPOINT_AFTER']
        SAVEPOINT_AFTER = self.training_params['SAVEPOINT_AFTER']
        TEST_BATCH_SIZE = self.training_params['TEST_BATCH_SIZE']

        model = self.model

        # X = self.features[:self.num_train]
        # Y = self.labels[:self.num_train]
        X = self.features[:500]
        Y = self.labels[:500]

        # Y = np.ones((6999, 130))*0.0

        # See: https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905/45
        training_loss = torch.nn.BCEWithLogitsLoss()
        opt = optim.Adam(model.parameters(), lr=3e-3, weight_decay=0.001)
        model.train()

        itr = 1
        for epoch in range(TRAINING_ITERATIONS):  # loop over the dataset multiple times
            t0 = time.time()
            running_loss = 0.0
            rand_idx = list(np.arange(0, X.shape[0]-1))
            random.shuffle(rand_idx)

            # Sample all data points
            indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]

            for ii, idx in enumerate(indices):
                # zero the parameter gradients
                opt.zero_grad()

                inputs = Variable(torch.from_numpy(X[idx, :])).float().to(device=self.device)
                y_true = Variable(torch.from_numpy(Y[idx, :])).float().to(device=self.device)

                # forward + backward + optimize
                outputs = model(inputs)

                loss = training_loss(outputs, y_true).float().to(device=self.device)
                loss.backward()
                opt.step()

                # print statistics\n",
                running_loss += loss.item()
                if itr % CHECKPOINT_AFTER == 0:
                    model.eval()
                    rand_idx = list(np.arange(0, X.shape[0]))
                    # random.shuffle(rand_idx)
                    TEST_BATCH_SIZE = len(rand_idx)
                    test_inds = rand_idx[:TEST_BATCH_SIZE]
                    inputs = Variable(torch.from_numpy(X[test_inds, :])).float().to(device=self.device)
                    y_out = Variable(torch.from_numpy(Y[test_inds])).float().to(device=self.device)

                    # forward + backward + optimize
                    outputs = model(inputs)

                    loss = training_loss(outputs, y_out).float().to(device=self.device)
                    outputs = Sigmoid()(outputs).round()

                    accuracy = [float(all(torch.eq(outputs[ii], y_out[ii]))) for ii in range(TEST_BATCH_SIZE)]
                    ss = sum(accuracy)
                    accuracy = np.mean(accuracy)
                    verbose and print("loss:   "+str(loss.item()) + " , acc: " + str(accuracy) + " , summation: " + str(ss))
                    model.train()

                if itr % SAVEPOINT_AFTER == 0:
                    torch.save(model.state_dict(), self.model_fn)
                    verbose and print('Saved model at {}'.format(self.model_fn))
                    # writer.add_scalar('Loss/train', running_loss, epoch)

                itr += 1

            # verbose and print('Done with epoch {} in {}s'.format(epoch, time.time()-t0))

        model.eval()

        torch.save(model.state_dict(), self.model_fn)
        print('Saved model at {}'.format(self.model_fn))

        print('Done training')

    def forward(self, features, solver=cp.GUROBI):  # I only have Gurobi solver
        print("Error: This should not be used !!")

    def forward_book(self, prob_params, features, iter_data, folder_name):

        feature_input = np.array(features.flatten())
        print("Feature is:")
        print(feature_input)

        input = Variable(torch.from_numpy(feature_input)).float().to(device=self.device)

        t0 = time.time()
        out = self.model(input).cpu().detach()

        torch.cuda.synchronize()
        total_time = time.time()-t0
        y_guess = Sigmoid()(out).round().numpy()[:]

        prob_success, cost, optvals = False, np.Inf, None

        bin_width = prob_params['bin_width']
        bin_height = prob_params['bin_height']
        num_of_item = prob_params['num_of_item']

        inst_shelf_geometry = ShelfGeometry(shelf_width=bin_width, shelf_height=bin_height)

        prob_success, cost, solve_time, optvals = fcn_book_problem_clustered_solver_pinned(inst_shelf_geometry,
                                                                                           num_of_item,
                                                                                           features,
                                                                                           [y_guess],
                                                                                           iter_data, folder_name)

        total_time += solve_time
        return prob_success, cost, total_time, optvals
