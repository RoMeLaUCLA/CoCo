import os
import cvxpy as cp
import pickle
import numpy as np
import time
import sys
import torch

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sigmoid
from datetime import datetime
from xgboost import XGBRFClassifier
sys.path.insert(1, os.environ['PWD'])
sys.path.insert(1, os.path.join(os.environ['PWD'], 'pytorch'))
sys.path.insert(1, os.path.join(os.environ['PWD'], 'book_problem'))
sys.path.insert(1, os.path.join(os.path.dirname(os.environ['PWD']), 'bookshelf_generator'))
sys.path.insert(1, os.path.join(os.path.dirname(os.environ['PWD']), 'utils'))
from core import Problem, Solver
from pytorch.models import FFNet
from pytorch.vae import VAE
from sklearn.ensemble import RandomForestClassifier
from fcn_book_problem_solver_pinned import fcn_book_problem_clustered_solver_pinned
from bookshelf_generator import main as bin_generation_main
from book_problem_classes import ShelfGeometry
import random
import collections
import pdb


class CoCo(Solver):
    def __init__(self, system, problem, prob_features, n_evals=1):
        """Constructor for CoCo class.

        Args:
            system: string for system name e.g. 'cartpole'
            problem: Problem object for system
            prob_features: list of features to be used
            n_evals: number of strategies attempted to be solved
        """
        super().__init__()
        self.system = system
        self.problem = problem
        self.prob_features = prob_features
        self.n_evals = n_evals

        self.num_train, self.num_test = 0, 0
        self.model, self.model_fn, self.model_vae, self.model_vae_fn = None, None, None, None
        self.rf_model = None

        # training parameters
        self.training_params = {}
        self.training_params['TRAINING_ITERATIONS'] = int(500)
        self.training_params['BATCH_SIZE'] = 20
        self.training_params['CHECKPOINT_AFTER'] = int(1000)
        self.training_params['SAVEPOINT_AFTER'] = int(30000)
        self.training_params['TEST_BATCH_SIZE'] = 32

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

        self.labels = np.zeros((num_probs, 1+self.n_y))
        self.n_strategies = 0

        for iter_prob in range(num_probs):
            # TODO(acauligi): check if transpose necessary with new pickle save format for Y
            y_true = self.Y[iter_prob, :]

            if tuple(y_true) not in self.strategy_dict.keys():
                # Dictionary key = strategy, entry = id + strategy
                self.strategy_dict[tuple(y_true)] = np.hstack((self.n_strategies, np.copy(y_true)))
                self.n_strategies += 1

            # strategy_dict is not repeated, self.labels is repeated (corresponding to problems)
            self.labels[iter_prob] = self.strategy_dict[tuple(y_true)]

        print("number of problems {}".format(num_probs))
        print("Number of strategies {}".format(self.n_strategies))

    def setup_network(self, depth=1, neurons=10000, device_id=0):
        self.device = torch.device('cuda:{}'.format(device_id))

        ff_shape = [self.n_features]
        for ii in range(depth):
            ff_shape.append(neurons)

        ff_shape.append(self.n_strategies)

        print("Neural network structure is:")
        print(ff_shape)

        self.model = FFNet(ff_shape, activation=torch.nn.ReLU(), drop_p=0.15).to(device=self.device)

        # file names for PyTorch models
        now = datetime.now().strftime('%Y%m%d_%H%M')
        model_fn = 'CoCo_{}_{}.pt'
        model_fn = os.path.join(os.getcwd(), model_fn)
        self.model_fn = model_fn.format(self.system, now)

    def setup_VAE_network(self, encoder_layer_sizes=[128, 512], decoder_layer_sizes=[512, 128], latent_size=10, device_id=0):

        self.device = torch.device('cuda:{}'.format(device_id))

        encoder_layer_sizes.insert(0, self.n_features)
        decoder_layer_sizes.append(self.n_y)

        print("VAE layer sizes are ...")
        print(encoder_layer_sizes)
        print(decoder_layer_sizes)

        self.model_vae = VAE(encoder_layer_sizes=encoder_layer_sizes,
                             latent_size=latent_size,
                             decoder_layer_sizes=decoder_layer_sizes,
                             conditional=False,
                             num_labels=0,
                             drop_p=0.2).to(self.device)

        # file names for PyTorch models
        now = datetime.now().strftime('%Y%m%d_%H%M')
        model_vae_fn = 'VAE_{}_{}.pt'
        model_vae_fn = os.path.join(os.getcwd(), model_vae_fn)
        self.model_vae_fn = model_vae_fn.format(self.system, now)

    def load_network(self, fn_classifier_model):
        if os.path.exists(fn_classifier_model):
            print('Loading presaved classifier model from {}'.format(fn_classifier_model))
            self.model.load_state_dict(torch.load(fn_classifier_model))
            self.model_fn = fn_classifier_model

    def load_VAE_network(self, fn_classifier_model):
        if os.path.exists(fn_classifier_model):
            print('Loading presaved classifier model from {}'.format(fn_classifier_model))
            self.model_vae.load_state_dict(torch.load(fn_classifier_model))
            self.model_vae_fn = fn_classifier_model
            print(self.model_vae)

    def train(self, verbose=True, network="CoCo"):

        assert network == "CoCo" or network == "VAE", "Unknown neural network !!!"

        # grab training params
        BATCH_SIZE = self.training_params['BATCH_SIZE']
        TRAINING_ITERATIONS = self.training_params['TRAINING_ITERATIONS']
        BATCH_SIZE = self.training_params['BATCH_SIZE']
        CHECKPOINT_AFTER = self.training_params['CHECKPOINT_AFTER']
        SAVEPOINT_AFTER = self.training_params['SAVEPOINT_AFTER']
        TEST_BATCH_SIZE = self.training_params['TEST_BATCH_SIZE']

        if network == "CoCo":
            model = self.model
        elif network == "VAE":
            model = self.model_vae
        else:
            assert False, "What is going on ??"

        X = self.features[:self.num_train]

        if network == "CoCo":
            Y = self.labels[:self.num_train, 0]  # Get all ids for strategies, just the id number
        elif network == "VAE":
            Y = self.labels[:self.num_train, 1::]  # Actual strategies excluding labels

        print("Printing features and labels")
        np.set_printoptions(threshold=np.inf)
        # print(len(self.features), len(self.labels))
        # print(self.features)

        if network == "CoCo":
            training_loss = torch.nn.CrossEntropyLoss()
        elif network == "VAE":
            training_loss = torch.nn.BCELoss()

        opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.00001)

        model.train()
        itr = 1
        for epoch in range(TRAINING_ITERATIONS):  # loop over the dataset multiple times
            t0 = time.time()
            running_loss = 0.0
            # rand_idx = list(np.arange(0, X.shape[0]-1))  # Why is there a -1 here?
            rand_idx = list(np.arange(0, X.shape[0]))
            random.shuffle(rand_idx)

            # Sample all data points
            indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]

            for ii, idx in enumerate(indices):
                # zero the parameter gradients
                opt.zero_grad()

                inputs = Variable(torch.from_numpy(X[idx, :])).float().to(device=self.device)

                if network == "CoCo":
                    labels = Variable(torch.from_numpy(Y[idx])).long().to(device=self.device)
                elif network == "VAE":
                    labels = Variable(torch.from_numpy(Y[idx, :])).long().to(device=self.device)

                # forward + backward + optimize
                if network == "CoCo":
                    outputs = model(inputs)
                elif network == "VAE":
                    outputs_all = model(inputs)
                    outputs = outputs_all[0]
                    labels = labels.to(torch.float32).to(device=self.device)

                loss = training_loss(outputs, labels).float().to(device=self.device)

                if network == "CoCo":
                    class_guesses = torch.argmax(outputs, 1)
                elif network == "VAE":
                    policy_ids = []
                    class_guesses_list = []
                    for iter_inside_batch in range(len(labels)):
                        policy_output = []
                        for ii in range(self.n_y):
                            dist0 = abs(outputs[iter_inside_batch][ii] - 0)
                            dist1 = abs(outputs[iter_inside_batch][ii] - 1)
                            if dist0 < dist1:
                                policy_output.append(0)
                            else:
                                policy_output.append(1)

                        class_guesses_list.append(policy_output)
                        # Look for the policy number

                        if tuple(policy_output) in self.strategy_dict.keys():
                            policy_ids.append(self.strategy_dict[tuple(policy_output)][0])
                        else:
                            policy_ids.append(-1)
                    class_guesses = torch.FloatTensor(class_guesses_list).to(device=self.device)
                    # print(policy_ids)

                accuracy = torch.mean(torch.eq(class_guesses, labels).float())
                loss.backward()
                #torch.nn.utils.clip_grad_norm(model.parameters(),0.1)
                opt.step()

                # print statistics\n",
                running_loss += loss.item()
                if itr % CHECKPOINT_AFTER == 0:
                    # rand_idx = list(np.arange(0, X.shape[0]-1))  # Why is there a -1 here?
                    rand_idx = list(np.arange(0, X.shape[0]))
                    random.shuffle(rand_idx)
                    test_inds = rand_idx[:TEST_BATCH_SIZE]
                    inputs = Variable(torch.from_numpy(X[test_inds, :])).float().to(device=self.device)

                    if network == "CoCo":
                        labels = Variable(torch.from_numpy(Y[test_inds])).long().to(device=self.device)
                    elif network == "VAE":
                        labels = Variable(torch.from_numpy(Y[test_inds, :])).long().to(device=self.device)

                    if network == "CoCo":
                        outputs = model(inputs)
                    elif network == "VAE":
                        outputs_all = model(inputs)
                        outputs = outputs_all[0]
                        labels = labels.to(torch.float32).to(device=self.device)

                    loss = training_loss(outputs, labels).float().to(device=self.device)

                    if network == "CoCo":
                        class_guesses = torch.argmax(outputs, 1)
                    elif network == "VAE":
                        class_guesses_list = []
                        for iter_inside_batch in range(len(labels)):
                            policy_output = []
                            for ii in range(self.n_y):
                                dist0 = abs(outputs[iter_inside_batch][ii] - 0)
                                dist1 = abs(outputs[iter_inside_batch][ii] - 1)
                                if dist0 < dist1:
                                    policy_output.append(0)
                                else:
                                    policy_output.append(1)

                            class_guesses_list.append(policy_output)

                        class_guesses = torch.FloatTensor(class_guesses_list).to(device=self.device)

                    accuracy = torch.mean(torch.eq(class_guesses, labels).float())
                    print("Accuracy is {}".format(accuracy))
                    verbose and print("loss:   "+str(loss.item())+",   acc:  "+str(accuracy.item()))

                if network == "CoCo":
                    if itr % SAVEPOINT_AFTER == 0:
                        torch.save(model.state_dict(), self.model_fn)
                        verbose and print('Saved model at {}'.format(self.model_fn))
                        # writer.add_scalar('Loss/train', running_loss, epoch)
                elif network == "VAE":
                    if itr % SAVEPOINT_AFTER == 0:
                        torch.save(model.state_dict(), self.model_vae_fn)
                        verbose and print('Saved model at {}'.format(self.model_vae_fn))
                        # writer.add_scalar('Loss/train', running_loss, epoch)

                itr += 1

            verbose and print('Done with epoch {} in {}s'.format(epoch, time.time()-t0))

        if network == "CoCo":
            torch.save(model.state_dict(), self.model_fn)
            print('Saved model at {}'.format(self.model_fn))
        elif network == "VAE":
            print("Saving model :")
            torch.save(model.state_dict(), self.model_vae_fn)
            print('Saved model at {}'.format(self.model_vae_fn))
            print(model)
            print(self.model_vae_fn)

        print('Done training')

        model.eval()

    def train_random_forest(self, num_of_train_rf, verbose=True):

        rand = 0

        self.num_rf_est = 150

        self.rf_model = RandomForestClassifier(
                        criterion='gini',  # gini, entropy
                        n_estimators=self.num_rf_est,
                        random_state=rand, max_depth=75
        )

        # self.rf_model = XGBRFClassifier(n_estimators=self.num_rf_est, subsample=0.9, colsample_bynode=0.2,
        #                                 use_label_encoder=False)

        # train_features = self.features[:num_of_train_rf]
        # train_labels = self.labels[:num_of_train_rf, 0]  # Get all ids for strategies, just the id number
        train_features = self.features
        train_labels = self.labels[:, 0]  # Get all ids for strategies, just the id number

        tt_labels = np.zeros(len(self.labels[:, 0]))
        for iter_ele in range(len(self.labels[:, 0])):
            tt_labels[iter_ele] = self.labels[iter_ele, 0]

        self.rf_model.fit(train_features, tt_labels)

        # Use forest to predict test
        predictions = self.rf_model.predict(train_features)

        # Calculate absolute errors
        errors = np.array(predictions) != np.array(train_labels)
        num_right = int(len(errors) - np.sum(errors))
        total = len(train_labels)

        # Print out mean absolute error
        print(f"Correct predictions vs total: {num_right} / {total}")
        print(f"Percent correct: {num_right / total * 100:.2f} %")

    def forward(self, features, solver=cp.GUROBI):  # I only have Gurobi solver
        print("Error: This should not be used !!")

    def forward_book(self, prob_params, features, iter_data, num_trials, random_baseline, folder_name):

        # TODO: See if the proposed policy makes sense - looks like just some error that causes infeasibility

        feature_input = np.array(features.flatten())

        print("Feature is:")
        print(feature_input)

        input = Variable(torch.from_numpy(feature_input)).float().to(device=self.device)

        time_begin_NN = time.time()

        if random_baseline:
            print("Random sampling !!")
            ind_max = random.sample(range(self.n_strategies), num_trials)
        else:
            scores = self.model(input).cpu().detach().numpy()[:]  # Running the FFNet model

            torch.cuda.synchronize()
            ind_max = np.argsort(scores)[-num_trials:][::-1]

        time_end_NN = time.time()
        time_NN = time_end_NN - time_begin_NN

        print("Ind_max:")
        print(ind_max)

        # scores[np.argsort(scores)] gives correct order
        # [-self.n_evals:] the last 10 (largest) elements, [::-1] reverse direction
        # scores[ind_max] is largest 10 elements from high to low

        y_guesses = np.zeros((num_trials, self.n_y), dtype=int)

        # There is a lot of repeats in the self.labels
        # number of integer variables: 4 (per round) x 10 round (horizon - 1)
        num_probs = self.num_train

        # len(self.labels) == self.num_train !!

        for ii, idx in enumerate(ind_max):
            for jj in range(num_probs):
                # first index of training label is that strategy's idx
                label = self.labels[jj]
                if label[0] == idx:
                    # remainder of training label is that strategy's binary pin
                    y_guesses[ii] = label[1:]  # y guesses corresponding to top 10 different scores
                    break

        prob_success, cost, n_evals, optvals = False, np.Inf, len(y_guesses), None

        bin_width = prob_params['bin_width']
        bin_height = prob_params['bin_height']
        num_of_item = prob_params['num_of_item']

        inst_shelf_geometry = ShelfGeometry(shelf_width=bin_width, shelf_height=bin_height)

        y_guess_all = y_guesses

        begin_time = time.time()
        prob_success, cost, solve_time, optvals = fcn_book_problem_clustered_solver_pinned(inst_shelf_geometry,
                                                                                           num_of_item,
                                                                                           features,
                                                                                           y_guess_all,
                                                                                           iter_data, folder_name)

        pass_time_with_runpy = time.time() - begin_time
        print("success? {}".format(prob_success))
        total_time = solve_time + time_NN

        return prob_success, cost, total_time, n_evals, optvals

    def forward_direct_bin(self, num_shelves, num_trials, folder_name):

        new_data = bin_generation_main(num_shelves)

        inst_shelf_geometry = new_data[0]["after"]["shelf"].shelf_geometry

        num_of_item_stored = new_data[0]["after"]["shelf"].num_of_item

        list_prob_success = []
        list_cost = []
        list_total_time = []
        list_n_evals = []
        list_optvals = []

        for iter_data in range(len(new_data)):

            print("###################################### Data number {} ######################################".format(iter_data))

            this_shelf = new_data[iter_data]["after"]["shelf"]
            feature = this_shelf.return_feature()
            feature_input = np.array(feature.flatten())

            print("Feature is:")
            print(feature_input)

            input = Variable(torch.from_numpy(feature_input)).float().to(device=self.device)

            time_begin_NN = time.time()

            scores = self.model(input).cpu().detach().numpy()[:]  # Running the FFNet model

            torch.cuda.synchronize()
            ind_max = np.argsort(scores)[-num_trials:][::-1]

            time_end_NN = time.time()
            time_NN = time_end_NN - time_begin_NN

            print("Ind_max:")
            print(ind_max)

            # scores[np.argsort(scores)] gives correct order
            # [-self.n_evals:] the last 10 (largest) elements, [::-1] reverse direction
            # scores[ind_max] is largest 10 elements from high to low

            y_guesses = np.zeros((num_trials, self.n_y), dtype=int)

            # There is a lot of repeats in the self.labels
            # number of integer variables: 4 (per round) x 10 round (horizon - 1)
            num_probs = self.num_train

            # len(self.labels) == self.num_train !!

            for ii, idx in enumerate(ind_max):
                for jj in range(num_probs):
                    # first index of training label is that strategy's idx
                    label = self.labels[jj]
                    if label[0] == idx:
                        # remainder of training label is that strategy's binary pin
                        y_guesses[ii] = label[1:]  # y guesses corresponding to top 10 different scores
                        break

            prob_success, cost, n_evals, optvals = False, np.Inf, len(y_guesses), None

            y_guess_all = y_guesses

            begin_time = time.time()
            prob_success, cost, solve_time, optvals = fcn_book_problem_clustered_solver_pinned(inst_shelf_geometry,
                                                                                               num_of_item_stored,
                                                                                               feature,
                                                                                               y_guess_all,
                                                                                               iter_data, folder_name)

            pass_time_with_runpy = time.time() - begin_time
            print("success? {}".format(prob_success))
            total_time = solve_time + time_NN

            list_prob_success.append(prob_success)
            list_cost.append(cost)
            list_total_time.append(total_time)
            list_n_evals.append(n_evals)
            list_optvals.append(optvals)

        return list_prob_success, list_cost, list_total_time, list_n_evals, list_optvals


    def forward_book_VAE(self, prob_params, features, iter_data, num_trials, folder_name):

        feature_input = np.array(features.flatten())

        print("Feature is:")
        print(feature_input)

        input = Variable(torch.from_numpy(feature_input)).float().to(device=self.device)

        total_time = 0.0
        n_evals = 0
        while n_evals < num_trials:
            print("====================== Running iteration {} =====================".format(n_evals))

            t0 = time.time()
            y_guess_raw = self.model_vae(input)[0].cpu().detach().numpy()[:]  # Running the VAE model
            torch.cuda.synchronize()
            total_time += time.time()-t0

            # print("y_guess_raw:")
            # print(y_guess_raw)

            y_guess = []
            for ii in range(self.n_y):
                dist0 = abs(y_guess_raw[ii] - 0)
                dist1 = abs(y_guess_raw[ii] - 1)
                if dist0 < dist1:
                    y_guess.append(0)
                else:
                    y_guess.append(1)

            print("y guesses:")
            print(y_guess)

            # try:
            #     self.print_policy_choice(y_guess, num_of_item=3)  # Verify if the policy is valid
            # except:
            #     continue

            bin_width = prob_params['bin_width']
            bin_height = prob_params['bin_height']
            num_of_item = prob_params['num_of_item']

            inst_shelf_geometry = ShelfGeometry(shelf_width=bin_width, shelf_height=bin_height)

            prob_success = False
            cost = 0
            optvals = 0

            try:
                prob_success, cost, solve_time, optvals = fcn_book_problem_clustered_solver_pinned(inst_shelf_geometry,
                    num_of_item, features, y_guess, iter_data, folder_name)
            except:
                pass

            total_time += solve_time
            n_evals += 1
            print("success? {}".format(prob_success))
            if prob_success:
                break

        return prob_success, cost, total_time, n_evals, optvals


    def forward_book_random_forest(self, prob_params, features, iter_data, num_trials, folder_name):

        feature_input = np.array(features.flatten())

        print("Feature is:")
        print(feature_input)

        t0 = time.time()

        # scores = self.rf_model.predict(np.array([feature_input]))  # Single prediction

        # return multiple policies
        time_begin_rf = time.time()
        list_all_predictions = []
        for iter_est in range(self.num_rf_est):
            # print("Estimator {} predicts {}".format(iter_est, rf_model.estimators_[iter_est].predict([feature_input])))
            list_all_predictions.append(self.rf_model.estimators_[iter_est].predict([feature_input])[0])

        # print("All predictions: {}".format(list_all_predictions))
        scores = self.get_top_policies(list_all_predictions, num_samples=num_trials)
        time_end_rf = time.time()
        time_rf = time_end_rf - time_begin_rf

        total_time = time.time()-t0
        # ind_max = np.argsort(scores)[-self.n_evals:][::-1]

        ind_max = scores

        print("Ind_max:")
        print(ind_max)

        # scores[np.argsort(scores)] gives correct order
        # [-self.n_evals:] the last 10 (largest) elements, [::-1] reverse direction
        # scores[ind_max] is largest 10 elements from high to low

        y_guesses = np.zeros((num_trials, self.n_y), dtype=int)

        # There is a lot of repeats in the self.labels
        # number of integer variables: 4 (per round) x 10 round (horizon - 1)
        num_probs = self.num_train

        # len(self.labels) == self.num_train !!

        for ii, idx in enumerate(ind_max):
            for jj in range(num_probs):
                # first index of training label is that strategy's idx
                label = self.labels[jj]
                if label[0] == idx:
                    # remainder of training label is that strategy's binary pin
                    y_guesses[ii] = label[1:]  # y guesses corresponding to top 10 different scores
                    break

        prob_success, cost, n_evals, optvals = False, np.Inf, len(y_guesses), None

        # Begin: shared setup
        bin_width = prob_params['bin_width']
        bin_height = prob_params['bin_height']
        num_of_item = prob_params['num_of_item']

        inst_shelf_geometry = ShelfGeometry(shelf_width=bin_width, shelf_height=bin_height)

        y_guess_all = y_guesses

        begin_time = time.time()
        prob_success, cost, solve_time, optvals = fcn_book_problem_clustered_solver_pinned(inst_shelf_geometry,
                                                                                           num_of_item,
                                                                                           features,
                                                                                           y_guess_all,
                                                                                           iter_data, folder_name)

        pass_time_with_runpy = time.time() - begin_time
        print("success? {}".format(prob_success))
        total_time = solve_time + time_rf
        # End: shared setup

        return prob_success, cost, total_time, n_evals, optvals

    @ staticmethod
    def print_policy_choice(y, num_of_item):
        num_stored_item_pairs = int(round(num_of_item * (num_of_item - 1) / 2))
        num_of_pairs = int(round(num_of_item * (num_of_item + 1) / 2))

        print("=======================================================================================================")
        # Item states
        item_state = []
        for iter_item in range(num_of_item):
            int_state = y[(iter_item*5+0):(iter_item*5+5)]
            if not isinstance(int_state, list):
                int_state.tolist()
            if all(int_state == [1, 0, 0, 0, 0]):
                item_state.append("Left horizontal")
            elif all(int_state == [0, 1, 0, 0, 0]):
                item_state.append("Left tilt")
            elif all(int_state == [0, 0, 1, 0, 0]):
                item_state.append("Vertical")
            elif all(int_state == [0, 0, 0, 1, 0]):
                item_state.append("Right tilt")
            elif all(int_state == [0, 0, 0, 0, 1]):
                item_state.append("Right horizontal")
            else:
                assert False, "Something wrong with item state"

        print("Proposed item states are {}".format(item_state))

        # Item orientation
        item_orientation = []
        for iter_item in range(num_of_item):
            int_orientation = y[num_of_item*5+iter_item*2]
            if int_orientation == 1:  # Tilt to left is 1
                item_orientation.append("Left angled")
            elif int_orientation == 0:  # Tilt to right is 1
                item_orientation.append("Right angled")
            else:
                assert False, "Something wrong with item angle"

        print("Proposed item angles are {}".format(item_orientation))

        # num_of_item x num_of_int_item_state + num_of_item x 2 (int_orientation) + 3 (stored item pairs)
        sign_rel_stored = [y[num_of_item * 5 + num_of_item * 2 + num_stored_item_pairs + 0],
                           y[num_of_item * 5 + num_of_item * 2 + num_stored_item_pairs + 1],
                           y[num_of_item * 5 + num_of_item * 2 + num_stored_item_pairs + 2]]
        # print("3 signs relative to stored items are {}".format(sign_rel_stored))
        if sign_rel_stored == [0, 0, 0]:
            pos = 0
        elif sign_rel_stored == [1, 0, 0]:
            pos = 1
        elif sign_rel_stored == [1, 1, 0]:
            pos = 2
        elif sign_rel_stored == [1, 1, 1]:
            pos = 3
        else:
            assert False, "Something wrong with the output position"
        print("The position of insertion is {}".format(pos))

        # y component of a
        a_y_comp = []
        for iter_pair in range(num_of_pairs):
            ayy = y[num_of_item * 5 + num_of_item * 2 + num_stored_item_pairs + num_of_item + iter_pair]
            if ayy == 1:
                a_y_comp.append("up")  # 1 is positive y component
            elif ayy == 0:
                a_y_comp.append("down")  # 0 is negative y component
            else:
                assert False, "Something wrong with the y component of separating plane"
        print("Proposed vertical direction for planes {}".format(a_y_comp))

        # Vertex region
        num_of_int_per_vertex = 2
        vertices_pos = []
        for iter_item in range(num_of_item+1):
            vertices_pos.append([])
            for iter_dim in range(2):
                vertices_pos[iter_item].append([])
                for iter_vertex in range(4):
                    int_before = iter_item*2*4*num_of_int_per_vertex + iter_dim*4*num_of_int_per_vertex + iter_vertex*num_of_int_per_vertex
                    int_this_vertex = [y[num_of_item * 5 + num_of_item * 2 + num_stored_item_pairs + num_of_item + num_of_pairs + int_before],
                                       y[num_of_item * 5 + num_of_item * 2 + num_stored_item_pairs + num_of_item + num_of_pairs + int_before+1]]

                    if iter_dim == 0:  # x component
                        if int_this_vertex == [0, 0]:
                            vertices_pos[iter_item][iter_dim].append("Left")
                        elif int_this_vertex == [1, 0]:
                            vertices_pos[iter_item][iter_dim].append("Middle left")
                        elif int_this_vertex == [0, 1]:
                            vertices_pos[iter_item][iter_dim].append("Middle right")
                        elif int_this_vertex == [1, 1]:
                            vertices_pos[iter_item][iter_dim].append("Right")
                        else:
                            assert False, "Something is wrong with the integer of vertices"

                    elif iter_dim == 1:
                        if int_this_vertex == [0, 0]:
                            vertices_pos[iter_item][iter_dim].append("Down")
                        elif int_this_vertex == [1, 0]:
                            vertices_pos[iter_item][iter_dim].append("Middle down")
                        elif int_this_vertex == [0, 1]:
                            vertices_pos[iter_item][iter_dim].append("Middle up")
                        elif int_this_vertex == [1, 1]:
                            vertices_pos[iter_item][iter_dim].append("Up")
                        else:
                            assert False, "Something is wrong with the integer of vertices"

                    else:
                        assert False, "Something wrong with the integer of vertices"

        for iter_item in range(num_of_item+1):
            print("Item {}, x component, 4 vertices are located at {}, {}, {}, {}".format(iter_item,
                                                                                          vertices_pos[iter_item][0][0],
                                                                                          vertices_pos[iter_item][0][1],
                                                                                          vertices_pos[iter_item][0][2],
                                                                                          vertices_pos[iter_item][0][3]))

            print("Item {}, y component, 4 vertices are located at {}, {}, {}, {}".format(iter_item,
                                                                                          vertices_pos[iter_item][1][0],
                                                                                          vertices_pos[iter_item][1][1],
                                                                                          vertices_pos[iter_item][1][2],
                                                                                          vertices_pos[iter_item][1][3]))

        print("=======================================================================================================")

    @staticmethod
    def get_top_policies(list_all_predictions, num_samples):
        """
        Get the most voted policies for random forest classifier
        Args:
            num_samples:

        Returns:

        """
        assert len(set(list_all_predictions)) >= num_samples, "Error: Cannot return more policies than provided !!"

        frequency = collections.Counter(list_all_predictions)
        dict_frequency = dict(frequency)
        # print("Frequency dictionary: {}".format(dict_frequency))

        list_frequency = list(dict_frequency.values())
        list_scores = list(dict_frequency.keys())
        # print("----------------------------------------------")
        # print("Frequencies: {}".format(list_frequency))
        # print("Scores: {}".format(list_scores))

        assert len(list_frequency) == len(list_scores), "Inconsistent length for frequencies and scores !!"

        index_sort = np.argsort(list_frequency)
        # print("Index sort: {}".format(index_sort))

        list_frequency_sorted = [list_frequency[index_sort[ii]] for ii in range(len(index_sort))]
        list_scores_sorted = [list_scores[index_sort[ii]] for ii in range(len(index_sort))]

        # print("----------------------------------------------")
        # print("Sorted frequencies: {}".format(list_frequency_sorted))
        # print("Sorted scores: {}".format(list_scores_sorted))

        ret_scores = []

        for iter_sample in range(num_samples):
            fr = list_frequency_sorted[-1]  # Get the last element of the sorted frequency array
            ss = list_scores_sorted[-1]  # Get the key from the frequency
            ret_scores.append(ss)
            list_frequency_sorted.pop()
            list_scores_sorted.pop()
            # print("-------------------------------")
            # print(iter_sample, fr, ss)
            # print(list_frequency_sorted)
            # print(list_scores_sorted)

        return ret_scores
