#!/usr/bin/env python
# coding: utf-8

# In[168]:

import numpy as np
import cvxpy as cp

import pickle, os, sys


# In[170]:


#load train/test data
relative_path = os.getcwd()
path_ReDUCE = os.path.dirname(relative_path)
sys.path.append(path_ReDUCE + "/utils")
from book_problem_classes import Shelf, ShelfGeometry, Item
dataset_fn = relative_path + '/book_problem/data/'


train_file = open(dataset_fn + '4000_data/train_separated_cluster49_1800data.p','rb')
train_data = pickle.load(train_file)
train_file.close()


# # Load Strategies

# In[171]:


from solvers.clustered_coco_book_problem import CoCo

system = 'book_problem'
prob = []
prob_features = []
coco_obj = CoCo(system, prob, prob_features)

n_features = 17
coco_obj.construct_strategies(n_features, train_data)
print(coco_obj.n_strategies)
#print(coco_obj.strategy_dict)


# # # Setup CoCo
#
# # In[172]:
#
#
# coco_obj.setup_network(device_id=0)
#
# # fn_saved = 'Fin_CoCo_book_problem_16000Data_100_clusters.pt'
# # coco_obj.load_network(fn_saved)
#
# # print("The model is :")
# # print(coco_obj.model)
#
#
# # # Setup VAE
#
# # In[14]:
#
#
# coco_obj.setup_VAE_network(encoder_layer_sizes=[256, 512], decoder_layer_sizes=[512, 256], latent_size=10, device_id=0)
# # vae_saved = 'VAE_book_problem_20210812_0025.pt'
# # coco_obj.load_VAE_network(vae_saved)
#
#
# # # Train CoCo
#
# # In[173]:
#
#
# coco_obj.training_params['TRAINING_ITERATIONS'] = 50
# coco_obj.train(verbose=False, network="CoCo")
#
# # print(coco_obj.model_fn)
#
#
# # # Train VAE
#
# # In[16]:
#
#
# coco_obj.training_params['TRAINING_ITERATIONS'] = 500
# coco_obj.train(verbose=False, network="VAE")
#
#
# # # Train Random Forest
#
# # In[4]:
#
#
# coco_obj.train_random_forest(verbose=False, num_of_train_rf=0)
#
#
# # # Direct Connection to Bookshelf Generator
#
# # In[8]:
#
#
# figure_folder = 'Direct_bookshelf_solving'
# list_prob_success, list_cost, list_total_time, list_n_evals, list_optvals =     coco_obj.forward_direct_bin(num_shelves=200, num_trials=50, folder_name=figure_folder)
#
# print("Success rate is {}".format(sum(list_prob_success)/len(list_prob_success)))
# print("Costs are:")
# print(list_cost)
# print("Average solving time is {}".format(np.average(np.array(list_total_time))))
# print("Max solving time is {}".format(max(list_total_time)))
# print("Numbers of evaluations are {}".format(list_n_evals))
#
#
# # # Original CoCo
#
# # In[174]:
#
#
# test_file = open(dataset_fn+'/4000_data/test_separated_cluster0_fixed.p', 'rb')
# figure_folder = 'final'
# test_data = pickle.load(test_file)
# test_file.close()
#
# test_params = test_data[0]
#
# test_X = test_data[2]
# test_Y = test_data[3]
# test_features = test_data[1]
# test_solve_times = test_data[-1]
# test_costs = test_data[-2]
#
# n_test = len(test_Y)
#
# print("Number of test data : {}".format(n_test))
#
# n_succ = 0
# count = 0
#
# costs_coco = []
# total_time_coco = []
# num_solves_coco = []
#
# cost_ratios_coco = []
# costs_ip = []
# total_time_ip = []
# all_times = []
#
# random_baseline = False  # This is for CoCo. Don't change!
#
# #num_solves_ip = []
#
# success_cases = []
#
# feature_save = []
# cost_save = []
# time_consumed_save = []
#
# #for ii in list(set(range(n_test))-set(solved_cases)):
# for ii in range(min(n_test, 500)):
#     try:
#         print("######################## Solving problem {} #############################".format(ii))
#         features = test_features[ii]
#
#         prob_success, cost, total_time, n_evals, optvals = coco_obj.forward_book(test_params, features, ii,
#                                                                                  num_trials=30,
#                                                                                  random_baseline=random_baseline,
#                                                                                  folder_name=figure_folder)
#
#         all_times.append(total_time)
#
#         count += 1
#         if prob_success:
#
#             # Shift the following 3 lines to other solvers
#             feature_save.append(features)
#             cost_save.append(cost)
#             time_consumed_save.append(total_time)
#             # ============================================
#
#             success_cases.append(ii)
#             n_succ += 1
#             costs_coco += [cost]
#             total_time_coco += [total_time]
#             num_solves_coco += [n_evals]
#
#             true_cost = test_costs[ii]
#             costs_ip += [true_cost]
#             total_time_ip += [test_solve_times[ii]]
#
#             cost_ratios_coco += [cost / true_cost]
#         else:
#
#             # Shift the following 3 lines to other solvers
#             feature_save.append(features)
#             cost_save.append(0)
#             time_consumed_save.append(-1)
#             # ============================================
#
#         print("Successful cases are {}".format(success_cases))
#         print("Actual costs are {}".format(costs_coco))
#         print("True costs are {}".format(costs_ip))
#         coco_percentage = 100 * float(n_succ) / float(count)
#         print("Success rate is {}".format(coco_percentage))
#         print("Average cost detoriate is {}".format(np.average(np.array(costs_coco) - np.array(costs_ip))))
#         print("Solving times are {}".format(total_time_coco))
#
#     except (KeyboardInterrupt, SystemExit):
#         raise
#
#     #except:
#         #print('First: Solver failed at {}'.format(ii))
#         #continue
#
#     if not prob_success:
#         print('Solver failed at {}'.format(ii))
#
#
# # Shift the following 3 lines to other solvers
# solve_path = relative_path + '/solved_data_for_careful_comparison/CoCo_solved.p'
# save_dict = {'features': feature_save, 'cost': cost_save,
#              'time_consumed': time_consumed_save, 'num_of_problem':count}
# with open(solve_path, 'wb') as train_file:
#     pickle.dump(save_dict, train_file)
# # ============================================
#
#
# costs_coco = np.array(costs_coco)
# cost_ratios_coco = np.array(cost_ratios_coco)
# total_time_coco = np.array(total_time_coco)
# num_solves_coco = np.array(num_solves_coco, dtype=int)
#
# costs_ip = np.array(costs_ip)
# total_time_ip = np.array(total_time_ip)
# #num_solves_ip = np.array(num_solves_ip, dtype=int)
#
# coco_percentage = 100 * float(n_succ) / float(count)
#
# print(coco_percentage)
#
# print(success_cases)
#
# print(costs_ip)
# print(costs_coco)
# print("Best cost is {}".format(np.sum(costs_ip)))
# print("Real cost is {}".format(np.sum(costs_coco)))
# print("Average cost detoriate is {}".format(np.average(costs_coco - costs_ip)))
#
# print("Average CoCo solve time is {}".format(np.average(total_time_coco)))
# print("Average MIP solve time is {}".format(np.average(total_time_ip)))
# print("Max CoCo solving time is {}".format(max(total_time_coco)))
# print("Max MIP solving time is {}".format(max(total_time_ip)))
#
#
# # # Random forest
#
# # In[ ]:
#
#
# test_file = open(dataset_fn+'/4000_data/test_separated_cluster0_fixed.p', 'rb')
# figure_folder = 'final'
# test_data = pickle.load(test_file)
# test_file.close()
#
# test_params = test_data[0]
#
# test_X = test_data[2]
# test_Y = test_data[3]
# test_features = test_data[1]
# test_solve_times = test_data[-1]
# test_costs = test_data[-2]
#
# n_test = len(test_Y)
#
# print("Number of test data : {}".format(n_test))
#
# n_succ = 0
# count = 0
#
# costs_coco = []
# total_time_coco = []
# num_solves_coco = []
#
# cost_ratios_coco = []
# costs_ip = []
# total_time_ip = []
# all_times = []
#
# #num_solves_ip = []
#
# success_cases = []
#
# feature_save = []
# cost_save = []
# time_consumed_save = []
#
# for ii in range(min(n_test, 500)):
#     # TODO: if you will remove some data, make sure you adjust the save_dict so the data are not misaligned
#     try:
#         print("######################## Solving problem {} #############################".format(ii))
#         features = test_features[ii]
#
#         prob_success, cost, total_time, n_evals, optvals = coco_obj.forward_book_random_forest(test_params,
#                                                                                                features, ii,
#                                                                                                num_trials=30,
#                                                                                                folder_name=figure_folder)
#
#         all_times.append(total_time)
#         count += 1
#         if prob_success:
#
#             # Shift the following 3 lines to other solvers
#             feature_save.append(features)
#             cost_save.append(cost)
#             time_consumed_save.append(total_time)
#             # ============================================
#
#             success_cases.append(ii)
#
#             n_succ += 1
#             costs_coco += [cost]
#             total_time_coco += [total_time]
#             num_solves_coco += [n_evals]
#
#             true_cost = test_costs[ii]
#             costs_ip += [true_cost]
#             total_time_ip += [test_solve_times[ii]]
#
#             cost_ratios_coco += [cost / true_cost]
#
#         else:
#
#             # Shift the following 3 lines to other solvers
#             feature_save.append(features)
#             cost_save.append(0)
#             time_consumed_save.append(-1)
#             # ============================================
#
#         print("Successful cases are {}".format(success_cases))
#         print("Actual costs are {}".format(costs_coco))
#         print("True costs are {}".format(costs_ip))
#         print("Average cost detoriate is {}".format(np.average(np.array(costs_coco) - np.array(costs_ip))))
#         coco_percentage = 100 * float(n_succ) / float(count)
#         print("Success rate is {}".format(coco_percentage))
#         print("Solving times are {}".format(total_time_coco))
#
#     except (KeyboardInterrupt, SystemExit):
#         raise
#
#     #except:
#         #print('First: Solver failed at {}'.format(ii))
#         #continue
#
#     if not prob_success:
#         print('Solver failed at {}'.format(ii))
#
# # Shift the following 3 lines to other solvers
# solve_path = relative_path + '/solved_data_for_careful_comparison/RF_solved.p'
# save_dict = {'features': feature_save, 'cost': cost_save,
#              'time_consumed': time_consumed_save, 'num_of_problem':count}
# with open(solve_path, 'wb') as train_file:
#     pickle.dump(save_dict, train_file)
# # ============================================
#
# costs_coco = np.array(costs_coco)
# cost_ratios_coco = np.array(cost_ratios_coco)
# total_time_coco = np.array(total_time_coco)
# num_solves_coco = np.array(num_solves_coco, dtype=int)
#
# costs_ip = np.array(costs_ip)
# total_time_ip = np.array(total_time_ip)
# #num_solves_ip = np.array(num_solves_ip, dtype=int)
#
# coco_percentage = 100 * float(n_succ) / float(count)
# print(coco_percentage)
#
# print(success_cases)
#
# print(costs_ip)
# print(costs_coco)
# print("Best cost is {}".format(np.sum(costs_ip)))
# print("Real cost is {}".format(np.sum(costs_coco)))
# print("Average cost detoriate is {}".format(np.average(costs_coco - costs_ip)))
#
# print("Average RF solve time is {}".format(np.average(total_time_coco)))
# print("Average MIP solve time is {}".format(np.average(total_time_ip)))
# print("Max RF solving time is {}".format(max(total_time_coco)))
# print("Max MIP solving time is {}".format(max(total_time_ip)))
#
#
# # # Random baseline
#
# # In[24]:
#
#
# test_file = open(dataset_fn+'/test_separated_cluster0_fixed.p','rb')
# figure_folder = 'cross_validation_0vs5'
#
# test_data = pickle.load(test_file)
# test_file.close()
#
# test_params = test_data[0]
#
# test_X = test_data[2]
# test_Y = test_data[3]
# test_features = test_data[1]
# test_solve_times = test_data[-1]
# test_costs = test_data[-2]
#
# n_test = len(test_Y)
#
# print("Number of test data : {}".format(n_test))
#
# n_succ = 0
# count = 0
#
# costs_coco = []
# total_time_coco = []
# num_solves_coco = []
#
# cost_ratios_coco = []
# costs_ip = []
# total_time_ip = []
# all_times = []
#
# random_baseline = True  # This cell is for random baseline
#
# #num_solves_ip = []
#
# success_cases = []
#
# #for ii in list(set(range(n_test))-set(solved_cases)):
# for ii in range(min(n_test, 500)):
#     try:
#         print("######################## Solving problem {} #############################".format(ii))
#         features = test_features[ii]
#
#         prob_success, cost, total_time, n_evals, optvals = coco_obj.forward_book(test_params, features, ii,
#                                                                                  num_trials=30,
#                                                                                  random_baseline=random_baseline,
#                                                                                  folder_name=figure_folder)
#
#         all_times.append(total_time)
#
#         count += 1
#         if prob_success:
#             success_cases.append(ii)
#             n_succ += 1
#             costs_coco += [cost]
#             total_time_coco += [total_time]
#             num_solves_coco += [n_evals]
#
#             true_cost = test_costs[ii]
#             costs_ip += [true_cost]
#             total_time_ip += [test_solve_times[ii]]
#
#             cost_ratios_coco += [cost / true_cost]
#
#         print("Successful cases are {}".format(success_cases))
#         print("Actual costs are {}".format(costs_coco))
#         print("True costs are {}".format(costs_ip))
#         coco_percentage = 100 * float(n_succ) / float(count)
#         print("Success rate is {}".format(coco_percentage))
#         print("Average cost detoriate is {}".format(np.average(np.array(costs_coco) - np.array(costs_ip))))
#         print("Solving times are {}".format(total_time_coco))
#
#     except (KeyboardInterrupt, SystemExit):
#         raise
#
#     #except:
#         #print('First: Solver failed at {}'.format(ii))
#         #continue
#
#     if not prob_success:
#         print('Solver failed at {}'.format(ii))
#
# costs_coco = np.array(costs_coco)
# cost_ratios_coco = np.array(cost_ratios_coco)
# total_time_coco = np.array(total_time_coco)
# num_solves_coco = np.array(num_solves_coco, dtype=int)
#
# costs_ip = np.array(costs_ip)
# total_time_ip = np.array(total_time_ip)
# #num_solves_ip = np.array(num_solves_ip, dtype=int)
#
# coco_percentage = 100 * float(n_succ) / float(count)
#
# print(coco_percentage)
#
# print(success_cases)
#
# print(costs_ip)
# print(costs_coco)
# print("Best cost is {}".format(np.sum(costs_ip)))
# print("Real cost is {}".format(np.sum(costs_coco)))
# print("Average cost detoriate is {}".format(np.average(costs_coco - costs_ip)))
#
# print("Average Random Baseline solve time is {}".format(np.average(total_time_coco)))
# print("Average MIP solve time is {}".format(np.average(total_time_ip)))
# print("Max Random Baseline solving time is {}".format(max(total_time_coco)))
# print("Max MIP solving time is {}".format(max(total_time_ip)))
#
#
# # # test Regression solver
#
# # In[97]:
#
#
# from solvers.clustered_regression import Regression
#
# system = 'book_problem'
# prob = []
# prob_features = []
# reg_obj = Regression(system, prob, prob_features)
#
# n_features = 17
# reg_obj.construct_strategies(n_features, train_data)
#
#
# # In[113]:
#
#
# reg_obj.setup_network()
#
# # fn_saved = 'models/regression_cartpole_20200708_1029.pt'
# # fn_saved = 'models/regression_cartpole_20210118_1832.pt'
# # reg_obj.load_network(fn_saved)
#
# reg_obj.model
#
#
# # In[141]:
#
#
# reg_obj.training_params['TRAINING_ITERATIONS'] = 20000
# reg_obj.training_params['BATCH_SIZE'] = 1000  # Batch size has to be sufficiently large otherwise acc=0.0
# reg_obj.train(verbose=True)
# print(reg_obj.model_fn)
#
#
# # In[142]:
#
#
# test_file = open(dataset_fn+'/test_separated_AllClusters.p', 'rb')
# figure_folder = 'regression'
# test_data = pickle.load(test_file)
# test_file.close()
#
# test_params = test_data[0]
#
# test_X = test_data[2]
# test_Y = test_data[3]
# test_features = test_data[1]
# test_solve_times = test_data[-1]
# test_costs = test_data[-2]
#
# n_test = len(test_Y)
#
# print("Number of test data : {}".format(n_test))
#
# n_succ = 0
# count = 0
#
# costs_coco = []
# total_time_coco = []
# num_solves_coco = []
#
# cost_ratios_coco = []
# costs_ip = []
# total_time_ip = []
# all_times = []
#
# #num_solves_ip = []
#
# success_cases = []
#
# feature_save = []
# cost_save = []
# time_consumed_save = []
#
# #for ii in list(set(range(n_test))-set(solved_cases)):
# for ii in range(min(n_test, 500)):
#     try:
#         print("######################## Solving problem {} #############################".format(ii))
#         features = test_features[ii]
#
#         prob_success, cost, total_time, optvals = reg_obj.forward_book(test_params, features, ii, folder_name=figure_folder)
#
#         all_times.append(total_time)
#
#         count += 1
#         if prob_success:
#
#             # Shift the following 3 lines to other solvers
#             feature_save.append(features)
#             cost_save.append(cost)
#             time_consumed_save.append(total_time)
#             # ============================================
#
#             success_cases.append(ii)
#             n_succ += 1
#             costs_coco += [cost]
#             total_time_coco += [total_time]
#
#             true_cost = test_costs[ii]
#             costs_ip += [true_cost]
#             total_time_ip += [test_solve_times[ii]]
#
#             cost_ratios_coco += [cost / true_cost]
#         else:
#
#             # Shift the following 3 lines to other solvers
#             feature_save.append(features)
#             cost_save.append(0)
#             time_consumed_save.append(-1)
#             # ============================================
#
#         print("Successful cases are {}".format(success_cases))
#         print("Actual costs are {}".format(costs_coco))
#         print("True costs are {}".format(costs_ip))
#         coco_percentage = 100 * float(n_succ) / float(count)
#         print("Success rate is {}".format(coco_percentage))
#         print("Average cost detoriate is {}".format(np.average(np.array(costs_coco) - np.array(costs_ip))))
#         print("Solving times are {}".format(total_time_coco))
#
#     except (KeyboardInterrupt, SystemExit):
#         raise
#
#     #except:
#         #print('First: Solver failed at {}'.format(ii))
#         #continue
#
#     if not prob_success:
#         print('Solver failed at {}'.format(ii))
#
#
# # Shift the following 3 lines to other solvers
# solve_path = relative_path + '/solved_data_for_careful_comparison/Regressor_solved.p'
# save_dict = {'features': feature_save, 'cost': cost_save,
#              'time_consumed': time_consumed_save, 'num_of_problem':count}
# with open(solve_path, 'wb') as train_file:
#     pickle.dump(save_dict, train_file)
# # ============================================
#
#
# costs_coco = np.array(costs_coco)
# cost_ratios_coco = np.array(cost_ratios_coco)
# total_time_coco = np.array(total_time_coco)
# num_solves_coco = np.array(num_solves_coco, dtype=int)
#
# costs_ip = np.array(costs_ip)
# total_time_ip = np.array(total_time_ip)
# #num_solves_ip = np.array(num_solves_ip, dtype=int)
#
# coco_percentage = 100 * float(n_succ) / float(count)
#
# print(coco_percentage)
#
# print(success_cases)
#
# print(costs_ip)
# print(costs_coco)
# print("Best cost is {}".format(np.sum(costs_ip)))
# print("Real cost is {}".format(np.sum(costs_coco)))
# print("Average cost detoriate is {}".format(np.average(costs_coco - costs_ip)))
#
# print("Average Regressor solve time is {}".format(np.average(total_time_coco)))
# print("Average MIP solve time is {}".format(np.average(total_time_ip)))
# print("Max Regressor solving time is {}".format(max(total_time_coco)))
# print("Max MIP solving time is {}".format(max(total_time_ip)))


# Martius & Zhu (2019)

# In[175]:


from solvers.clustered_knn import KNN


# In[176]:


system = 'book_problem'
prob = []
prob_features = []
knn_obj = KNN(system, prob, prob_features, knn=30)

n_features = 17
knn_obj.train(n_features, train_data)


# In[177]:


test_file = open(dataset_fn+'/4000_data/test_separated_cluster34_fixed.p', 'rb')
figure_folder = 'KNN'
test_data = pickle.load(test_file)
test_file.close()

test_params = test_data[0]

test_X = test_data[2]
test_Y = test_data[3]
test_features = test_data[1]
test_solve_times = test_data[-1]
test_costs = test_data[-2]

n_test = len(test_Y)

print("Number of test data : {}".format(n_test))

n_succ = 0
count = 0

costs_coco = []
total_time_coco = []
num_solves_coco = []

cost_ratios_coco = []
costs_ip = []
total_time_ip = []
all_times = []

#num_solves_ip = []

success_cases = []

feature_save = []
cost_save = []
time_consumed_save = []

#for ii in list(set(range(n_test))-set(solved_cases)):
for ii in range(min(n_test, 500)):
    try:
        print("######################## Solving problem {} #############################".format(ii))
        features = test_features[ii]

        prob_success, cost, total_time, optvals = knn_obj.forward_book(test_params, features, ii, folder_name=figure_folder)

        all_times.append(total_time)

        count += 1
        if prob_success:
            
            # Shift the following 3 lines to other solvers
            feature_save.append(features)
            cost_save.append(cost)
            time_consumed_save.append(total_time)
            # ============================================
            
            success_cases.append(ii)
            n_succ += 1
            costs_coco += [cost]
            total_time_coco += [total_time]

            true_cost = test_costs[ii]
            costs_ip += [true_cost]
            total_time_ip += [test_solve_times[ii]]

            cost_ratios_coco += [cost / true_cost]
        else:
            
            # Shift the following 3 lines to other solvers
            feature_save.append(features)
            cost_save.append(0)
            time_consumed_save.append(-1)
            # ============================================

        print("Successful cases are {}".format(success_cases))
        print("Actual costs are {}".format(costs_coco))
        print("True costs are {}".format(costs_ip))
        coco_percentage = 100 * float(n_succ) / float(count)
        print("Success rate is {}".format(coco_percentage))
        print("Average cost detoriate is {}".format(np.average(np.array(costs_coco) - np.array(costs_ip))))
        print("Solving times are {}".format(total_time_coco))

    except (KeyboardInterrupt, SystemExit):
        raise

    #except:
        #print('First: Solver failed at {}'.format(ii))
        #continue

    if not prob_success:
        print('Solver failed at {}'.format(ii))

        
# Shift the following 3 lines to other solvers
solve_path = relative_path + '/solved_data_for_careful_comparison/Regressor_solved.p'
save_dict = {'features': feature_save, 'cost': cost_save, 
             'time_consumed': time_consumed_save, 'num_of_problem':count}
with open(solve_path, 'wb') as train_file:
    pickle.dump(save_dict, train_file)
# ============================================
            
            
costs_coco = np.array(costs_coco)
cost_ratios_coco = np.array(cost_ratios_coco)
total_time_coco = np.array(total_time_coco)
num_solves_coco = np.array(num_solves_coco, dtype=int)

costs_ip = np.array(costs_ip)
total_time_ip = np.array(total_time_ip)
#num_solves_ip = np.array(num_solves_ip, dtype=int)

coco_percentage = 100 * float(n_succ) / float(count)

print(coco_percentage)

print(success_cases)

print(costs_ip)
print(costs_coco)
print("Best cost is {}".format(np.sum(costs_ip)))
print("Real cost is {}".format(np.sum(costs_coco)))
print("Average cost detoriate is {}".format(np.average(costs_coco - costs_ip)))

print("Average kNN solve time is {}".format(np.average(total_time_coco)))
print("Average MIP solve time is {}".format(np.average(total_time_ip)))
print("Max kNN solving time is {}".format(max(total_time_coco)))
print("Max MIP solving time is {}".format(max(total_time_ip)))


# In[ ]:




