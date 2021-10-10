#!/usr/bin/env python
# coding: utf-8

# In[12]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import cvxpy as cp

import pickle, os

import pdb

# In[13]:


# load train/test data
relative_path = os.getcwd()
dataset_fn = relative_path + '/book_problem/data/'

train_file = open(dataset_fn + 'train_separated_cluster68_3000data.p', 'rb')
train_data = pickle.load(train_file)
train_file.close()


# # Load Strategies

# In[14]:


from solvers.clustered_coco_book_problem import CoCo

system = 'book_problem'
prob = []
prob_features = []
coco_obj = CoCo(system, prob, prob_features)

n_features = 17
coco_obj.construct_strategies(n_features, train_data)
print(coco_obj.n_strategies)
#print(coco_obj.strategy_dict)


# Setup CoCo

# In[15]:


coco_obj.setup_network(device_id=0)

# fn_saved = 'Fin_CoCo_book_problem_16000Data_100_clusters.pt'
# coco_obj.load_network(fn_saved)
#
# print("The model is :")
# print(coco_obj.model)


# # # Setup VAE
#
# # In[36]:
#
#
# coco_obj.setup_VAE_network(encoder_layer_sizes=[256, 512], decoder_layer_sizes=[512, 256], latent_size=10, device_id=0)
# # vae_saved = 'VAE_book_problem_20210812_0025.pt'
# # coco_obj.load_VAE_network(vae_saved)


# Train CoCo

# In[16]:


coco_obj.training_params['TRAINING_ITERATIONS'] = 30
coco_obj.train(verbose=False, network="CoCo")

print(coco_obj.model_fn)

pdb.set_trace()

# # # Train VAE
#
# # In[37]:
#
#
# coco_obj.training_params['TRAINING_ITERATIONS'] = 500
# coco_obj.train(verbose=False, network="VAE")


# # Train Random Forest

# In[6]:


# coco_obj.train_random_forest(verbose=False, num_of_train_rf=0)
# pdb.set_trace()


# # Original CoCo

# In[17]:


test_file = open(dataset_fn+'/test_cluster68_fixed.p', 'rb')
figure_folder = 'cross_validation_0vs5'
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

random_baseline = False  # This is for CoCo. Don't change!

#num_solves_ip = []

success_cases = []

#for ii in list(set(range(n_test))-set(solved_cases)):
for ii in range(min(n_test, 500)):
    try:
        print("######################## Solving problem {} #############################".format(ii))
        features = {'item_width_stored': test_features['item_width_stored'][ii],
                    'item_height_stored': test_features['item_height_stored'][ii],
                    'item_center_x_stored': test_features['item_center_x_stored'][ii],
                    'item_center_y_stored': test_features['item_center_y_stored'][ii],
                    'item_angle_stored': test_features['item_angle_stored'][ii],
                    'item_width_in_hand': test_features['item_width_in_hand'][ii],
                    'item_height_in_hand': test_features['item_height_in_hand'][ii]}

        prob_success, cost, total_time, n_evals, optvals = coco_obj.forward_book(test_params, features, ii,
                                                                                 num_trials=30,
                                                                                 random_baseline=random_baseline,
                                                                                 folder_name=figure_folder)

        all_times.append(total_time)

        count += 1
        if prob_success:
            success_cases.append(ii)
            n_succ += 1
            costs_coco += [cost]
            total_time_coco += [total_time]
            num_solves_coco += [n_evals]

            true_cost = test_costs[ii]
            costs_ip += [true_cost]
            total_time_ip += [test_solve_times[ii]]

            cost_ratios_coco += [cost / true_cost]

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

print("Average CoCo solve time is {}".format(np.average(total_time_coco)))
print("Average MIP solve time is {}".format(np.average(total_time_ip)))
print("Max CoCo solving time is {}".format(max(total_time_coco)))
print("Max MIP solving time is {}".format(max(total_time_ip)))

pdb.set_trace()


# # Random forest

# In[8]:


# test_file = open(dataset_fn+'/test_cluster68_fixed.p', 'rb')
# figure_folder = 'cross_validation_0vs5'
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
# for ii in range(min(n_test, 500)):
#     try:
#         print("######################## Solving problem {} #############################".format(ii))
#         features = {'item_width_stored': test_features['item_width_stored'][ii],
#                     'item_height_stored': test_features['item_height_stored'][ii],
#                     'item_center_x_stored': test_features['item_center_x_stored'][ii],
#                     'item_center_y_stored': test_features['item_center_y_stored'][ii],
#                     'item_angle_stored': test_features['item_angle_stored'][ii],
#                     'item_width_in_hand': test_features['item_width_in_hand'][ii],
#                     'item_height_in_hand': test_features['item_height_in_hand'][ii]}
#
#         prob_success, cost, total_time, n_evals, optvals = coco_obj.forward_book_random_forest(test_params,
#                                                                                                features, ii,
#                                                                                                num_trials=30,
#                                                                                                folder_name=figure_folder)
#
#         all_times.append(total_time)
#         count += 1
#         if prob_success:
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
# pdb.set_trace()

# # Random baseline

# In[34]:


# test_file = open(dataset_fn+'/test_cluster68_fixed.p', 'rb')
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
#         features = {'item_width_stored': test_features['item_width_stored'][ii],
#                     'item_height_stored': test_features['item_height_stored'][ii],
#                     'item_center_x_stored': test_features['item_center_x_stored'][ii],
#                     'item_center_y_stored': test_features['item_center_y_stored'][ii],
#                     'item_angle_stored': test_features['item_angle_stored'][ii],
#                     'item_width_in_hand': test_features['item_width_in_hand'][ii],
#                     'item_height_in_hand': test_features['item_height_in_hand'][ii]}
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

pdb.set_trace()

# # test Regression solver

# In[33]:


from solvers.regression import Regression

system = 'cartpole'
prob_features = ['x0', 'xg']
reg_obj = Regression(system, prob, prob_features)

n_features = 8
reg_obj.construct_strategies(n_features, train_data)


# In[ ]:


reg_obj.setup_network()

fn_saved = 'models/regression_cartpole_20200708_1029.pt'
fn_saved = 'models/regression_cartpole_20210118_1832.pt'
reg_obj.load_network(fn_saved)

reg_obj.model


# In[ ]:


reg_obj.training_params['TRAINING_ITERATIONS'] = 500
reg_obj.train(verbose=False)
print(reg_obj.model_fn)


# In[ ]:


n_succ = 0
count = 0

costs_reg = []
total_time_reg = []
num_solves_reg = []

cost_ratios_reg = []

for ii in range(n_test):
    prob_params = {}
    for k in p_test.keys():
        prob_params[k] = p_test[k][ii]

    try:
        prob_success, cost, total_time, optvals = reg_obj.forward(prob_params)

        count += 1
        if prob_success:
            n_succ += 1
            costs_reg += [cost]
            total_time_reg += [total_time]
            num_solves_reg += [1]

            true_cost = c_test[ii]
            cost_ratios_reg += [cost / true_cost]
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        print('Solver failed at {}'.format(ii))
        continue

costs_reg = np.array(costs_reg)
cost_ratios_reg = np.array(cost_ratios_reg)
total_time_reg = np.array(total_time_reg)
num_solves_reg = np.array(num_solves_reg, dtype=int)

reg_percentage = 100 * float(n_succ) / float(count)
reg_percentage


# # Martius & Zhu (2019)

# In[53]:


from solvers.knn import KNN


# In[54]:


system = 'cartpole'
prob_features = ['x0', 'xg']

knn_obj = KNN(system, prob, prob_features, knn=coco_obj.n_evals)

n_features = 8
knn_obj.train(n_features, train_data)


# In[55]:


n_succ = 0
count = 0

costs_knn = []
total_time_knn = []
num_solves_knn = []

cost_ratios_knn = []

for ii in range(n_test):
    if ii % 1000 == 0:
        print('{} / {}'.format(ii,n_test))
    prob_params = {}
    for k in p_test.keys():
        prob_params[k] = p_test[k][ii]

    try:
        prob_success, cost, total_time, n_evals, optvals = knn_obj.forward(prob_params, solver=cp.GUROBI)

        count += 1
        if prob_success:
            n_succ += 1
            costs_knn += [cost]
            total_time_knn += [total_time]
            num_solves_knn += [n_evals]

            true_cost = c_test[ii]
            cost_ratios_knn += [cost / true_cost]
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        print('Solver failed at {}'.format(ii))
        continue

costs_knn = np.array(costs_knn)
cost_ratios_knn = np.array(cost_ratios_knn)
total_time_knn = np.array(total_time_knn)
num_solves_knn = np.array(num_solves_knn, dtype=int)

knn_percentage = 100 * float(n_succ) / float(count)
knn_percentage


# # Mosek
# 
# ### Make sure set value for max number of feasible solutions for solver

# In[56]:


print('Cutoff time for Mosek: {}'.format(np.max(total_time_coco)))


# In[57]:


solver = cp.GUROBI

n_succ = 0
count = 0

costs_sol = []
cost_ratios_sol = []
total_time_sol = []

for ii in range(n_test):
    count += 1
    if ii % 1000 == 0:
        print('{} / {}'.format(ii,n_test))
    prob_params = {}
    for k in p_test.keys():
        prob_params[k] = p_test[k][ii]

    try:
        prob_success, cost, total_time, optvals = prob.solve_micp(prob_params, solver=solver)

        if prob_success:
            n_succ += 1
            costs_sol += [cost]

            true_cost = c_test[ii]
            cost_ratios_sol += [cost / true_cost]
            total_time_sol += [total_time]
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
#         print('Solver failed at {}'.format(ii))
        continue

costs_sol = np.array(costs_sol)
cost_ratios_sol = np.array(cost_ratios_sol)
total_time_sol = np.array(total_time_sol)

sol_percentage = 100 * float(n_succ) / float(count)
sol_percentage


# # Results

# In[58]:


import math
import seaborn as sns
sns.set(font_scale=2., font="serif", style="whitegrid")
import pandas as pd
import h5py
import matplotlib

flierprops = {'alpha':0.2}

params = {
#     'backend': 'ps',
#               'text.latex.preamble': ['\\usepackage{gensymb}'],
#               'axes.labelsize': 12, # fontsize for x and y labels (was 12 and before 10)
#               'axes.titlesize': 12,
#               'font.size': 12, # was 12 and before 10
              'legend.fontsize': 26, # was 12 and before 10
#               'xtick.labelsize': 12,
#               'ytick.labelsize': 12,
#               'text.usetex': True,
#               'font.family': 'serif',
#               'font.sans-serif':['Helvetica Neue']
    }

sns.set(font_scale=2.5)
matplotlib.rcParams.update(params)


# In[59]:


# hf = h5py.File('cartpole_dev.h5', 'w')

# hf.create_dataset('percentage', data=np.expand_dims(np.asarray([sol_percentage, mlopt_percentage, reg_percentage, knn_percentage]), axis=1))

# num_solves_grp = hf.create_group('num_solves')
# num_solves_grp.create_dataset('num_solves_mlopt', data=np.expand_dims(num_solves_mlopt, axis=1))
# num_solves_grp.create_dataset('num_solves_reg', data=np.expand_dims(num_solves_reg, axis=1))
# num_solves_grp.create_dataset('num_solves_knn', data=np.expand_dims(num_solves_knn, axis=1))

# total_time_grp = hf.create_group('total_time')
# total_time_grp.create_dataset('total_time_sol', data=np.expand_dims(total_time_sol, axis=1))
# total_time_grp.create_dataset('total_time_mlopt', data=np.expand_dims(total_time_mlopt, axis=1))
# total_time_grp.create_dataset('total_time_reg', data=np.expand_dims(total_time_reg, axis=1))
# total_time_grp.create_dataset('total_time_knn', data=np.expand_dims(total_time_knn, axis=1))

# cost_ratios_grp = hf.create_group('cost_ratios')
# cost_ratios_grp.create_dataset('cost_ratios_sol', data=np.expand_dims(cost_ratios_sol, axis=1))
# cost_ratios_grp.create_dataset('cost_ratios_mlopt', data=np.expand_dims(cost_ratios_mlopt, axis=1))
# cost_ratios_grp.create_dataset('cost_ratios_reg', data=np.expand_dims(cost_ratios_reg, axis=1))
# cost_ratios_grp.create_dataset('cost_ratios_knn', data=np.expand_dims(cost_ratios_knn, axis=1))

# hf.close()


# In[60]:


hf = h5py.File('cartpole_dev.h5', 'r')

num_solves_grp = hf.get('num_solves')
num_solves_coco = np.squeeze(np.array(num_solves_grp.get('num_solves_mlopt')))
num_solves_reg = np.squeeze(np.array(num_solves_grp.get('num_solves_reg')))
num_solves_knn = np.squeeze(np.array(num_solves_grp.get('num_solves_knn')))

total_time_grp = hf.get('total_time')
total_time_sol = np.ndarray.tolist(np.squeeze(np.array(total_time_grp.get('total_time_sol'))))
total_time_coco = np.ndarray.tolist(np.squeeze(np.array(total_time_grp.get('total_time_mlopt'))))
total_time_reg = np.ndarray.tolist(np.squeeze(np.array(total_time_grp.get('total_time_reg'))))
total_time_knn = np.ndarray.tolist(np.squeeze(np.array(total_time_grp.get('total_time_knn'))))

cost_ratios_grp = hf.get('cost_ratios')
cost_ratios_sol = np.squeeze(np.array(cost_ratios_grp.get('cost_ratios_sol')))
cost_ratios_coco = np.squeeze(np.array(cost_ratios_grp.get('cost_ratios_mlopt')))
cost_ratios_reg = np.squeeze(np.array(cost_ratios_grp.get('cost_ratios_reg')))
cost_ratios_knn = np.squeeze(np.array(cost_ratios_grp.get('cost_ratios_knn')))

sol_percentage, coco_percentage, reg_percentage, knn_percentage = np.ndarray.tolist(np.squeeze(np.array(hf.get('percentage'))))

hf.close()


# In[52]:


results = {'Mosek':[sol_percentage], 'CoCo':[coco_percentage], 'Reg.':[reg_percentage], 'KNN':[knn_percentage]}
results = pd.DataFrame(results)

plt.figure(figsize=(8,4))
plt.ylim(0,100)
plt.tight_layout()

ax1 = sns.barplot(data=results, palette="Set3")
# ax1.set(xlabel="", ylabel="Percent Success")
ax1.set(xlabel="", ylabel="")

figure = ax1.get_figure()
figure.savefig("cartpole_percent_success.png", bbox_inches='tight')


# In[36]:


np.sum(cost_ratios_coco < 1.01) / len(cost_ratios_coco)
np.sum(cost_ratios_sol < 1.01) / len(cost_ratios_sol)
np.sum(num_solves_coco <= 1) / len(num_solves_coco)
sol_percentage


# In[9]:


results = {}
results['Policy'] = ['CoCo']*len(num_solves_coco) + ['Reg.']*len(num_solves_reg) + ['KNN']*len(num_solves_knn)
results['Solves'] = np.hstack((np.log10(num_solves_coco), np.log10(num_solves_reg), np.log10(num_solves_knn)))
results = pd.DataFrame(results)

plt.figure(figsize=(8,4))
plt.ylim(0,1.05)
plt.tight_layout()

ax1 = sns.boxplot(x=results['Policy'], y=results['Solves'], palette="Set3", flierprops=flierprops)                            
# ax1.set(xlabel="", ylabel="log(QPs Solved)")
ax1.set(xlabel="", ylabel="")

figure = ax1.get_figure()
figure.savefig("cartpole_solved.png", bbox_inches='tight')


# In[10]:


results = {}
results['Policy'] = ['Mosek']*len(total_time_sol) +  ['CoCo']*len(total_time_coco) + ['Reg.']*len(total_time_reg) + ['KNN']*len(total_time_knn)
results['Time'] = np.hstack((np.log10(total_time_sol), np.log10(total_time_coco), np.log10(total_time_reg), np.log10(total_time_knn)))
results = pd.DataFrame(results)

plt.figure(figsize=(8,4))
plt.tight_layout()

ax1 = sns.boxplot(x=results['Policy'], y=results['Time'], palette="Set3", flierprops=flierprops)
ax1.set(xlabel="", ylabel="Time [s]")
ax1.set(xlabel="", ylabel="")

figure = ax1.get_figure()
figure.savefig("cartpole_time.png", bbox_inches='tight')


# In[11]:


results = {}
results['Policy'] = ['Mosek']*len(cost_ratios_sol) + ['CoCo']*len(cost_ratios_coco) + ['Reg.']*len(cost_ratios_reg) + ['KNN']*len(cost_ratios_knn)
results['Costs'] = np.hstack((100*cost_ratios_sol, 100*cost_ratios_coco, 100*cost_ratios_reg, 100*cost_ratios_knn))
results = pd.DataFrame(results)

plt.figure(figsize=(8,4))
plt.ylim(99,1000)
plt.yticks([100, 250, 500, 750, 1000])
plt.tight_layout()

ax1 = sns.boxplot(x=results['Policy'], y=results['Costs'], palette="Set3", flierprops=flierprops)
ax1.set(xlabel="", ylabel="Relative Cost [%]")
ax1.set(xlabel="", ylabel="")

figure = ax1.get_figure()
figure.savefig("cartpole_cost.png", bbox_inches='tight')


# In[ ]:




