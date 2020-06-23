import os
import cvxpy as cp
import pickle
import numpy as np
import sys

sys.path.insert(1, os.environ['MLOPT'])

from utils import sample_points 
from core import Problem

class Manipulation(Problem):
    """Class to setup + solve manipulation problems."""

    def __init__(self, config=None, solver=cp.MOSEK):
        """Constructor for Manipulation class.

        Args:
            config: full path to config file. if None, load default config.
            solver: solver object to be used by cvxpy
        """
        super().__init__()

        ## TODO(pculbertson): allow different sets of params to vary.
        if config is None: #use default config
            relative_path = os.path.dirname(os.path.abspath(__file__))
            config = relative_path + '/config/default.p'

        config_file = open(config,"rb")
        _, prob_params, self.sampled_params = pickle.load(config_file)
        config_file.close()
        self.init_problem(prob_params)

    def init_problem(self,prob_params):
        self.N_v, self.N_h, self.num_grasps, self.F_max, \
            self.h_min, self.h_max, self.h, \
            self.r_min, self.r_max, self.r, \
            self.mu_min, self.mu_max, self.w_std = prob_params

        # sample points on cylinder
        self.Gr, self.p = sample_points(self.N_v, self.N_h, self.h, self.r)

        self.init_bin_problem()
        self.init_mlopt_problem()

    def init_bin_problem(self):
        # Implementation of manipulation system with contacts From "Fast Computation of Optimal Contact Forces, Boyd & Wegbreit (2007)"
        mu = cp.Parameter(1)
        w = cp.Parameter(12)

        self.bin_prob_parameters = {'mu': mu, 'w': w}

        N = self.N_v * self.N_h # total number of points

        # Grasp optimization w.r.t. task specification
        V = np.hstack((np.eye(6), -np.eye(6)))

        # choose points
        G, p = sample_points(self.N_v, self.N_h, self.h, self.r, 0.02)

        # setup problem
        y = cp.Variable(N, boolean=True) # indicator of if point is used
        a = cp.Variable(12) # magnitudes of achievable wrenches in each basis dir.

        # contact forces; one for each point, for each basis direction
        f = [cp.Variable((3,12)) for _ in range(N)]

        self.bin_prob_variables = {'a': a, 'f': f, 'y': y}

        # maximize weighted polyhedron volume
        obj = w.T @ a

        cons = []

        # ray coefficients must be positive
        cons += [a >= 0.]

        for ii in range(12):
          # generate wrench in basis dir.
          cons += [cp.sum([G[jj]@f[jj][:,ii] for jj in range(N)]) == a[ii]*V[:,ii]]

          # friction cone
          for jj in range(N):
            cons += [cp.norm(f[jj][:2,ii]) <= mu*f[jj][2,ii]]

          for jj in range(N):
            # add max normal force
            cons += [f[jj][2,ii] <= self.F_max*y[jj]]

        # limit number of chosen grasps
        cons += [cp.sum(y) <= self.num_grasps]

        self.bin_prob = cp.Problem(cp.Maximize(obj), cons)

    def init_mlopt_problem(self):
        N = self.N_v * self.N_h # total number of points

        mu = cp.Parameter(1)
        w = cp.Parameter(12)
        y = cp.Parameter(N) # indicator of if point is used

        self.mlopt_prob_parameters = {'mu': mu, 'w': w, 'y': y}

        # Grasp optimization w.r.t. task specification
        V = np.hstack((np.eye(6), -np.eye(6)))

        # choose points
        G, p = sample_points(self.N_v, self.N_h, self.h, self.r, 0.02)

        # setup problem
        a = cp.Variable(12) # magnitudes of achievable wrenches in each basis dir.

        # contact forces; one for each point, for each basis direction
        f = [cp.Variable((3,12)) for _ in range(N)]

        # maximize weighted polyhedron volume
        obj = w.T @ a

        cons = []

        # ray coefficients must be positive
        cons += [a >= 0.]

        for ii in range(12):
          # generate wrench in basis dir.
          cons += [cp.sum([G[jj]@f[jj][:,ii] for jj in range(N)]) == a[ii]*V[:,ii]]

          # friction cone
          for jj in range(N):
            cons += [cp.norm(f[jj][:2,ii]) <= mu*f[jj][2,ii]]

          for jj in range(N):
            # add max normal force
            cons += [f[jj][2,ii] <= self.F_max*y[jj]]

        # limit number of chosen grasps
        cons += [cp.sum(y) <= self.num_grasps]

        self.mlopt_prob = cp.Problem(cp.Maximize(obj), cons)

    def solve_micp(self, params, solver=cp.MOSEK):
        """High-level method to solve parameterized MICP.
        
        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            solver: cvxpy Solver object; defaults to Mosek.
        """
        # set cvxpy parameters to their values
        for p in self.sampled_params:
            self.bin_prob_parameters[p].value = params[p]

        ## TODO(pculbertson): allow different sets of params to vary.

        # solve problem with cvxpy
        prob_success, cost, solve_time = False, np.Inf, np.Inf
        a_star, y_star = None, None
        f_star = [None for f_var in self.bin_prob_variables['f']]
        self.bin_prob.solve(solver=solver)

        solve_time = self.bin_prob.solver_stats.solve_time
        if self.bin_prob.status == 'optimal':
            prob_success = True
            cost = self.bin_prob.value
            a_star = self.bin_prob_variables['a'].value
            f_star = [f_var.value for f_var in self.bin_prob_variables['f']]
            y_star = self.bin_prob_variables['y'].value.astype(int)

        return prob_success, cost, solve_time, (a_star, f_star, y_star) 

    def solve_pinned(self, params, strat, solver=cp.MOSEK):
        """High-level method to solve MICP with pinned params & integer values.
        
        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            strat: numpy integer array, corresponding to integer values for the
                desired strategy.
            solver: cvxpy Solver object; defaults to Mosek.
        """
        # set cvxpy params to their values
        for p in self.sampled_params:
            self.mlopt_prob_parameters[p].value = params[p]

        self.mlopt_prob_parameters['y'].value = strat

        ## TODO(pculbertson): allow different sets of params to vary.

        # solve problem with cvxpy
        prob_success, cost, solve_time = False, np.Inf, np.Inf
        self.mlopt_prob.solve(solver=solver)

        solve_time = self.mlopt_prob.solver_stats.solve_time
        if self.mlopt_prob.status == 'optimal':
            prob_success = True
            cost = self.mlopt_prob.value

        # Clear any saved params
        for p in self.sampled_params:
            self.mlopt_prob_parameters[p].value = None
        self.mlopt_prob_parameters['y'].value = None

        return prob_success, cost, solve_time