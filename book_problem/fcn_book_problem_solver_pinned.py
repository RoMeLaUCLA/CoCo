import os, sys
dir_ReDUCE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_ReDUCE + "/utils")
sys.path.append(dir_ReDUCE + "/bookshelf_MIP_solver")

import gurobipy as go
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from get_vertices import get_vertices, plot_rectangle, plot_bilinear
from max_min_trig import find_min_max_cos, find_min_max_sin
from dec2bin import dec2bin
from add_McCormick_envelope_constraint import add_vertex_polytope_constraint_gurobi, limit2vertex, add_bilinear_constraint_gurobi
from add_piecewise_linear_constraint import add_piecewise_linear_constraint
from get_pair_number import get_pair_number
from generate_int_list import generate_int_list
import math, time, runpy, pickle

# Since we assume easy order of items inside the bin, the input items should always be counting from left to
# right 0, 1, 2, 3, 4, 5, etc


def fcn_book_problem_clustered_solver_pinned(shelf_geometry, num_of_item, feature_in, y_guess_all, iter_data, folder_name):

    bin_width = shelf_geometry.shelf_width
    bin_height = shelf_geometry.shelf_height
    bin_left = shelf_geometry.shelf_left
    bin_right = shelf_geometry.shelf_right
    bin_ground = shelf_geometry.shelf_ground
    bin_up = shelf_geometry.shelf_up
    v_bin = shelf_geometry.v_bin

    feature_flatten = feature_in.flatten_to_dictionary()
    item_width_stored = feature_flatten['item_width_stored']
    item_height_stored = feature_flatten['item_height_stored']
    x_stored = feature_flatten['item_center_x_stored']
    y_stored = feature_flatten['item_center_y_stored']
    item_center_stored = np.array([[x_stored[iter_item], y_stored[iter_item]]
                                   for iter_item in range(num_of_item)])
    item_angle_stored = feature_flatten['item_angle_stored']
    item_width_in_hand = feature_flatten['item_width_in_hand']
    item_height_in_hand = feature_flatten['item_height_in_hand']

    init_globals = {'num_of_item': num_of_item, 'bin_width': bin_width, 'bin_height': bin_height, 'item_width_stored': item_width_stored}
    ret_dict = runpy.run_module('setup_variable_range', init_globals=init_globals)

    dim_2D = ret_dict['dim_2D']
    num_of_vertices = ret_dict['num_of_vertices']
    num_of_pairs = ret_dict['num_of_pairs']
    list_pairs = ret_dict['list_pairs']
    num_of_states_stored = ret_dict['num_of_states_stored']
    num_of_states_in_hand = ret_dict['num_of_states_in_hand']
    num_of_int_R = ret_dict['num_of_int_R']

    # Rotation matrix
    item_angle_ranges = ret_dict['item_ranges']
    R_knots_0000 = ret_dict['R_knots_0000']
    k_R_sq_0000 = ret_dict['k_R_sq_0000']
    b_R_sq_0000 = ret_dict['b_R_sq_0000']
    R_knots_1010 = ret_dict['R_knots_1010']
    k_R_sq_1010 = ret_dict['k_R_sq_1010']
    b_R_sq_1010 = ret_dict['b_R_sq_1010']
    v_all_R_0001 = ret_dict['v_all_R_0001']
    num_of_vertices_R_0001 = ret_dict['num_of_vertices_R_0001']
    num_of_polygons_R_0001 = ret_dict['num_of_polygons_R_0001']
    len_sections_R_0001 = ret_dict['len_sections_R_0001']

    a_knots_00 = ret_dict['a_knots_00']
    k_a_sq_00 = ret_dict['k_a_sq_00']
    b_a_sq_00 = ret_dict['b_a_sq_00']
    a_knots_11 = ret_dict['a_knots_11']
    k_a_sq_11 = ret_dict['k_a_sq_11']
    b_a_sq_11 = ret_dict['b_a_sq_11']
    num_of_int_a_00 = ret_dict['num_of_int_a_00']
    num_of_int_a_11 = ret_dict['num_of_int_a_11']

    # Vertices
    v_knots_common_x = ret_dict['v_knots_common_x']
    v_knots_common_y = ret_dict['v_knots_common_y']
    num_of_integer_v = ret_dict['num_of_integer_v']
    num_of_vertices_av = ret_dict['num_of_vertices_av']
    num_of_polygons_av = ret_dict['num_of_polygons_av']
    v_all_av = ret_dict['v_all_av']

    bigM = 10000
    INF = go.GRB.INFINITY
    ct_item_in_hand = num_of_item

    # fig = plt.figure(1)
    # ax = plt.axes(projection="3d")
    # ax.scatter3D(v_all_av[0, :], v_all_av[1, :], v_all_av[2, :], marker='.')
    # plt.show()

    begin_time = time.time()

    m = go.Model("Bin_organization")
    m.setParam('MIPGap', 1e-2)

    x_item = m.addVars(num_of_item+1, dim_2D, lb=-bigM, ub=bigM)  # Positions for stored items
    R_wb = m.addVars(num_of_item+1, dim_2D, dim_2D, lb=-1.0, ub=1.0)  # Rotation matrices for stored items
    v_item = m.addVars(num_of_item+1, num_of_vertices, dim_2D, lb=-bigM, ub=bigM)  # Positions for vertices of stored items

    R_wb_0000 = m.addVars(num_of_item+1, lb=-1.0, ub=1.0)
    R_wb_1010 = m.addVars(num_of_item+1, lb=-1.0, ub=1.0)
    R_wb_0001 = m.addVars(num_of_item+1, lb=-1.0, ub=1.0)

    a_sep = m.addVars(num_of_pairs, dim_2D, lb=-1.0, ub=1.0)  # Variables to formulate a^{T}x<=b
    b_sep = m.addVars(num_of_pairs, lb=-bigM, ub=bigM)
    a_sep_sq = m.addVars(num_of_pairs, dim_2D, lb=-1.0, ub=1.0)

    # The meaning dimension for a_times_v: #pair, which item in pair, 2D plane dimension, #vertex
    a_times_v = m.addVars(num_of_pairs, dim_2D, dim_2D, num_of_vertices, lb=-bigM, ub=bigM)

    # Lambda variables =================================================================================================
    lam_0001 = []
    for iter_item in range(num_of_item+1):
        lam_0001.append(m.addVars(num_of_vertices_R_0001[iter_item], lb=0.0, ub=1.0))  # For some reason elements
        # within numpy arrays cannot be used for indexing addVars

    lam_av = []
    for iter_pair in range(num_of_pairs):
        lam_av.append([])
        for iter_paired in range(2):
            lam_av[iter_pair].append([])
            for iter_dim in range(dim_2D):
                lam_av[iter_pair][iter_paired].append([])
                for iter_vertex in range(num_of_vertices):
                    lam_av[iter_pair][iter_paired][iter_dim].append(m.addVars(
                        num_of_vertices_av[iter_pair][iter_paired][iter_dim][iter_vertex], lb=0.0, ub=1.0))

    m.update()

    # Constraint: Vertices is related to center position and orientation ===============================================
    for iter_item in range(num_of_item + 1):

        if iter_item == num_of_item:
            W = item_width_in_hand / 2.0
            H = item_height_in_hand / 2.0
        else:
            W = item_width_stored[iter_item] / 2.0
            H = item_height_stored[iter_item] / 2.0

        for iter_dim in range(dim_2D):
            m.addConstr(v_item[iter_item, 0, iter_dim] == (x_item[iter_item, iter_dim]
                                                           + R_wb[iter_item, iter_dim, 0] * W
                                                           + R_wb[iter_item, iter_dim, 1] * H))

            m.addConstr(v_item[iter_item, 1, iter_dim] == (x_item[iter_item, iter_dim]
                                                           + R_wb[iter_item, iter_dim, 0] * W
                                                           - R_wb[iter_item, iter_dim, 1] * H))

            m.addConstr(v_item[iter_item, 2, iter_dim] == (x_item[iter_item, iter_dim]
                                                           - R_wb[iter_item, iter_dim, 0] * W
                                                           - R_wb[iter_item, iter_dim, 1] * H))

            m.addConstr(v_item[iter_item, 3, iter_dim] == (x_item[iter_item, iter_dim]
                                                           - R_wb[iter_item, iter_dim, 0] * W
                                                           + R_wb[iter_item, iter_dim, 1] * H))

    # Constraint: all objects within bin ===============================================================================
    for iter_item in range(num_of_item + 1):
        for iter_vertex in range(num_of_vertices):
            m.addConstr(v_item[iter_item, iter_vertex, 0] <= bin_right)
            m.addConstr(v_item[iter_item, iter_vertex, 0] >= bin_left)
            m.addConstr(v_item[iter_item, iter_vertex, 1] <= bin_up)
            m.addConstr(v_item[iter_item, iter_vertex, 1] >= bin_ground)

    # Rotation angles within -90 deg to 90 deg =========================================================================
    for iter_item in range(num_of_item + 1):
        m.addConstr(R_wb[iter_item, 0, 0] >= 0.0)

    # Symmetric rotation matrix ========================================================================================
    for iter_item in range(num_of_item + 1):
        # This implies orthogonality
        m.addConstr(R_wb[iter_item, 0, 0] == R_wb[iter_item, 1, 1])
        m.addConstr(R_wb[iter_item, 1, 0] + R_wb[iter_item, 0, 1] == 0)

    # Bilinear rotation matrix constraint ==============================================================================
    # First, bilinear variables become linear
    for iter_item in range(num_of_item + 1):
        # Orthogonality is automatically satisfied
        # Determinant is 1
        m.addConstr((R_wb_0000[iter_item] + R_wb_1010[iter_item]) == 1)  # 0011 changed to 0000, 1001 changed to -1010

    num_of_candidates = len(y_guess_all)

    all_constr_to_remove = []

    for iter_candidate in range(num_of_candidates):

        # Remove constraints if the list is not empty
        for iter_remove_constr in range(len(all_constr_to_remove)):
            m.remove(all_constr_to_remove[iter_remove_constr])
        all_constr_to_remove = []

        y_guess = y_guess_all[iter_candidate]

        # Transform integer variables into convex regions ==================================================================
        int_item_state = np.zeros([num_of_item+1, num_of_states_stored])
        int_item_state_in_hand = np.zeros(num_of_states_in_hand)
        int_R = np.zeros([num_of_item+1, num_of_int_R[0]])  # All items should have the same amount of int var
        int_a_00 = np.zeros([num_of_pairs, num_of_int_a_00[0]])  # All items should have the same amount of int var
        int_a_11 = np.zeros([num_of_pairs, num_of_int_a_11[0]])  # All items should have the same amount of int var
        int_v = np.zeros([num_of_item+1, num_of_vertices, dim_2D, num_of_integer_v])  # All items should have the same amount of int var

        iter_Y = 0
        # Int state integers
        for iter_item in range(num_of_item):
            for iter_int in range(num_of_states_stored):
                int_item_state[iter_item, iter_int] = y_guess[iter_Y]
                iter_Y += 1

        for iter_int in range(num_of_states_in_hand):
            int_item_state_in_hand[iter_int] = y_guess[iter_Y]
            iter_Y += 1

        # Int R integers
        for iter_item in range(num_of_item+1):
            for iter_int in range(num_of_int_R[0]):  # Small iter_int means smaller digit
                int_R[iter_item, iter_int] = y_guess[iter_Y]
                iter_Y += 1

        # Int a integers. Int_a_00 first
        for iter_pair in range(num_of_pairs):
            for iter_int in range(num_of_int_a_00[0]):
                int_a_00[iter_pair, iter_int] = y_guess[iter_Y]
                iter_Y += 1

        # Int a integers. Int_a_11 second
        for iter_pair in range(num_of_pairs):
            for iter_int in range(num_of_int_a_11[0]):
                int_a_11[iter_pair, iter_int] = y_guess[iter_Y]
                iter_Y += 1

        int_a = [int_a_00, int_a_11]

        # Int v integers.
        for iter_item in range(num_of_item + 1):
            for iter_vertex in range(num_of_vertices):
                for iter_dim in range(dim_2D):
                    for iter_int in range(num_of_integer_v):
                        int_v[iter_item, iter_vertex, iter_dim, iter_int] = y_guess[iter_Y]
                        iter_Y += 1

        assert iter_Y == len(y_guess), "Error in iter_Y !!"

        # Squared terms
        # Stored items
        for iter_item in range(num_of_item+1):

            int_list_0000 = generate_int_list(int_R[iter_item])
            int_list_1010 = generate_int_list(int_R[iter_item])

            # print("====================================================================================")
            # print("Item {}".format(iter_item))
            # print("R_0000")
            # print([R_knots_0000[iter_item][elem] for elem in filter_active_0000])
            # print([k_R_sq_0000[iter_item][elem] for elem in filter_active_0000])
            # print([b_R_sq_0000[iter_item][elem] for elem in filter_active_0000])
            # print("------------------------------------------------------------------------------------")
            # print("R_1010")
            # print([R_knots_1010[iter_item][elem] for elem in filter_active_1010])
            # print([k_R_sq_1010[iter_item][elem] for elem in filter_active_1010])
            # print([b_R_sq_1010[iter_item][elem] for elem in filter_active_1010])
            # print("------------------------------------------------------------------------------------")

            # R_wb_stored_0000 = R_wb_stored[0, 0]*R_wb_stored[0, 0]
            ret = add_piecewise_linear_constraint(m, R_wb[iter_item, 0, 0], R_wb_0000[iter_item],
                                            R_knots_0000[iter_item],
                                            k_R_sq_0000[iter_item],
                                            b_R_sq_0000[iter_item],
                                            int_list_0000, bigM, pinned=True)

            all_constr_to_remove.extend(ret)

            # R_wb_stored_1010 = R_wb_stored[1, 0]*R_wb_stored[1, 0]
            ret = add_piecewise_linear_constraint(m, R_wb[iter_item, 1, 0], R_wb_1010[iter_item],
                                            R_knots_1010[iter_item],
                                            k_R_sq_1010[iter_item],
                                            b_R_sq_1010[iter_item],
                                            int_list_1010, bigM, pinned=True)

            all_constr_to_remove.extend(ret)

        # Cross terms
        for iter_item in range(num_of_item+1):
            # R_wb_stored_0001 = R_wb_stored[0, 0]*R_wb_stored[0, 1]
            # Filter v_all_R_0001.
            # No need to filter lam, num_of_polygons_R_0001, int_list - they are already based on the filtered vertices.
            x_0001 = [R_wb[iter_item, 0, 0], R_wb[iter_item, 0, 1], R_wb_0001[iter_item]]

            list_lam_0001 = [lam_0001[iter_item][iter_lam] for iter_lam in range(num_of_vertices_R_0001[iter_item])]
            int_list_0001 = [int_R[iter_item][iter_zz] for iter_zz in range(len(int_R[iter_item]))]

            # print("------------------------------------------------------------------------------------------")
            # print("Item {}".format(iter_item))
            # print(int_list_0001)
            # print(v_all_R_0001[iter_item])

            ret = add_bilinear_constraint_gurobi(m, x_0001, list_lam_0001, int_list_0001, num_of_polygons_R_0001[iter_item],
                                           v_all_R_0001[iter_item], pinned=True)

            all_constr_to_remove.extend(ret)

        # Item state constraint ============================================================================================
        bin_offset = 1.5  # To decrease numerical issue
        angle_offset = 0.05

        # m.addConstr(go.quicksum(int_item_state_in_hand[iter_in_hand_state]
        #                         for iter_in_hand_state in range(num_of_states_in_hand)) == 1.0)

        # State z0 - left fall, sin(theta) = 1
        all_constr_to_remove.append(m.addConstr(R_wb[ct_item_in_hand, 1, 0] >= 1.0 - angle_offset - bigM * (1 - int_item_state_in_hand[0])))
        all_constr_to_remove.append(m.addConstr(x_item[ct_item_in_hand, 1] >= item_width_in_hand / 2.0 - bin_offset - bigM * (1 - int_item_state_in_hand[0])))
        all_constr_to_remove.append(m.addConstr(x_item[ct_item_in_hand, 1] <= item_width_in_hand / 2.0 + bin_offset + bigM * (1 - int_item_state_in_hand[0])))

        # State z1 - upright, sin(theta) = 0
        all_constr_to_remove.append(m.addConstr(R_wb[ct_item_in_hand, 1, 0] >= 0.0 - angle_offset - bigM * (1 - int_item_state_in_hand[1])))
        all_constr_to_remove.append(m.addConstr(R_wb[ct_item_in_hand, 1, 0] <= 0.0 + angle_offset + bigM * (1 - int_item_state_in_hand[1])))
        all_constr_to_remove.append(m.addConstr(x_item[ct_item_in_hand, 1] >= item_height_in_hand / 2.0 - bin_offset - bigM * (1 - int_item_state_in_hand[1])))
        all_constr_to_remove.append(m.addConstr(x_item[ct_item_in_hand, 1] <= item_height_in_hand / 2.0 + bin_offset + bigM * (1 - int_item_state_in_hand[1])))

        # State z2 - right fall, sin(theta) = -1
        all_constr_to_remove.append(m.addConstr(R_wb[ct_item_in_hand, 1, 0] <= -1.0 + angle_offset + bigM*(1 - int_item_state_in_hand[2])))
        all_constr_to_remove.append(m.addConstr(x_item[ct_item_in_hand, 1] >= item_width_in_hand / 2.0 - bin_offset - bigM * (1 - int_item_state_in_hand[2])))
        all_constr_to_remove.append(m.addConstr(x_item[ct_item_in_hand, 1] <= item_width_in_hand / 2.0 + bin_offset + bigM * (1 - int_item_state_in_hand[2])))

        # States of stored items
        for iter_item in range(num_of_item):
            # TODO: We have one issue here: stored object cannot lean onto the object in hand.
            #  We can fix that but maybe this is a reasonable assumption.
            #  In fact, logically any item (including in hand), if it is left tilting, it needs to lean on some other object
            #  and we can just set the separating plane between this item and the other item to cross those 2 vertices.
            #  Same thing for right tilting.

            # m.addConstr(go.quicksum(int_item_state[iter_item, iter_state] for iter_state in range(num_of_states_stored)) == 1.0)

            # State 0: Left fall: the item is 90 degrees on the ground =====================================================
            all_constr_to_remove.append(m.addConstr(R_wb[iter_item, 1, 0] >= 1.0 - angle_offset - bigM * (1-int_item_state[iter_item, 0])))
            all_constr_to_remove.append(m.addConstr(x_item[iter_item, 1] >= bin_ground - bin_offset + item_width_stored[iter_item] / 2.0 - bigM * (1-int_item_state[iter_item, 0])))
            all_constr_to_remove.append(m.addConstr(x_item[iter_item, 1] <= bin_ground + bin_offset + item_width_stored[iter_item] / 2.0 + bigM * (1-int_item_state[iter_item, 0])))

            # State 1: the item is 0~90 degrees leaning towards left =======================================================
            if iter_item == 0:
                # TODO: Another way to do this is to make the left/right wall also items in the bin
                all_constr_to_remove.append(m.addConstr(v_item[iter_item, 3, 0] >= bin_left - bin_offset - bigM*(1-int_item_state[iter_item, 1])))
                all_constr_to_remove.append(m.addConstr(v_item[iter_item, 3, 0] <= bin_left + 7 + bin_offset + bigM*(1-int_item_state[iter_item, 1])))  # The thickness of wall is 7

            # Otherwise, enforce leaning left constraint
            else:
                # iter_pair == 0 means the item on the left
                item_left = iter_item - 1
                this_item = iter_item

                ct_pair = get_pair_number(list_pairs, item_left, this_item)

                # The meaning dimension for a_times_v: #pair, which item in pair, 2D plane dimension, #vertex
                # the plane goes across the vertex 3 on the right object, and vertex 0 on the left object
                all_constr_to_remove.append(m.addConstr((a_times_v[ct_pair, 1, 0, 3] + a_times_v[ct_pair, 1, 1, 3]) <= b_sep[ct_pair] + bigM*(1-int_item_state[iter_item, 1])))
                all_constr_to_remove.append(m.addConstr((a_times_v[ct_pair, 1, 0, 3] + a_times_v[ct_pair, 1, 1, 3]) >= b_sep[ct_pair] - bigM*(1-int_item_state[iter_item, 1])))
                all_constr_to_remove.append(m.addConstr((a_times_v[ct_pair, 0, 0, 0] + a_times_v[ct_pair, 0, 1, 0]) <= b_sep[ct_pair] + bigM*(1-int_item_state[iter_item, 1])))
                all_constr_to_remove.append(m.addConstr((a_times_v[ct_pair, 0, 0, 0] + a_times_v[ct_pair, 0, 1, 0]) >= b_sep[ct_pair] - bigM*(1-int_item_state[iter_item, 1])))

                # Stability x[left_item] <= x[this_item] <= v2_x[this_item]
                all_constr_to_remove.append(m.addConstr(x_item[this_item, 0] >= x_item[item_left, 0] - bigM*(1-int_item_state[iter_item, 1])))
                all_constr_to_remove.append(m.addConstr(x_item[this_item, 0] <= v_item[this_item, 2, 0] + bigM*(1-int_item_state[iter_item, 1])))

            # v2_y[this_item] touches the ground
            all_constr_to_remove.append(m.addConstr(v_item[iter_item, 2, 1] <= bin_ground + bin_offset + bigM*(1-int_item_state[iter_item, 1])))
            all_constr_to_remove.append(m.addConstr(v_item[iter_item, 2, 1] >= bin_ground - bin_offset - bigM*(1-int_item_state[iter_item, 1])))

            # Angle > 0
            all_constr_to_remove.append(m.addConstr(R_wb[iter_item, 1, 0] >= 0.0 - bigM*(1-int_item_state[iter_item, 1])))

            # State 2: the item is upright =================================================================================
            all_constr_to_remove.append(m.addConstr(R_wb[iter_item, 1, 0] >= 0.0 - angle_offset - bigM * (1-int_item_state[iter_item, 2])))
            all_constr_to_remove.append(m.addConstr(R_wb[iter_item, 1, 0] <= 0.0 + angle_offset + bigM * (1-int_item_state[iter_item, 2])))
            all_constr_to_remove.append(m.addConstr(x_item[iter_item, 1] >= bin_ground - bin_offset + item_height_stored[iter_item] / 2.0 - bigM * (1-int_item_state[iter_item, 2])))
            all_constr_to_remove.append(m.addConstr(x_item[iter_item, 1] <= bin_ground + bin_offset + item_height_stored[iter_item] / 2.0 + bigM * (1-int_item_state[iter_item, 2])))

            # State 3: the item is -90~0 degrees leaning towards right =====================================================
            # If it is the last item, special treatment
            if iter_item == num_of_item - 1:
                # TODO: Another way to do this is to make the left/right wall also items in the bin
                all_constr_to_remove.append(m.addConstr(v_item[iter_item, 0, 0] >= bin_right - bin_offset - bigM*(1-int_item_state[iter_item, 3])))
                all_constr_to_remove.append(m.addConstr(v_item[iter_item, 0, 0] <= bin_right + bin_offset + bigM*(1-int_item_state[iter_item, 3])))

            # Otherwise, enforce leaning left constraint
            else:
                # iter_pair == 0 means the item on the left
                this_item = iter_item
                item_right = iter_item + 1

                ct_pair = get_pair_number(list_pairs, this_item, item_right)

                # The meaning dimension for a_times_v: #pair, which item in pair, 2D plane dimension, #vertex
                # the plane goes across the vertex 3 on the right object, and vertex 0 on the left object
                all_constr_to_remove.append(m.addConstr((a_times_v[ct_pair, 0, 0, 0] + a_times_v[ct_pair, 0, 1, 0]) <= b_sep[ct_pair] + bigM*(1-int_item_state[iter_item, 3])))
                all_constr_to_remove.append(m.addConstr((a_times_v[ct_pair, 0, 0, 0] + a_times_v[ct_pair, 0, 1, 0]) >= b_sep[ct_pair] - bigM*(1-int_item_state[iter_item, 3])))
                all_constr_to_remove.append(m.addConstr((a_times_v[ct_pair, 1, 0, 3] + a_times_v[ct_pair, 1, 1, 3]) <= b_sep[ct_pair] + bigM*(1-int_item_state[iter_item, 3])))
                all_constr_to_remove.append(m.addConstr((a_times_v[ct_pair, 1, 0, 3] + a_times_v[ct_pair, 1, 1, 3]) >= b_sep[ct_pair] - bigM*(1-int_item_state[iter_item, 3])))

                # Stability v1_x[this_item] <= x[this_item] <= x[right_item]
                all_constr_to_remove.append(m.addConstr(x_item[this_item, 0] >= v_item[this_item, 1, 0] - bigM*(1-int_item_state[iter_item, 3])))
                all_constr_to_remove.append(m.addConstr(x_item[this_item, 0] <= x_item[item_right, 0] + bigM*(1-int_item_state[iter_item, 3])))

            # v1_y[this_item] touches the ground
            all_constr_to_remove.append(m.addConstr(v_item[iter_item, 1, 1] <= bin_ground + bin_offset + bigM*(1-int_item_state[iter_item, 3])))
            all_constr_to_remove.append(m.addConstr(v_item[iter_item, 1, 1] >= bin_ground - bin_offset - bigM*(1-int_item_state[iter_item, 3])))

            # Angle < 0
            all_constr_to_remove.append(m.addConstr(R_wb[iter_item, 1, 0] <= 0 + bigM*(1-int_item_state[iter_item, 3])))

            # State 4: Right fall: the item is -90 degrees on the ground ===================================================
            all_constr_to_remove.append(m.addConstr(R_wb[iter_item, 1, 0] <= -1.0 + angle_offset + bigM*(1-int_item_state[iter_item, 4])))
            all_constr_to_remove.append(m.addConstr(x_item[iter_item, 1] >= bin_ground - bin_offset + item_width_stored[iter_item] / 2.0 - bigM*(1-int_item_state[iter_item, 4])))
            all_constr_to_remove.append(m.addConstr(x_item[iter_item, 1] <= bin_ground + bin_offset + item_width_stored[iter_item] / 2.0 + bigM*(1-int_item_state[iter_item, 4])))

        # Collision free constraints =======================================================================================
        for iter_pair in range(num_of_pairs):
            # a is a unit vector
            all_constr_to_remove.append(m.addConstr(1.0 == (a_sep_sq[iter_pair, 0] + a_sep_sq[iter_pair, 1])))

            item_l = int(list_pairs[iter_pair, 0])
            item_r = int(list_pairs[iter_pair, 1])
            assert item_l != num_of_item, "Error: the first item cannot be the item in hand !"

            # a_sq terms are squared of a components
            # a_sep_sq[iter_pair, 0] = a_sep[iter_pair, 0]*a_sep[iter_pair, 0] ---------------------------------------------

            int_aaa00 = generate_int_list(int_a_00[iter_pair])
            # To reassign integer variables, filter R_knots, k, b. No need to filter int_list.

            ret = add_piecewise_linear_constraint(m, a_sep[iter_pair, 0], a_sep_sq[iter_pair, 0],
                                            a_knots_00[iter_pair],
                                            k_a_sq_00[iter_pair],
                                            b_a_sq_00[iter_pair], int_aaa00, bigM, pinned=True)

            all_constr_to_remove.append(ret)

            # a_sep_sq[iter_pair, 1] = a_sep[iter_pair, 1]*a_sep[iter_pair, 1] ---------------------------------------------

            int_aaa11 = generate_int_list(int_a_11[iter_pair])

            # To reassign integer variables, filter R_knots, k, b. No need to filter int_list.
            ret = add_piecewise_linear_constraint(m, a_sep[iter_pair, 1], a_sep_sq[iter_pair, 1],
                                            a_knots_11[iter_pair],
                                            k_a_sq_11[iter_pair],
                                            b_a_sq_11[iter_pair], int_aaa11, bigM, pinned=True)

            all_constr_to_remove.append(ret)

            # print("=======================================================================================================")
            # print("Pair {}".format(iter_pair))
            # print(a_knots_00_filtered[iter_pair])
            # print(a_knots_11_filtered[iter_pair])

            # --------------------------------------------------------------------------------------------------------------
            for iter_vertex in range(num_of_vertices):
                # a*v on one side of the plane
                # The meaning dimension for a_times_v: #pair, which item in pair, 2D plane dimension, #vertex
                # TODO: Note! The order (<= or >=) doesn't matter. If the objects flip sides, the optimizer will get (-a, -b)
                #  instead of (a, b). Since we are getting global optimal, the sides won't be flipped.
                all_constr_to_remove.append(m.addConstr(a_times_v[iter_pair, 0, 0, iter_vertex] + a_times_v[iter_pair, 0, 1, iter_vertex] <= b_sep[iter_pair]))
                all_constr_to_remove.append(m.addConstr(a_times_v[iter_pair, 1, 0, iter_vertex] + a_times_v[iter_pair, 1, 1, iter_vertex] >= b_sep[iter_pair]))

                # Cross terms
                # Use filtered v_all_av
                for iter_dim in range(dim_2D):
                    # Pair left item ---------------------------------------------------------------------------------------
                    x = [a_sep[iter_pair, iter_dim], v_item[item_l, iter_vertex, iter_dim], a_times_v[iter_pair, 0, iter_dim, iter_vertex]]
                    lam = [lam_av[iter_pair][0][iter_dim][iter_vertex][iter_lam] for iter_lam in
                           range(num_of_vertices_av[iter_pair][0][iter_dim][iter_vertex])]

                    int_var = [int_v[item_l][iter_vertex][iter_dim][iter_zz] for iter_zz in range(num_of_integer_v)] + \
                              [int_a[iter_dim][iter_pair][iter_zz] for iter_zz in range(num_of_int_a_00[0])]

                    ret = add_bilinear_constraint_gurobi(m, x, lam, int_var,
                                                   num_of_polygons_av[iter_pair][0][iter_dim][iter_vertex],
                                                   v_all_av[iter_pair][0][iter_dim][iter_vertex], pinned=True)

                    all_constr_to_remove.append(ret)

                    # print("-----------------------------------------------------------------------------------------------")
                    # print("Pair {}, Paired {}, Dim {}, Vertex {}".format(iter_pair, 0, iter_dim, iter_vertex))
                    # print(int_var)
                    # print(num_of_polygons_av_filtered[iter_pair][0][iter_dim][iter_vertex])
                    # print(v_all_av_filtered[iter_pair][0][iter_dim][iter_vertex])

                    # Pair right item --------------------------------------------------------------------------------------
                    x = [a_sep[iter_pair, iter_dim], v_item[item_r, iter_vertex, iter_dim], a_times_v[iter_pair, 1, iter_dim, iter_vertex]]
                    lam = [lam_av[iter_pair][1][iter_dim][iter_vertex][iter_lam] for iter_lam in
                           range(num_of_vertices_av[iter_pair][1][iter_dim][iter_vertex])]

                    int_var = [int_v[item_r][iter_vertex][iter_dim][iter_zz] for iter_zz in range(num_of_integer_v)] + \
                              [int_a[iter_dim][iter_pair][iter_zz] for iter_zz in range(num_of_int_a_00[0])]

                    ret = add_bilinear_constraint_gurobi(m, x, lam, int_var,
                                                   num_of_polygons_av[iter_pair][1][iter_dim][iter_vertex],
                                                   v_all_av[iter_pair][1][iter_dim][iter_vertex], pinned=True)

                    all_constr_to_remove.append(ret)

                    # print("-----------------------------------------------------------------------------------------------")
                    # print("Pair {}, Paired {}, Dim {}, Vertex {}".format(iter_pair, 1, iter_dim, iter_vertex))
                    # print(int_var)
                    # print(num_of_polygons_av_filtered[iter_pair][1][iter_dim][iter_vertex])
                    # print(v_all_av_filtered[iter_pair][1][iter_dim][iter_vertex])

        obj1 = go.quicksum(go.quicksum((x_item[iter_item, iter_dim] * x_item[iter_item, iter_dim] -
                        2 * item_center_stored[iter_item, iter_dim] * x_item[iter_item, iter_dim]) for iter_item in
                       range(num_of_item)) for iter_dim in range(dim_2D))

        obj2 = go.quicksum((R_wb[iter_item, 0, 0] * R_wb[iter_item, 0, 0] - 2 * np.cos(item_angle_stored[iter_item]) *
                    R_wb[iter_item, 0, 0]) for iter_item in range(num_of_item))

        obj = obj1 + obj2

        m.setObjective(obj, go.GRB.MINIMIZE)

        m.optimize()
        end_time = time.time()

        time_ret = end_time - begin_time
        print("Solving time for single problem is {} ms".format(1000 * time_ret))

        if m.SolCount > 0:
            prob_success = True
        else:
            prob_success = False

        cost_ret = 0
        X_ret = []

        plot = False

        if plot:
            # Plot original bin
            fig, (ax1, ax2) = plt.subplots(2, 1)

            plot_rectangle(ax1, v_bin, color='black', show=False)

            for iter_item in range(num_of_item):
                theta = item_angle_stored[iter_item]
                R_wb_original = np.array([[np.cos(theta), -np.sin(theta)],
                                          [np.sin(theta), np.cos(theta)]])

                v_item_original = get_vertices(item_center_stored[iter_item, :],
                                               R_wb_original,
                                               np.array([item_height_stored[iter_item], item_width_stored[iter_item]]))

                plot_rectangle(ax1, v_item_original, color='red', show=False)

            ax1.set_xlim([bin_left - 10, bin_right + 10])
            ax1.set_ylim([bin_ground - 10, bin_up + 10])
            ax1.set_aspect('equal', adjustable='box')
            ax1.grid()
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')

        if prob_success:
            # print("========================= Begin: item angles ===================================")
            # for iter_item in range(num_of_item+1):
            #     print(math.atan2(R_wb[iter_item, 1, 0].X, R_wb[iter_item, 0, 0].X))
            #
            # print("========================= Begin: check orthogonality for rotation matrices ============================")
            # for iter_item in range(num_of_item+1):
            #     R_stored = np.array([[R_wb[iter_item, 0, 0].X, R_wb[iter_item, 0, 1].X],
            #                          [R_wb[iter_item, 1, 0].X, R_wb[iter_item, 1, 1].X]])
            #     print(R_stored)
            #     print(R_stored.dot(R_stored.transpose()))
            #     print("---------------------------------------------------------------------------------------------------")
            # print("========================= End: check orthogonality for rotation matrices ==============================")
            #
            # print("======================== Item states ================================")
            # for iter_item in range(num_of_item):
            #     print([int_item_state[iter_item, iter_int] for iter_int in range(num_of_states_stored)])
            #
            # print([int_item_state_in_hand[iter_int] for iter_int in range(num_of_states_in_hand)])
            # print("======================== End: Item states ================================")
            #
            # print("=============================== Begin: Check Separating planes ========================================")
            # for iter_pair in range(num_of_pairs):
            #     print("---------------------------------------------------------------------------------------------------")
            #     print([a_sep[iter_pair, 0].X, a_sep[iter_pair, 1].X])
            #     print([a_sep[iter_pair, 0].X ** 2, a_sep[iter_pair, 1].X ** 2])
            #     print([a_sep_sq[iter_pair, 0].X, a_sep_sq[iter_pair, 1].X])
            #     print(np.sqrt(a_sep[iter_pair, 0].X ** 2 + a_sep[iter_pair, 1].X ** 2))
            # print("=============================== End: Check Separating planes ==========================================")

            if plot:
                # Plot solved bin
                plot_rectangle(ax2, v_bin, color='black', show=False)

                v_item_stored_sol = []
                for iter_item in range(num_of_item):
                    v_item_stored_sol.append(np.array([[v_item[iter_item, 0, 0].X, v_item[iter_item, 0, 1].X],
                                                       [v_item[iter_item, 1, 0].X, v_item[iter_item, 1, 1].X],
                                                       [v_item[iter_item, 2, 0].X, v_item[iter_item, 2, 1].X],
                                                       [v_item[iter_item, 3, 0].X, v_item[iter_item, 3, 1].X]]))

                    plot_rectangle(ax2, v_item_stored_sol[iter_item], color='red', show=False)

                v_item_in_hand_sol = np.array([[v_item[ct_item_in_hand, 0, 0].X, v_item[ct_item_in_hand, 0, 1].X],
                                               [v_item[ct_item_in_hand, 1, 0].X, v_item[ct_item_in_hand, 1, 1].X],
                                               [v_item[ct_item_in_hand, 2, 0].X, v_item[ct_item_in_hand, 2, 1].X],
                                               [v_item[ct_item_in_hand, 3, 0].X, v_item[ct_item_in_hand, 3, 1].X]])

                plot_rectangle(ax2, v_item_in_hand_sol, color='blue', show=False)

                # # Plot separating planes
                # for iter_pair in range(num_of_pairs):
                #     yy = np.linspace(bin_ground, bin_up, 100)
                #     xx = (b_sep[iter_pair].X - a_sep[iter_pair, 1].X * yy) / a_sep[iter_pair, 0].X
                #     plt.plot(xx, yy, 'green')

                ax2.set_xlim([bin_left - 10, bin_right + 10])
                ax2.set_ylim([bin_ground - 10, bin_up + 10])
                ax2.set_aspect('equal', adjustable='box')
                ax2.grid()
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')

                plt.savefig('saved_figures/' + str(folder_name) + '/Test_{}_feasible.png'.format(iter_data), dpi=300, bbox_inches='tight')
                plt.show(block=False)

            # states are: item_center_stored, item_angle_stored, item_center_in_hand, item_angle_in_hand, a_sep, b_sep
            # Centers for stored items
            for iter_item in range(num_of_item):
                for iter_dim in range(dim_2D):
                    X_ret.append(x_item[iter_item, iter_dim].X)

            # Angles for stored items
            for iter_item in range(num_of_item):
                X_ret.append(math.atan2(R_wb[iter_item, 1, 0].X, R_wb[iter_item, 0, 0].X))

            # Center for item-in-hand
            for iter_dim in range(dim_2D):
                X_ret.append(x_item[ct_item_in_hand, iter_dim].X)

            # Angle for item-in-hand
            X_ret.append(math.atan2(R_wb[ct_item_in_hand, 1, 0].X, R_wb[ct_item_in_hand, 0, 0].X))

            # Separating planes a
            for iter_pair in range(num_of_pairs):
                for iter_dim in range(dim_2D):
                    X_ret.append(a_sep[iter_pair, iter_dim].X)

            # Separating planes b
            for iter_pair in range(num_of_pairs):
                X_ret.append(b_sep[iter_pair].X)

            cost_ret = obj.getValue()

        elif not prob_success:
            if plot:
                plt.savefig('saved_figures/' + str(folder_name) + '/Test_{}_infeasible.png'.format(iter_data), dpi=300, bbox_inches='tight')
                plt.show(block=False)

        else:
            assert False, "What is going on ??"

        if prob_success:
            break

    return prob_success, cost_ret, time_ret, X_ret


def main():
    bin_width = 176
    bin_height = 110

    bin_left = -88.0
    bin_right = 88.0
    bin_ground = 0.0
    bin_up = 110

    v_bin = np.array([[88., 110.],
                      [88., 0.],
                      [-88., 0.],
                      [-88., 110.]])

    num_of_item = 3

    with open('dataset_with_y_guess_for_debugging.pkl', 'rb') as f:
        data_with_y_guess_debug = pickle.load(f)


    all_features = data_with_y_guess_debug[0]
    all_y_guess = data_with_y_guess_debug[1]
    all_successful = []

    for iter_data in range(len(all_features)):
        this_feature = all_features[iter_data]
        y_guess = all_y_guess[iter_data]

        assert len(this_feature) == 17, "Inconsistent feature length !!"
        assert len(y_guess) == 130, "Inconsistent integer variable length !!"

        item_width_stored = [this_feature[5 * 0 + 4], this_feature[5 * 1 + 4], this_feature[5 * 2 + 4]]
        item_height_stored = [this_feature[5 * 0 + 3], this_feature[5 * 1 + 3], this_feature[5 * 2 + 3]]
        item_center_stored = np.array([[this_feature[5 * 0 + 0], this_feature[5 * 0 + 1]],
                                       [this_feature[5 * 1 + 0], this_feature[5 * 1 + 1]],
                                       [this_feature[5 * 2 + 0], this_feature[5 * 2 + 1]]])
        item_angle_stored = [this_feature[5 * 0 + 2], this_feature[5 * 1 + 2], this_feature[5 * 2 + 2]]
        item_width_in_hand = this_feature[16]
        item_height_in_hand = this_feature[15]

        prob_success, cost_ret, time_ret, X_ret = fcn_mode_clustered_solver_pinned(bin_width, bin_height,
                                        bin_left, bin_right, bin_ground, bin_up, v_bin, num_of_item, item_width_stored,
                                         item_height_stored, item_center_stored, item_angle_stored, item_width_in_hand,
                                         item_height_in_hand, y_guess, iter_data)

        all_successful.append(prob_success)

    print(all_successful)


if __name__ == "__main__":
    main()
