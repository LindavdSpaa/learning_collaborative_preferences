"""
Implements maximum entropy inverse reinforcement learning (Ziebart et al., 2008)

Matthew Alger, 2015
matthew.alger@anu.edu.au

Modified to:
- update alpha only once, normalizing to max|alpha| = 1;
- considering finite trajectories, cropping expected_svf at terminal state
by Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

from itertools import product

import numpy as np
import numpy.random as rn

from . import value_iteration

def irl(feature_matrix, n_actions, discount, transition_probability,
        trajectories, alpha, learning_rate, epochs):
    """
    Find the reward function for the given trajectories.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    alpha: Array with feature weights to be np.max(np.abs(alpha)).max(np.abs(alpha))ated. NumPy array of length D.
    learning_rate: Gradient descent learning rate. float.
    epochs: Number of gradient descent steps. int.
    -> Reward vector with shape (N,).
    """

    n_states, d_states = feature_matrix.shape

    # Calculate the feature expectations \tilde{phi}.
    feature_expectations = find_feature_expectations(feature_matrix,
                                                     trajectories)

    # Gradient descent on alpha.
    for i in range(epochs):
        # print("i: {}".format(i))
        r = feature_matrix.dot(alpha)
        ## Random human policy is biased towards wrapped actions
        expected_svf = find_expected_svf(n_states, r, n_actions, discount,
                                         transition_probability, trajectories)
        grad = feature_expectations - feature_matrix.T.dot(expected_svf)

        ## Can never get perfect, especially if robot was more likely to do something different
        alpha += learning_rate * grad
        ## Normalize 
        if (maxAlpha := np.max(np.abs(alpha))) > 0:
            alpha /= maxAlpha

    return alpha

def find_svf(n_states, trajectories):
    """
    Find the state visitation frequency from trajectories.

    n_states: Number of states. int.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> State visitation frequencies vector with shape (N,).
    """

    svf = np.zeros(n_states)

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            svf[state] += 1

    svf /= trajectories.shape[0]

    return svf

def find_feature_expectations(feature_matrix, trajectories):
    """
    Find the feature expectations for the given trajectories. This is the
    average path feature vector.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Feature expectations vector with shape (D,).
    """

    feature_expectations = np.zeros(feature_matrix.shape[1])

    for trajectory in trajectories:
        for state, action, nextState in trajectory:
            feature_expectations += feature_matrix[state]

    feature_expectations /= trajectories.shape[0]

    return feature_expectations

def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories):
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.

    n_states: Number of states N. int.
    alpha: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    """

    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]

    policy = value_iteration.find_policy(n_states, n_actions,
                                         transition_probability, r, discount)

    start_state_count = np.zeros(n_states)
    terminal_state_set = []
    for trajectory in trajectories:
        start_state_count[trajectory[0, 0]] += 1
        
        if trajectory[-1,0] not in terminal_state_set:
            terminal_state_set += [trajectory[-1,0]]
    p_start_state = start_state_count/n_trajectories

    expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            if i not in terminal_state_set: # exclude transitions from terminal state
                expected_svf[k, t] += (expected_svf[i, t-1] * 
                                    policy[i, j] * # Stochastic policy
                                    transition_probability[i, j, k])

    return expected_svf.sum(axis=1)
