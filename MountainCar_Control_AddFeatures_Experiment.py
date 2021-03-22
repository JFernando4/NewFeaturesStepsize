import os
import argparse
import time
import pickle
import numpy as np
import copy

from src import Config, MountainCar, LinearFunctionApproximator, RadialBasisFunction, moving_sum
from src import SGD, Adam, IDBD, AutoStep, SIDBD

BEST_PARAMETER_VALUE = {
    'sgd': 0.0075,          # found by sweeping over values in {0.25 0.2 0.1 0.05 0.03 0.01 0.0075 0.005 0.003 0.001}
    'adam': 0.02,           # found by sweeping over values in {0.06 0.05 0.04 0.03 0.02 0.01 0.009 0.008 0.007 0.005}
    'slow_adam': 0.03,      # found by sweeping over values in {0.1 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.02 0.01}
    'idbd': 0.02,           # found by sweeping over values in {0.045 0.04 0.035 0.03 0.025 0.02 0.015 0.01 0.005 0.001}
    'autostep': 0.009,      # found by sweeping over values in {0.05 0.04 0.03 0.02 0.01 0.009 0.008 0.007 0.006 0.005}
    'rescaled_sgd': 0.01,   # found by sweeping over values in {0.1 0.05 0.03 0.02 0.01 0.009 0.008 0.007 0.005 0.001}
    'restart_adam': 0.02,   # same as adam
    'sidbd': 0.02,          # found by sweeping over values in {0.05 0.04 0.035 0.03 0.025 0.02 0.015 0.01 0.007 0.005}
}

OPTIMIZER_DICT = {'sgd': SGD, 'adam': Adam, 'idbd': IDBD, 'autostep': AutoStep, 'rescaled_sgd': SGD,
                  'restart_adam': Adam, 'sidbd': SIDBD, 'slow_adam': Adam}

TRAINING_DATA_SIZE = 200000     # number of training examples per run
MIDPOINT = 100000               # number of iterations for first phase of training
ADD_FEATURE_INTERVAL = 1000     # number of iterations before adding another feature when 'continuously_add_bad'
CHECKPOINT = 1000               # how often store the mean squared error
STEPSIZE_GROWTH_FACTOR = 10     # how much to increase or decrease the stepsize for sgd
DEBUG = False


class Experiment:

    def __init__(self, exp_arguments, results_path):

        self.config = Config()

        self.experiment_type = exp_arguments.experiment_type
        self.phased_training = (self.experiment_type != 'continuously_add_bad')
        self.sample_size = exp_arguments.sample_size
        self.verbose = exp_arguments.verbose
        self.results_path = results_path
        self.method = exp_arguments.method
        self.add_fake_features = exp_arguments.add_fake_features
        self.num_transitions = TRAINING_DATA_SIZE if not exp_arguments.extended_training else TRAINING_DATA_SIZE * 2
        self.baseline = exp_arguments.baseline

        " Feature Function Setup "
        self.config.state_dims = 2                                              # dimensions in mountain car
        self.config.state_lims = np.array(((-1,1), (-1,1)), dtype=np.float64)   # state bounds in mountain car
        # centers for the Radial Basis Functions
        self.config.initial_centers = np.array(((0,0),(.25,.25),(.25,-.25),(-.25,-.25),(-.25,.25)), dtype=np.float64)
        self.config.sigma = 0.5                 # width of each feature
        self.config.init_noise_mean = 0.0       # mean and variance for the
        self.config.init_noise_var = 0.01 if not self.baseline else 0.0     # noise of each feature

        " Environment Setup "
        self.config.norm_state = True       # normalized the state between -1 and 1
        self.num_actions = 3                # number of actions in the mountain car environment
        # number of observable features at the start of training
        self.config.num_obs_features = self.config.initial_centers.shape[0]
        # max_num_features should be at least num_obs_features + num_new_features
        self.config.max_num_features = self.config.num_obs_features + self.get_num_new_features()
        self.epsilon = 0.1                  # reasonable choice in mountain car
        self.gamma = 0.99                   # discount factor

        " Optimizer Setup "
        self.config.parameter_size = self.config.num_obs_features
        if self.method in ['sgd', 'rescaled_sgd']:
            self.config.alpha = BEST_PARAMETER_VALUE[self.method]
            self.config.rescale = (self.method == 'rescaled_sgd')
        elif self.method in ['adam', 'restart_adam', 'slow_adam']:
            self.config.beta1 = 0.9 if self.method in ['adam', 'restart_adam'] else 0.0
            self.config.beta2 = 0.99
            self.config.eps = 1e-08
            self.config.init_alpha = BEST_PARAMETER_VALUE[self.method]
            self.config.restart_ma = (self.method == 'restart_adam')
        elif self.method in ['idbd', 'sidbd']:
            self.config.init_beta = np.log(0.001) if self.method == 'idbd' else -np.log((1/0.001) - 1)
            self.config.theta = BEST_PARAMETER_VALUE[self.method]
        elif self.method == 'autostep':
            self.config.tau = 10000.0
            self.config.init_stepsize = 0.001
            self.config.mu = BEST_PARAMETER_VALUE[self.method]
        else:
            raise ValueError("{0} is not a valid stepsize adaptation method.".format(exp_arguments.method))

    def _print(self, some_string):
        if self.verbose:
            print(some_string)

    def setup_baseline(self, alpha_value):
        if self.method == 'sgd' and self.baseline:
            self.config.alpha = alpha_value

    def run(self):
        results_dir = {'sample_size': self.sample_size}
        alphas, names = self.get_alphas_and_names()

        for a, alpha in enumerate(alphas):
            self._print("Currently working on: {0}".format(names[a]))
            self.setup_baseline(alpha)

            # For measuring performance
            reward_per_step = np.zeros((self.sample_size, self.num_transitions), dtype=np.int8)
            diverging_runs = np.zeros(self.sample_size, dtype=np.int8)
            action_counter = np.zeros((self.sample_size, self.num_transitions // CHECKPOINT,
                                       self.num_actions), dtype=np.int32)
            weight_sum_per_checkpoint = np.zeros((self.sample_size, self.num_transitions // CHECKPOINT,
                                                  self.num_actions, self.config.max_num_features), dtype=np.float64)
            # For keeping track of stepsizes
            if self.method != 'sgd':
                stepsize_sum_per_checkpoint = np.zeros((self.sample_size, self.num_transitions // CHECKPOINT,
                                                        self.num_actions, self.config.max_num_features),
                                                       dtype=np.float64)
            # start processing samples
            for i in range(self.sample_size):
                np.random.seed(i)
                self._print("\tCurrent sample: {0}".format(i+1))

                ff = RadialBasisFunction(self.config)   # feature function
                env = MountainCar(self.config)          # environment: Mountain Car
                approximators = []                      # one function approximator per action
                optimizers = []                         # one optimizer per action
                for _ in range(self.num_actions):
                    approximators.append(LinearFunctionApproximator(self.config))
                    optimizers.append(OPTIMIZER_DICT[self.method](self.config))
                # learning_approximators = approximators
                # pe_phase = False    # pe = policy evaluation
                # curr_pe_iterations = 0
                # total_pe_iterations = 100

                curr_checkpoint = 0
                """ Start of Training """
                curr_s = env.get_current_state()                                    # current state
                curr_obs_feats = ff.get_observable_features(curr_s)                 # current observable features
                for j in range(self.num_transitions):
                    curr_avs = self.get_action_values(curr_obs_feats, approximators)    # current action values
                    curr_a = self.epsilon_greedy_policy(curr_avs)                       # current action
                    # execute action
                    next_s, r, terminal = env.step(curr_a)               # r = reward
                    # compute next action values and action
                    next_obs_feats = ff.get_observable_features(next_s)
                    next_avs = self.get_action_values(next_obs_feats, approximators)
                    next_a = self.epsilon_greedy_policy(next_avs)
                    # compute TD error of Sarsa(0)
                    # curr_av = learning_approximators[curr_a].get_prediction(curr_obs_feats)
                    # next_av = learning_approximators[next_a].get_prediction(next_obs_feats)
                    # td_error = r + self.gamma * (1-terminal) * next_av - curr_av
                    # _, ss, new_weights = optimizers[curr_a].update_weight_vector(td_error, curr_obs_feats,
                    #                                                 learning_approximators[curr_a].get_weight_vector())
                    # learning_approximators[curr_a].update_weight_vector(new_weights)
                    # if pe_phase:
                    #     curr_pe_iterations += 1
                    #     if curr_pe_iterations == total_pe_iterations:
                    #         pe_phase = False
                    #         curr_pe_iterations = 0
                    #         approximators = learning_approximators#copy.deepcopy(learning_approximators)

                    td_error = r + self.gamma * (1-terminal) * next_avs[next_a] - curr_avs[curr_a]
                    _, ss, new_weights = optimizers[curr_a].update_weight_vector(td_error, curr_obs_feats,
                                                                        approximators[curr_a].get_weight_vector())
                    # update weight vector
                    approximators[curr_a].update_weight_vector(new_weights)
                    # update state information and progress
                    curr_obs_feats = next_obs_feats
                    reward_per_step[i][j] += np.int8(r)
                    action_counter[i][curr_checkpoint][curr_a] += np.int32(1)
                    weight_sum_per_checkpoint[i][curr_checkpoint][curr_a][:new_weights.size] += new_weights
                    if self.method != 'sgd':
                        stepsize_sum_per_checkpoint[i][curr_checkpoint][curr_a][:ss.size] += ss

                    # handle cases where weights diverge
                    if np.sum(np.isnan(new_weights)) > 0 or np.sum(np.isinf(new_weights)) > 0:
                        print("\tThe weights diverged on iteration: {0}!".format(j+1))
                        reward_per_step[i][j+1:] += np.int8(-1)
                        diverging_runs[i] += np.int8(1)
                        break

                    # check if terminal state
                    if terminal:
                        env.reset()
                        curr_s = env.get_current_state()
                        curr_obs_feats = ff.get_observable_features(curr_s)

                    # process checkpoints
                    if (j + 1) % CHECKPOINT == 0:
                        curr_checkpoint += 1

                    if self.add_features(ff, approximators, optimizers, alpha, j):
                        curr_obs_feats = ff.get_observable_features(curr_s)
                        # learning_approximators = copy.deepcopy(approximators)
                        # pe_phase = True

            results_dir[names[a]] = {'reward_per_step': reward_per_step,
                                     'diverging_runs': diverging_runs,
                                     'action_counter': action_counter,
                                     'weight_sum_per_checkpoint': weight_sum_per_checkpoint}
            if self.method != 'sgd':
                results_dir[names[a]]['stepsize_sum_per_checkpoint'] = stepsize_sum_per_checkpoint

            if DEBUG:
                agg_results = np.average(reward_per_step, axis=0)
                ms = moving_sum(agg_results, n=CHECKPOINT) + CHECKPOINT
                import matplotlib.pyplot as plt
                plt.plot(np.arange(ms.size)+1, ms)
                plt.vlines(MIDPOINT, ymin=ms.min(), ymax=ms.max())
                plt.show()
                plt.close()

        self.store_results(results_dir)

    def get_alphas_and_names(self):
        # If not using SGD, we don't need to se the stepsize of new features manually
        if self.method in ['adam', 'idbd', 'autostep', 'rescaled_sgd', 'restart_adam', 'sidbd', 'slow_adam']:
            alphas = ['']
            names = [self.method]
            return alphas, names
        # Otherwise, set the stepsize of the new features to small, medium, or large
        og_alphas = np.array((BEST_PARAMETER_VALUE[self.method] / STEPSIZE_GROWTH_FACTOR,   # small stepsize
                             BEST_PARAMETER_VALUE[self.method],                             # medium
                             BEST_PARAMETER_VALUE[self.method] * STEPSIZE_GROWTH_FACTOR))   # large
        og_names = ['small', 'med', 'large']
        if self.experiment_type in ['add_good_feats', 'add_bad_feats', 'continuously_add_bad', 'continuously_add_random']:
            alphas = og_alphas
            names = [i + '_stepsize' for i in og_names]
        elif self.experiment_type in ['add_5_good_5_bad', 'add_5_good_20_bad', 'add_5_good_100_bad']:
            alphas = []
            names = []
            for i in range(len(og_alphas)):     # every combination of (x,y) for x,y in {'small', 'med', 'large'}
                for j in range(len(og_alphas)):
                    alphas.append((og_alphas[i], og_alphas[j]))
                    names.append('good-' + og_names[i] + '_bad-' + og_names[j])
        else:
            raise ValueError("{0} is not a valid experiment type.".format(self.experiment_type))
        return alphas, names

    def add_features(self, feat_func: RadialBasisFunction, approximator_list, optimizer_list, alpha, iteration_number):

        if self.baseline:
            return False

        # Add features half way through training
        if self.phased_training and iteration_number + 1 == MIDPOINT:
            num_good_features = 0
            num_bad_features = 0

            # add 45 new good centers to the feature function
            if self.experiment_type in ['add_good_feats','add_5_good_5_bad','add_5_good_20_bad','add_5_good_100_bad']:
                feat_func.add_feature(5, noise_mean=0, noise_var=0, fake_feature=False)
                num_good_features += 5

            # add irrelevant features to the feature function
            if self.experiment_type in ['add_bad_feats','add_5_good_5_bad','add_5_good_20_bad','add_5_good_100_bad']:
                if self.experiment_type in ['add_bad_feats', 'add_5_good_5_bad']:
                    num_bad_features += 5
                elif self.experiment_type == 'add_5_good_20_bad':
                    num_bad_features += 20
                elif self.experiment_type == 'add_5_good_100_bad':
                    num_bad_features += 100
                feat_func.add_feature(num_bad_features, noise_mean=0.0, noise_var=5*self.config.init_noise_var,
                                      fake_feature=self.add_fake_features)

            # add features to the optimizer and the function approximator
            num_new_features = num_good_features + num_bad_features
            for k in range(self.num_actions):
                approximator_list[k].increase_num_features(num_new_features)

                if self.method != 'sgd':    # if method is not sgd, stepsizes are set automatically
                    optimizer_list[k].increase_size(num_new_features)
                else:                       # otherwise, set the stepsize manually
                    if isinstance(alpha, tuple):
                        # if adding good and bad features, add good features with stepsize = alpha[0]
                        # and bad features with stepsize = alpha[1]
                        good_features_stepsize, bad_features_stepsize = alpha
                        optimizer_list[k].increase_size(num_good_features, good_features_stepsize)
                        optimizer_list[k].increase_size(num_bad_features, bad_features_stepsize)
                    else:
                        # otherwise, alpha is a scalar
                        optimizer_list[k].increase_size(num_new_features, alpha)
            return True     # True indicates that features were added to the representation

        # Add features every certain number of steps specified by ADD_FEATURE_INTERVAL
        elif self.experiment_type == 'continuously_add_bad' and (iteration_number + 1) % ADD_FEATURE_INTERVAL == 0:
            # add bad features
            feat_func.add_feature(1, noise_mean=0.0, noise_var=5*self.config.init_noise_var,
                                  fake_feature=self.add_fake_features)

            for k in range(self.num_actions):   # extend function approximator and optimizer
                approximator_list[k].increase_num_features(1)
                if self.method != 'sgd':        # if not sgd, stepsizes are set automatically
                    optimizer_list[k].increase_size(1)
                else:                           # if sgd, stepsizes are set manually
                    optimizer_list[k].increase_size(1, init_stepsize=alpha)
            return True     # True indicates that features were added to the representation

        return False        # False indicates that no feature was added to the representation

    def store_results(self, results):
        results_filepath = os.path.join(self.results_path, 'results.p')
        with open(results_filepath, mode='wb') as results_file:
            pickle.dump(results, results_file)
        print("Results were successfully stored.")

    def get_action_values(self, features, approximators):
        action_values = np.zeros(self.num_actions, dtype=np.float64)
        for k in range(self.num_actions):
            action_values[k] += approximators[k].get_prediction(features)
        return action_values

    def epsilon_greedy_policy(self, action_values:np.ndarray):
        p = np.random.rand()
        if p > self.epsilon:
            argmax_av = np.random.choice(np.flatnonzero(action_values == action_values.max()))
            return argmax_av
        else:
            return np.random.randint(self.num_actions)

    def get_num_new_features(self):
        if self.experiment_type in ['add_good_feats', 'add_bad_feats']:
            return 5
        elif self.experiment_type == 'add_5_good_5_bad':
            return 10
        elif self.experiment_type == 'add_5_good_20_bad':
            return 25
        elif self.experiment_type == 'add_5_good_100_bad':
            return 105
        elif self.experiment_type == 'continuously_add_bad':
            return 200
        else:
            raise ValueError("{0} is not a valid experiment type.".format(self.experiment_type))


def main():
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ss', '--sample_size', action='store', default=1, type=int)
    parser.add_argument('-et', '--experiment_type', action='store', default='add_good_feats',
                            choices=['add_good_feats', 'add_bad_feats', 'add_5_good_5_bad', 'add_5_good_20_bad',
                                     'add_5_good_100_bad', 'continuously_add_bad'])
    parser.add_argument('-m', '--method', action='store', default='sgd', type=str,
                        choices=['sgd', 'adam', 'idbd', 'autostep', 'rescaled_sgd', 'restart_adam', 'sidbd',
                                 'slow_adam'])
    parser.add_argument('-aff', '--add_fake_features', action='store_true', default=False)
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='Baseline where the initial features do not have noise and new features are not added.')
    parser.add_argument('-etr', '--extended_training', action='store_true', default=False,
                        help='Add features after 100k steps and then trains for another 300k steps.')
    parser.add_argument('-v', '--verbose', action='store_true')
    exp_parameters = parser.parse_args()

    if exp_parameters.baseline:
        experiment_name = 'mountain_car_control_baseline'
        results_path = os.path.join(os.getcwd(), 'results', experiment_name, exp_parameters.method)
    else:
        experiment_name = 'mountain_car_control_task_add'
        if exp_parameters.add_fake_features:
            experiment_name += '_fake'
        experiment_name += '_features'
        if exp_parameters.extended_training:
            experiment_name += '_extended_training'
        results_path = os.path.join(os.getcwd(), 'results', experiment_name, exp_parameters.method,
                                    exp_parameters.experiment_type)

    os.makedirs(results_path, exist_ok=True)

    init_time = time.time()
    exp = Experiment(exp_parameters, results_path)
    exp.run()
    finish_time = time.time()
    elapsed_time = (finish_time - init_time) / 60
    print("Running time in minutes: {0}".format(elapsed_time))


if __name__ == '__main__':
    main()
