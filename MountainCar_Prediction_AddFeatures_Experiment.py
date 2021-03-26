import os
import argparse
import time
import pickle
import numpy as np
import copy

from src import Config, LinearFunctionApproximator, RadialBasisFunction
from src import SGD, Adam, IDBD, AutoStep, SIDBD

BEST_PARAMETER_VALUE = {
    'sgd': 0.04,            # found by sweeping over values in {0.1 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.02 0.01}
    'adam': 0.2,            # found by sweeping over values in {0.5 0.4 0.3 0.2 0.1 0.09 0.08 0.07 0.06 0.05}
    'slow_adam': 0.08,      # found by sweeping over values in {0.5 0.3 0.2 0.1 0.09 0.08 0.07 0.06 0.05 0.01}
    # found by sweeping over values in {0.02 0.01 0.009 0.008 0.007 0.006 0.005 0.004 0.003 0.002}
    **dict.fromkeys(['keep_idbd', 'reset_idbd', 'max_idbd'], 0.02),
    'autostep': 0.02,       # found by sweeping over values in {0.04 0.03 0.02 0.01 0.009 0.008 0.007 0.006 0.005 0.001}
    'rescaled_sgd': 0.05,   # found by sweeping over values in {0.5 0.1 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.01}
    'restart_adam': 0.2,    # same as adam
    # found by sweeping over values in {0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.02 0.01 0.009}
    **dict.fromkeys(['keep_sidbd', 'reset_sidbd', 'max_sidbd'], 0.09),
    'dense_baseline': 0.02, # found by sweeping over values in {}
}

OPTIMIZER_DICT = {
    **dict.fromkeys(['sgd', 'rescaled_sgd', 'dense_baseline'], SGD),
    **dict.fromkeys(['adam', 'restart_adam', 'slow_adam'], Adam),
    **dict.fromkeys(['keep_sidbd', 'reset_sidbd', 'max_sidbd'], SIDBD),
    **dict.fromkeys(['keep_idbd', 'reset_idbd', 'max_idbd'], IDBD),
    'autostep': AutoStep,
                  }

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
        self.num_transitions = TRAINING_DATA_SIZE
        self.baseline = exp_arguments.baseline

        self.data_path = os.path.join(os.getcwd(), 'mountain_car_prediction_data_30evaluations')
        self.init_seed = 30     # the first 30 seeds were used for parameter tuning
        # checks if there are enough data files
        assert len(os.listdir(self.data_path)) - self.init_seed >= self.sample_size

        " Feature Function Setup "
        self.config.state_dims = 2                                              # dimensions in mountain car
        self.config.state_lims = np.array(((-1,1), (-1,1)), dtype=np.float64)   # state bounds in mountain car

        # centers for the Radial Basis Functions
        if self.method == 'dense_baseline':
            # 121 centers distributed evenly across the state space
            x = np.arange(-1,1.2, 2/10)
            self.config.initial_centers = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
        else:
            # 5 centers distributed evenly around (0,0)
            self.config.initial_centers = np.array(((0,0),(.25,.25),(.25,-.25),(-.25,-.25),(-.25,.25)),dtype=np.float64)

        self.config.sigma = 0.5                 # width of each feature
        self.config.init_noise_mean = 0.0       # mean of the noise of each features
        if self.baseline or (self.method == 'dense_baseline'):  # variance of the noise of each feature
            self.config.init_noise_var = 0.0
        else:
            self.config.init_noise_var = 0.01

        " Environment Setup "
        self.num_actions = 3                # number of actions in the mountain car environment
        # number of observable features at the start of training
        self.config.num_obs_features = self.config.initial_centers.shape[0]
        # max_num_features should be at least num_obs_features + num_new_features
        self.config.max_num_features = self.config.num_obs_features + self.get_num_new_features()
        self.gamma = 0.99                   # discount factor

        " Optimizer Setup "
        self.config.parameter_size = self.config.num_obs_features
        if self.method in ['sgd', 'rescaled_sgd', 'dense_baseline']:
            self.config.alpha = BEST_PARAMETER_VALUE[self.method]
            self.config.rescale = (self.method == 'rescaled_sgd')
        elif self.method in ['adam', 'restart_adam', 'slow_adam']:
            self.config.beta1 = 0.9 if self.method in ['adam', 'restart_adam'] else 0.0
            self.config.beta2 = 0.99
            self.config.eps = 1e-08
            self.config.init_alpha = BEST_PARAMETER_VALUE[self.method]
            self.config.restart_ma = (self.method == 'restart_adam')
        elif self.method in ['keep_idbd', 'reset_idbd', 'max_idbd', 'keep_sidbd', 'reset_sidbd', 'max_sidbd']:
            self.config.init_beta = np.log(0.001) if self.method == 'idbd' else -np.log((1/0.001) - 1)
            self.config.theta = BEST_PARAMETER_VALUE[self.method]
            self.config.increase_setting = self.method.split('_')[0]
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
            avg_mse_per_checkpoint = np.zeros((self.sample_size, self.num_transitions // CHECKPOINT), dtype=np.float64)
            diverging_runs = np.zeros(self.sample_size, dtype=np.int8)
            action_counter = np.zeros((self.sample_size, self.num_transitions // CHECKPOINT,
                                       self.num_actions), dtype=np.int32)
            weight_sum_per_checkpoint = np.zeros((self.sample_size, self.num_transitions // CHECKPOINT,
                                                  self.num_actions, self.config.max_num_features), dtype=np.float64)
            # For keeping track of stepsizes
            if self.method not in ['sgd', 'dense_baseline']:
                stepsize_sum_per_checkpoint = np.zeros((self.sample_size, self.num_transitions // CHECKPOINT,
                                                        self.num_actions, self.config.max_num_features),
                                                       dtype=np.float64)
            # start processing samples
            for i in range(self.sample_size):
                seed_number = self.init_seed + i
                self._print("\tCurrent Seed: {0}".format(seed_number))

                ff = RadialBasisFunction(self.config)   # feature function
                approximators = []                      # one function approximator per action
                optimizers = []                         # one optimizer per action
                for _ in range(self.num_actions):
                    approximators.append(LinearFunctionApproximator(self.config))
                    optimizers.append(OPTIMIZER_DICT[self.method](self.config))

                curr_checkpoint = 0
                # load data
                states, actions, rewards, terminations, avg_disc_return = self.load_data(seed_number=seed_number)
                """ Start of Training """
                curr_obs_feats = ff.get_observable_features(states[0])      # current observable features
                j = 0
                while j < self.num_transitions:
                    curr_a = actions[j]                                                 # current action
                    curr_av = approximators[curr_a].get_prediction(curr_obs_feats)      # current value
                    next_s = states[j+1]                # next state
                    next_r = rewards[j+1]               # next reward
                    next_term = terminations[j+1]       # next termination
                    next_a = actions[j+1]               # next action

                    # get next observable features and action-value
                    next_obs_feats = ff.get_observable_features(next_s)
                    next_av = approximators[next_a].get_prediction(next_obs_feats)
                    # compute TD error for Sarsa(0)
                    td_error = next_r + self.gamma * (1-next_term) * next_av - curr_av
                    # update weight vector
                    _, ss, new_weights = optimizers[curr_a].update_weight_vector(td_error, curr_obs_feats,
                                                                        approximators[curr_a].get_weight_vector())
                    approximators[curr_a].update_weight_vector(new_weights)

                    # handle cases where weights diverge
                    if np.sum(np.isnan(new_weights)) > 0 or np.sum(np.isinf(new_weights)) > 0:
                        print("\tThe weights diverged on iteration: {0}!".format(j+1))
                        avg_mse_per_checkpoint[i][curr_checkpoint:] += 1000000
                        diverging_runs[i] += np.int8(1)
                        break

                    # update state information and progress
                    curr_obs_feats = next_obs_feats
                    avg_mse_per_checkpoint[i][curr_checkpoint] += np.square(curr_av - avg_disc_return[j]) / CHECKPOINT
                    action_counter[i][curr_checkpoint][curr_a] += np.int32(1)
                    weight_sum_per_checkpoint[i][curr_checkpoint][curr_a][:new_weights.size] += new_weights
                    if self.method not in ['sgd', 'dense_baseline']:
                        stepsize_sum_per_checkpoint[i][curr_checkpoint][curr_a][:ss.size] += ss

                    # increase iteration number and process checkpoints
                    j += 1
                    if j % CHECKPOINT == 0: curr_checkpoint += 1

                    # add features
                    if self.add_features(ff, approximators, optimizers, alpha, j):
                        curr_obs_feats = ff.get_observable_features(states[j])

                    # handle terminal states
                    if next_term and j < self.num_transitions:
                        j += 1  # skips terminal state
                        if j % CHECKPOINT == 0: curr_checkpoint += 1
                        curr_obs_feats = ff.get_observable_features(states[j])

            results_dir[names[a]] = {'avg_mse_per_checkpoint': avg_mse_per_checkpoint,
                                     'diverging_runs': diverging_runs,
                                     'action_counter': action_counter,
                                     'weight_sum_per_checkpoint': weight_sum_per_checkpoint}
            if self.method not in ['sgd', 'dense_baseline']:
                results_dir[names[a]]['stepsize_sum_per_checkpoint'] = stepsize_sum_per_checkpoint

            if DEBUG:
                agg_results = np.average(avg_mse_per_checkpoint, axis=0)
                import matplotlib.pyplot as plt
                plt.plot((np.arange(agg_results.size) + 1) * CHECKPOINT, agg_results)
                plt.vlines(MIDPOINT, ymin=agg_results.min(), ymax=agg_results.max())
                plt.show()
                plt.close()

        self.store_results(results_dir)

    def get_alphas_and_names(self):
        # If not using SGD, we don't need to set the stepsize of new features manually
        if self.method != 'sgd':
            alphas = ['']
            names = [self.method]
            return alphas, names
        # Otherwise, set the stepsize of the new features to small, medium, or large
        og_alphas = np.array((BEST_PARAMETER_VALUE[self.method] / STEPSIZE_GROWTH_FACTOR,       # small
                              BEST_PARAMETER_VALUE[self.method],                                # medium
                              BEST_PARAMETER_VALUE[self.method] * STEPSIZE_GROWTH_FACTOR))      # large
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

        if self.baseline or (self.method == 'dense_baseline'):
            return False    # False indicates that no feature was added to the representation

        # Add features half way through training
        if self.phased_training and iteration_number == MIDPOINT:
            num_good_features = 0
            num_bad_features = 0

            # add new good centers to the feature function
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
        elif self.experiment_type == 'continuously_add_bad' and iteration_number % ADD_FEATURE_INTERVAL == 0:
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

    def load_data(self, seed_number):
        with open(os.path.join(self.data_path, 'seed' + str(seed_number) + '.p'), mode='rb') as data_file:
            data_dict = pickle.load(data_file)
        states = data_dict['states']
        actions = data_dict['actions']
        rewards = data_dict['rewards']
        terminations = data_dict['terminations']
        avg_discounted_return = data_dict['avg_discounted_return']
        return states, actions, rewards, terminations, avg_discounted_return

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
                        choices=['sgd', 'adam', 'idbd', 'autostep', 'rescaled_sgd', 'restart_adam',
                                 'keep_idbd', 'reset_idbd', 'max_idbd',
                                 'keep_sidbd', 'reset_sidbd', 'max_sidbd',
                                 'slow_adam', 'dense_baseline'])
    parser.add_argument('-aff', '--add_fake_features', action='store_true', default=False)
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='Add features after 100k steps and then trains for another 300k steps.')
    parser.add_argument('-v', '--verbose', action='store_true')
    exp_parameters = parser.parse_args()

    if exp_parameters.baseline:
        experiment_name = 'mountain_car_prediction_baseline'
        results_path = os.path.join(os.getcwd(), 'results', experiment_name, exp_parameters.method)
    else:
        experiment_name = 'mountain_car_prediction_task_add'
        if exp_parameters.add_fake_features:
            experiment_name += '_fake'
        experiment_name += '_features'
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
