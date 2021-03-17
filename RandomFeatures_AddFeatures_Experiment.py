import os
import argparse
import time
import pickle
import numpy as np

from src import Config, RandomFeatures, LinearFunctionApproximator
from src import SGD, Adam, IDBD, AutoStep, SIDBD

BEST_PARAMETER_VALUE = {
    'sgd': 0.005,           # found by sweeping over values in {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001}
    'adam': 0.005,          # found by sweeping over values in {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001}
    'idbd': 0.5,            # found by sweeping over values in {1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001}
    'autostep': 0.05,       # found by sweeping over values in {0.5, 0.1, 0.05, 0.01, 0.005, 0.001}
    'rescaled_sgd': 0.01,   # found by sweeping over values in {0.5, 0.1, 0.05, 0.01, 0.005, 0.001}
    'sidbd': 0.5,
}

OPTIMIZER_DICT = {'sgd': SGD, 'adam': Adam, 'idbd': IDBD, 'autostep': AutoStep, 'rescaled_sgd': SGD, 'sidbd': SIDBD}

TRAINING_DATA_SIZE = 100000     # number of training examples per run
MIDPOINT = 50000                # number of iterations for first phase of training
ADD_FEATURE_INTERVAL = 500      # number of iterations before adding another feature when 'continuously_add_bad'
CHECKPOINT = 50                 # how often store the mean squared error
STEPSIZE_GROWTH_FACTOR = 10


class Experiment:

    def __init__(self, exp_arguments, results_path):

        self.config = Config()

        self.experiment_type = exp_arguments.experiment_type
        self.sample_size = exp_arguments.sample_size
        self.verbose = exp_arguments.verbose
        self.results_path = results_path
        self.method = exp_arguments.method
        self.noisy = exp_arguments.noisy
        " Environment Setup "
        self.config.num_true_features = exp_arguments.num_true_features
        self.config.num_obs_features = exp_arguments.num_true_features - 1
        self.config.max_num_features = 250  # as long as it is more than 101

        " Optimizer Setup "
        self.config.parameter_size = self.config.num_obs_features

        if self.method in ['sgd', 'rescaled_sgd']:
            self.config.alpha = BEST_PARAMETER_VALUE[self.method]
            self.config.rescale = (self.method == 'rescaled_sgd')
        elif self.method == 'adam':
            self.config.beta1 = 0.9
            self.config.beta2 = 0.99
            self.config.eps = 1e-08
            self.config.init_alpha = BEST_PARAMETER_VALUE[self.method]
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

    def run(self):

        results_dir = {'sample_size': self.sample_size}
        alphas, names = self.get_alphas_and_names()

        for a, alpha in enumerate(alphas):
            self._print("Currently working on: {0}".format(names[a]))
            results = np.zeros((self.sample_size, TRAINING_DATA_SIZE // CHECKPOINT), dtype=np.float64)
            if self.method != 'sgd':
                stepsizes = np.zeros((self.sample_size, TRAINING_DATA_SIZE // CHECKPOINT, self.config.max_num_features),
                                     dtype=np.float64)

            for i in range(self.sample_size):
                self._print("\tCurrent sample: {0}".format(i+1))
                env = RandomFeatures(self.config)
                approximator = LinearFunctionApproximator(self.config)
                optimizer = OPTIMIZER_DICT[self.method](self.config)

                mse = 0
                current_checkpoint = 0
                current_stepsizes = np.zeros(self.config.max_num_features, dtype=np.float64)
                for j in range(TRAINING_DATA_SIZE):
                    target, observable_features, best_approximation = env.sample_observation(noisy=self.noisy)
                    prediction = approximator.get_prediction(observable_features)
                    error = target - prediction
                    _, ss, new_weights = optimizer.update_weight_vector(error, observable_features,
                                                                                approximator.get_weight_vector())
                    approximator.update_weight_vector(new_weights)

                    mse += np.square(error) / CHECKPOINT
                    if self.method != 'sgd':
                        current_stepsizes[:ss.size] += ss / CHECKPOINT
                    if (j + 1) % CHECKPOINT == 0:
                        results[i][current_checkpoint] += mse
                        mse *= 0
                        if self.method != 'sgd':
                            stepsizes[i, current_checkpoint] += current_stepsizes
                            current_stepsizes *= 0
                        current_checkpoint += 1

                    self.add_features(env, approximator, optimizer, alpha, j)

            results_dir[names[a]] = results
            if self.method != 'sgd':
                results_dir['stepsizes'] = stepsizes

            # agg_results = np.average(results, axis=0)
            # import matplotlib.pyplot as plt
            # plt.plot(np.arange(agg_results.size)+1, agg_results)
            # plt.show()
            # plt.close()

        self.store_results(results_dir)

    def get_alphas_and_names(self):
        if self.method in ['adam', 'idbd', 'autostep', 'rescaled_sgd', 'sidbd']:
            alphas = ['']
            names = [self.method]
            return alphas, names
        og_alphas = np.array((BEST_PARAMETER_VALUE[self.method] / STEPSIZE_GROWTH_FACTOR,
                             BEST_PARAMETER_VALUE[self.method],
                             BEST_PARAMETER_VALUE[self.method] * STEPSIZE_GROWTH_FACTOR))
        og_names = ['small', 'med', 'large']
        if self.experiment_type in ['add_good_feat', 'add_bad_feat', 'continuously_add_bad']:
            alphas = og_alphas
            names = [i + '_stepsize' for i in og_names]
        elif self.experiment_type in ['add_one_good_one_bad', 'add_one_good_10_bad', 'add_one_good_100_bad',
                                      'add_one_good_1000_bad']:
            alphas = []
            names = []
            for i in range(len(og_alphas)):     # every combination of (x,y) for x,y in {'small', 'med', 'large'}
                for j in range(len(og_alphas)):
                    alphas.append((og_alphas[i], og_alphas[j]))
                    names.append('good-' + og_names[i] + '_bad-' + og_names[j])
        else:
            raise ValueError("{0} is not a valid experiment type.".format(self.experiment_type))
        return alphas, names

    def add_features(self, env: RandomFeatures, approximator, optimizer, alpha, iteration_number):
        if self.is_phased_training() and iteration_number + 1 == MIDPOINT:
            if self.experiment_type in ['add_good_feat', 'add_bad_feat']:
                env.add_new_feature(1, true_feature=self.experiment_type == 'add_good_feat')
                approximator.increase_num_features(1)
                if self.method == 'sgd':
                    assert isinstance(alpha, float)
                    optimizer.increase_size(1, init_stepsize=alpha)
                else: optimizer.increase_size(1)
            elif self.experiment_type in ['add_one_good_one_bad', 'add_one_good_10_bad', 'add_one_good_100_bad',
                                          'add_one_good_1000_bad']:
                if self.experiment_type == 'add_one_good_one_bad':
                    k = 2
                elif self.experiment_type == 'add_one_good_10_bad':
                    k = 11
                elif self.experiment_type == 'add_one_good_100_bad':
                    k = 101
                else:
                    k = 1001
                env.add_new_feature(1, true_feature=True)       # add 1 true feature
                env.add_new_feature(k-1, true_feature=False)    # add k - 1 fake features
                approximator.increase_num_features(k)
                if self.method == 'sgd':
                    assert isinstance(alpha, tuple) and len(alpha) == 2
                    new_stepsizes = np.ones(k)
                    new_stepsizes[0] *= alpha[0]
                    new_stepsizes[1:] *= alpha[1]
                    optimizer.increase_size(k, new_stepsizes)
                else: optimizer.increase_size(k)
            else:
                raise ValueError("{0} is not a valid experiment type.".format(self.experiment_type))
        elif self.experiment_type == 'continuously_add_bad' and (iteration_number + 1) % ADD_FEATURE_INTERVAL == 0:
            env.add_new_feature(1, true_feature=False)
            approximator.increase_num_features(1)
            if self.method == 'sgd':
                assert isinstance(alpha, float)
                optimizer.increase_size(1, init_stepsize=alpha)
            else: optimizer.increase_size(1)

    def is_phased_training(self):
        phased_training = self.experiment_type in ['add_good_feat', 'add_bad_feat', 'add_one_good_one_bad',
                                                   'add_one_good_10_bad', 'add_one_good_100_bad',
                                                   'add_one_good_1000_bad']
        return phased_training

    def store_results(self, results):
        results_filepath = os.path.join(self.results_path, 'results.p')
        with open(results_filepath, mode='wb') as results_file:
            pickle.dump(results, results_file)
        print("Results were successfully stored.")


def main():
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ss', '--sample_size', action='store', default=1, type=int)
    parser.add_argument('-et', '--experiment_type', action='store', default='add_good_feature',
                            choices=['add_good_feat', 'add_bad_feat', 'add_one_good_one_bad', 'add_one_good_10_bad',
                                     'add_one_good_100_bad', 'add_one_good_1000_bad', 'continuously_add_bad'])
    parser.add_argument('-m', '--method', action='store', default='sgd', type=str,
                        choices=['sgd', 'adam', 'idbd', 'autostep', 'rescaled_sgd', 'sidbd'])
    parser.add_argument('-ntf', '--num_true_features', action='store', default=4, type=int)
    parser.add_argument('--noisy', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true')
    exp_parameters = parser.parse_args()

    experiment_name = 'random_features_task_add_features'
    if exp_parameters.noisy:
        experiment_name = 'noisy_' + experiment_name
    results_path = os.path.join(os.getcwd(), 'results', experiment_name,
                                'num_true_features_' + str(exp_parameters.num_true_features), exp_parameters.method,
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
