import os
import argparse
import time
import pickle
import numpy as np

from src import Config, BoyanChain, LinearFunctionApproximator
from src import SGD, Adam, IDBD, AutoStep, SIDBD

BEST_PARAMETER_VALUE = {
    'sgd': 0.01,            # found by sweeping over values in {0.5, 0.1, 0.05, 0.01, 0.005, 0.001}
    'adam': 0.001,          # found by sweeping over values in {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005}
    'idbd': 5.0,            # found by sweeping over values in {10.0, 5.0, 1.0, 0.5, 0.1, 0.05}
    'autostep': 0.05,       # found by sweeping over values in {0.5, 0.1, 0.05, 0.01, 0.005, 0.001}
    'rescaled_sgd': 0.01,   # found by sweeping over values in {0.5, 0.1, 0.05, 0.01, 0.005, 0.001}
    'restart_adam': 0.001,  # same as adam
    'sidbd': 10.0,          # found by sweeping over values in {50.0 10.0 5.0 1.0 0.5 0.1}
}

OPTIMIZER_DICT = {'sgd': SGD, 'adam': Adam, 'idbd': IDBD, 'autostep': AutoStep, 'rescaled_sgd': SGD,
                  'restart_adam': Adam, 'sidbd': SIDBD}

TRAINING_DATA_SIZE = 200000     # number of training examples per run
MIDPOINT = 100000               # number of iterations for first phase of training
ADD_FEATURE_INTERVAL = 1000     # number of iterations before adding another feature when 'continuously_add_bad'
CHECKPOINT = 100                # how often store the mean squared error
STEPSIZE_GROWTH_FACTOR = 10


class Experiment:

    def __init__(self, exp_arguments, results_path):

        self.config = Config()

        self.experiment_type = exp_arguments.experiment_type
        self.sample_size = exp_arguments.sample_size
        self.verbose = exp_arguments.verbose
        self.results_path = results_path
        self.method = exp_arguments.method
        self.feature_noise = exp_arguments.feature_noise
        self.bad_feature_noise = self.feature_noise * 2.0
        self.add_fake_feature = exp_arguments.add_fake_features
        " Environment Setup "
        self.num_true_features = 4  # the boyan chain environment has only 4 features
        self.config.init_noise_var = self.feature_noise
        self.config.num_obs_features = 4
        self.config.max_num_features = 210  # as long as it is more than 204

        " Optimizer Setup "
        self.config.parameter_size = self.config.num_obs_features

        if self.method in ['sgd', 'rescaled_sgd']:
            self.config.alpha = BEST_PARAMETER_VALUE[self.method]
            self.config.rescale = (self.method == 'rescaled_sgd')
        elif self.method in ['adam', 'restart_adam']:
            self.config.beta1 = 0.9
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

    def run(self):

        results_dir = {'sample_size': self.sample_size}
        alphas, names = self.get_alphas_and_names()

        for a, alpha in enumerate(alphas):
            self._print("Currently working on: {0}".format(names[a]))
            sample_msve_per_run = np.zeros((self.sample_size, TRAINING_DATA_SIZE // CHECKPOINT), dtype=np.float64)
            average_weight_per_checkpoint = np.zeros((self.sample_size, TRAINING_DATA_SIZE // CHECKPOINT,
                                                      self.config.max_num_features), dtype=np.float64)
            average_stepsize_per_checkpoint = np.zeros((self.sample_size, TRAINING_DATA_SIZE // CHECKPOINT,
                                                        self.config.max_num_features), dtype=np.float64)

            for i in range(self.sample_size):
                np.random.seed(i)
                self._print("\tCurrent sample: {0}".format(i+1))
                env = BoyanChain(self.config)
                approximator = LinearFunctionApproximator(self.config)
                optimizer = OPTIMIZER_DICT[self.method](self.config)

                """ Start of Training """
                curr_checkpoint = 0
                for j in range(TRAINING_DATA_SIZE):
                    current_obs_features = env.get_observable_features()
                    state_value = approximator.get_prediction(current_obs_features)
                    optimal_value = env.compute_true_value()

                    _, reward, next_obs_features, terminal = env.step()

                    next_state_value = approximator.get_prediction(next_obs_features)
                    td_error = reward + (1 - terminal) * next_state_value - state_value
                    _, ss, new_weights = optimizer.update_weight_vector(td_error, current_obs_features,
                                                                        approximator.get_weight_vector())
                    approximator.update_weight_vector(new_weights)

                    # store summaries and update checkpoint
                    sample_msve_per_run[i, curr_checkpoint] += np.square(optimal_value - state_value) / CHECKPOINT
                    average_weight_per_checkpoint[i, curr_checkpoint][:new_weights.size] += new_weights / CHECKPOINT
                    if self.method != 'sgd':
                        average_stepsize_per_checkpoint[i, curr_checkpoint][:ss.size] += ss / CHECKPOINT
                    if (j + 1) % CHECKPOINT == 0:
                        curr_checkpoint += 1

                    if terminal:
                        env.reset()

                    self.add_features(env, approximator, optimizer, alpha, j)

            results_dir[names[a]] = {'sample_msve': sample_msve_per_run,
                                     'average_weight_per_checkpoint': average_weight_per_checkpoint}
            if self.method != 'sgd':
                results_dir['average_stepsize_per_checkpoint'] = average_stepsize_per_checkpoint

            # agg_results = np.average(sample_msve_per_run, axis=0)
            # import matplotlib.pyplot as plt
            # plt.plot(np.arange(agg_results.size)+1, agg_results)
            # plt.ylim((0,25))
            # plt.show()
            # plt.close()

        self.store_results(results_dir)

    def get_alphas_and_names(self):
        if self.method in ['adam', 'idbd', 'autostep', 'rescaled_sgd', 'restart_adam', 'sidbd']:
            alphas = ['']
            names = [self.method]
            return alphas, names
        og_alphas = np.array((BEST_PARAMETER_VALUE[self.method] / STEPSIZE_GROWTH_FACTOR,
                             BEST_PARAMETER_VALUE[self.method],
                             BEST_PARAMETER_VALUE[self.method] * STEPSIZE_GROWTH_FACTOR))
        og_names = ['small', 'med', 'large']
        if self.experiment_type in ['add_good_feats', 'add_bad_feats', 'continuously_add_bad']:
            alphas = og_alphas
            names = [i + '_stepsize' for i in og_names]
        elif self.experiment_type in ['add_4_good_4_bad', 'add_4_good_20_bad', 'add_4_good_100_bad']:
            alphas = []
            names = []
            for i in range(len(og_alphas)):     # every combination of (x,y) for x,y in {'small', 'med', 'large'}
                for j in range(len(og_alphas)):
                    alphas.append((og_alphas[i], og_alphas[j]))
                    names.append('good-' + og_names[i] + '_bad-' + og_names[j])
        else:
            raise ValueError("{0} is not a valid experiment type.".format(self.experiment_type))
        return alphas, names

    def add_features(self, env: BoyanChain, approximator, optimizer, alpha, iteration_number):
        if self.is_phased_training() and iteration_number + 1 == MIDPOINT:
            if self.experiment_type in ['add_good_feats', 'add_bad_feats']:
                add_bad = (self.experiment_type == 'add_bad_feats')
                fake_features = add_bad and self.add_fake_feature
                env.add_feature(4, noise=self.bad_feature_noise * add_bad, fake_feature=fake_features)
                approximator.increase_num_features(4)
                if self.method == 'sgd':
                    assert isinstance(alpha, float)
                    optimizer.increase_size(4, init_stepsize=alpha)
                else: optimizer.increase_size(4)
            elif self.experiment_type in ['add_4_good_4_bad', 'add_4_good_20_bad', 'add_4_good_100_bad']:
                if self.experiment_type == 'add_4_good_4_bad':
                    k = 8
                elif self.experiment_type == 'add_4_good_20_bad':
                    k = 24
                else:
                    k = 104
                env.add_feature(4, noise=0.0, fake_feature=False)   # add 4 good feature
                env.add_feature(k-4, noise=self.bad_feature_noise, fake_feature=self.add_fake_feature)  # add k - 4 bad features
                approximator.increase_num_features(k)
                if self.method == 'sgd':
                    assert isinstance(alpha, tuple) and len(alpha) == 2
                    new_stepsizes = np.ones(k)
                    new_stepsizes[:4] *= alpha[0]
                    new_stepsizes[4:] *= alpha[1]
                    optimizer.increase_size(k, new_stepsizes)
                else: optimizer.increase_size(k)
            else:
                raise ValueError("{0} is not a valid experiment type.".format(self.experiment_type))
        elif self.experiment_type == 'continuously_add_bad' and (iteration_number + 1) % ADD_FEATURE_INTERVAL == 0:
            env.add_feature(1, noise=self.bad_feature_noise, fake_feature=self.add_fake_feature)
            approximator.increase_num_features(1)
            if self.method == 'sgd':
                assert isinstance(alpha, float)
                optimizer.increase_size(1, init_stepsize=alpha)
            else: optimizer.increase_size(1)

    def is_phased_training(self):
        phased_training = self.experiment_type in ['add_good_feats', 'add_bad_feats', 'add_4_good_4_bad',
                                                   'add_4_good_20_bad', 'add_4_good_100_bad']
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
                            choices=['add_good_feats', 'add_bad_feats', 'add_4_good_4_bad', 'add_4_good_20_bad',
                                     'add_4_good_100_bad', 'continuously_add_bad'])
    parser.add_argument('-m', '--method', action='store', default='sgd', type=str,
                        choices=['sgd', 'adam', 'idbd', 'autostep', 'rescaled_sgd', 'restart_adam', 'sidbd'])
    parser.add_argument('-fs', '--feature_noise', action='store', default=0.1, type=float)
    parser.add_argument('-aff', '--add_fake_features', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true')
    exp_parameters = parser.parse_args()

    experiment_name = 'boyan_chain_task_add'
    if exp_parameters.add_fake_features:
        experiment_name += '_fake'
    experiment_name += '_features'
    results_path = os.path.join(os.getcwd(), 'results', experiment_name,
                                'feature_noise' + str(exp_parameters.feature_noise), exp_parameters.method,
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
