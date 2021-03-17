import numpy as np
import pickle
import os
import time
import argparse

from src import RandomFeatures, Config, LinearFunctionApproximator
from src import IDBD, Adam, AutoStep, SGD, SIDBD


class Experiment:

    def __init__(self, exp_arguments, results_path, tunable_parameter_values):

        self.results_path = results_path
        self.verbose = exp_arguments.verbose
        self.tunable_parameter_values = tunable_parameter_values
        self.stepsize_method = exp_arguments.stepsize_method
        self.noisy = exp_arguments.noisy
        self.config = Config()

        """ Environment Setup """
        self.config.num_true_features = exp_arguments.num_true_features
        self.config.num_obs_features = exp_arguments.num_true_features
        self.config.max_num_features = 200  # arbitrary since it's not used
        self.training_data = exp_arguments.training_data_size
        self.checkpoint = 10
        assert self.training_data % self.checkpoint == 0

        """ Experiment Setup """
        self.sample_size = exp_arguments.sample_size

        """ Stepsize adaptation settings"""
        self.config.parameter_size = self.config.num_true_features
        if self.stepsize_method == 'idbd':
            # non-tunable parameters
            self.config.init_beta = np.log(0.001)
            self.parameter_name = 'meta_stepsize'
            self.stepsize_method_class = IDBD
        elif self.stepsize_method == 'sidbd':
            self.config.init_beta = -np.log((1/0.001) - 1)  # equivalent to starting with a stepsize of 0.001
            self.parameter_name = 'meta_stepsize'
            self.stepsize_method_class = SIDBD
        elif self.stepsize_method == 'adam':
            # non-tunable parameters
            self.config.beta1 = 0.9
            self.config.beta2 = 0.99
            self.config.eps = 1e-08
            self.parameter_name = 'initial_stepsize'
            self.stepsize_method_class = Adam
        elif self.stepsize_method == 'autostep':
            # non-tunable parameters
            self.config.tau = 10000.0
            self.config.init_stepsize = 0.001
            self.parameter_name = 'meta_stepsize'
            self.stepsize_method_class = AutoStep
        elif self.stepsize_method in ['sgd', 'rescaled_sgd']:
            # non-tunable parameters
            self.parameter_name = 'stepsize'
            self.stepsize_method_class = SGD
            self.config.rescale = (self.stepsize_method == 'rescaled_sgd')
        else:
            raise ValueError("Unrecognized stepsize adaptation method.")

    def _print(self, astring):
        if self.verbose:
            print(astring)

    def set_tunable_parameter_value(self, val):
        if self.stepsize_method in ['idbd', 'sidbd']:
            self.config.theta = val
        elif self.stepsize_method == 'adam':
            self.config.init_alpha = val
        elif self.stepsize_method == 'autostep':
            self.config.mu = val
        elif self.stepsize_method in ['sgd', 'rescaled_sgd']:
            self.config.alpha = val
        else:
            raise ValueError("Unrecognized stepsize adaptation method.")

    def run(self):
        results = {'parameter_name': self.parameter_name,
                   'sample_size': self.sample_size,
                   'parameter_values': self.tunable_parameter_values,
                   'avg_l2_norm_diff': np.zeros(len(self.tunable_parameter_values)),
                   'avg_mse': np.zeros(len(self.tunable_parameter_values))}

        for j, pv in enumerate(self.tunable_parameter_values):
            self._print("Parameter value: {0}".format(pv))

            self.set_tunable_parameter_value(pv)
            l2_norm_diff_per_run = np.zeros(self.sample_size)
            avg_mse_per_run = np.zeros(self.sample_size)

            for i in range(self.sample_size):
                self._print("\tRun number: {0}".format(i+1))
                env = RandomFeatures(self.config)
                approximator = LinearFunctionApproximator(self.config)
                stepsize_method = self.stepsize_method_class(self.config)
                run_mse = np.zeros(self.training_data // self.checkpoint)

                mse = 0
                current_checkpoint = 0
                for k in range(self.training_data):
                    target, observable_features, best_approximation = env.sample_observation(noisy=self.noisy)
                    prediction = approximator.get_prediction(observable_features)
                    error = target - prediction
                    _, _, new_weights = stepsize_method.update_weight_vector(error, observable_features,
                                                                             approximator.get_weight_vector())
                    approximator.update_weight_vector(new_weights)

                    mse += np.square(error) / self.checkpoint
                    if (k + 1) % self.checkpoint == 0:
                        run_mse[current_checkpoint] += mse
                        mse *= 0
                        current_checkpoint += 1

                theta_star = env.theta
                approx_theta = approximator.get_weight_vector()
                l2_norm_diff = np.sqrt(np.sum(np.square(theta_star - approx_theta)))
                l2_norm_diff_per_run[i] += l2_norm_diff
                avg_mse_per_run[i] += np.average(run_mse)
                self._print("\t\tL2 Norm Difference: {0:.4f}".format(l2_norm_diff))
                self._print("\t\tAverage MSE: {0:.4f}".format(np.average(run_mse)))

            results['avg_l2_norm_diff'][j] += np.average(l2_norm_diff_per_run)
            results['avg_mse'][j] += np.average(avg_mse_per_run)
            self._print("Average L2 Norm Difference: {0:.4f}".format(np.average(l2_norm_diff_per_run)))
            self._print("Average MSE: {0:.4f}".format(np.average(avg_mse_per_run)))

        self.store_results(results)

    def store_results(self, results):
        file_path = os.path.join(self.results_path, 'parameter_tuning_results.p')
        with open(file_path, mode='wb') as results_file:
            pickle.dump(results, results_file)
        print("Results successfully stored.")


def main():
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ss', '--sample_size', action='store', default=1, type=int)
    parser.add_argument('-tds', '--training_data_size', action='store', default=100000, type=int)
    parser.add_argument('-ntf', '--num_true_features', action='store', default=3, type=int)
    parser.add_argument('-ssm', '--stepsize_method', action='store', default='sgd', type=str,
                        choices=['sgd', 'adam', 'idbd', 'autostep', 'rescaled_sgd', 'sidbd'])
    parser.add_argument('-tpv', '--tunable_parameter_values', action='store', nargs='+', type=float, required=True)
    parser.add_argument('--noisy', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true')
    exp_parameters = parser.parse_args()

    task_name = 'random_features_task'
    if exp_parameters.noisy:
        task_name = 'noisy_' + task_name

    results_path = os.path.join(os.getcwd(), 'results', 'parameter_tuning', task_name,
                                "num_true_features_" + str(exp_parameters.num_true_features),
                                exp_parameters.stepsize_method)
    os.makedirs(results_path, exist_ok=True)

    init_time = time.time()
    exp = Experiment(exp_parameters, results_path, exp_parameters.tunable_parameter_values)
    exp.run()
    finish_time = time.time()
    elapsed_time = (finish_time - init_time) / 60
    print("Running time in minutes: {0:.4f}".format(elapsed_time))


if __name__ == '__main__':
    main()
    # SGD parameter values: stepsize in {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001}
    #   - 0.005 had the lowest MSVE
    # IDBD parameter values: theta in {1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001}
    #   - 0.5 had the lowest MSVE
    # Adam parameter values: initial stepsize in {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001}
    #   - 0.005 had the lowest MSVE
    # AutoStep parameter values: mu in {0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005}
    #   - 0.05 had the lowest MSVE
    # Rescaled SGD parameter values: stepsize in {0.5, 0.1, 0.05, 0.01, 0.005, 0.001}
    #   - 0.01 had the lowest MSVE
    # SIDBD parameter values: theta in {5.0 1.0 0.5 0.1 0.05 0.01 0.005}
    #   - 0.5 had the lowest MSVE
