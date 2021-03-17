import numpy as np
import pickle
import os
import time
import argparse

from src import BoyanChain, Config, LinearFunctionApproximator
from src import IDBD, Adam, AutoStep, SGD, SIDBD


class Experiment:

    def __init__(self, exp_arguments, results_path, tunable_parameter_values):

        self.results_path = results_path
        self.verbose = exp_arguments.verbose
        self.tunable_parameter_values = tunable_parameter_values
        self.stepsize_method = exp_arguments.stepsize_method
        self.feature_noise = exp_arguments.feature_noise
        self.config = Config()

        """ Environment Setup """
        self.config.init_noise_var = self.feature_noise
        self.config.num_obs_features = 4
        self.config.max_num_features = 200  # arbitrary since it's not used
        self.training_data = exp_arguments.training_data_size
        self.checkpoint = 100
        assert self.training_data % self.checkpoint == 0

        """ Experiment Setup """
        self.sample_size = exp_arguments.sample_size

        """ Stepsize adaptation settings"""
        self.config.parameter_size = 4
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
                   'true_msve': np.zeros(len(self.tunable_parameter_values)),
                   'avg_msve': np.zeros(len(self.tunable_parameter_values))}

        for j, pv in enumerate(self.tunable_parameter_values):
            self._print("Parameter value: {0}".format(pv))

            self.set_tunable_parameter_value(pv)
            true_msve_per_run = np.zeros(self.sample_size)
            avg_msve_per_run = np.zeros(self.sample_size)

            for i in range(self.sample_size):
                self._print("\tRun number: {0}".format(i+1))
                env = BoyanChain(self.config)
                approximator = LinearFunctionApproximator(self.config)
                stepsize_method = self.stepsize_method_class(self.config)
                run_true_msve = np.zeros(self.training_data // self.checkpoint)
                run_avg_msve = np.zeros(self.training_data // self.checkpoint)

                avg_msve = 0
                current_checkpoint = 0

                current_obs_features = env.get_observable_features()
                for k in range(self.training_data):
                    state_value = approximator.get_prediction(current_obs_features)
                    optimal_value = env.compute_true_value()

                    _, reward, next_obs_features, terminal = env.step()

                    next_state_value = approximator.get_prediction(next_obs_features)

                    td_error = reward + (1 - terminal) * next_state_value - state_value
                    _, _, new_weights = stepsize_method.update_weight_vector(td_error, current_obs_features,
                                                                             approximator.get_weight_vector())
                    approximator.update_weight_vector(new_weights)

                    current_obs_features = next_obs_features
                    avg_msve += np.square(state_value - optimal_value) / self.checkpoint
                    if (k + 1) % self.checkpoint == 0:
                        run_true_msve[current_checkpoint] += env.compute_msve(new_weights)
                        run_avg_msve[current_checkpoint] += avg_msve

                        avg_msve *= 0
                        current_checkpoint += 1

                    if terminal:
                        env.reset()
                        current_obs_features = env.get_observable_features()

                true_msve_per_run[i] = np.average(run_true_msve)
                avg_msve_per_run[i] = np.average(run_avg_msve)
                self._print("\t\tTrue MSVE: {0:.4f}".format(true_msve_per_run[i]))
                self._print("\t\tAverage MSVE: {0:.4f}".format(avg_msve_per_run[i]))

            results['true_msve'][j] += np.average(true_msve_per_run)
            results['avg_msve'][j] += np.average(avg_msve_per_run)
            self._print("True MSVE: {0:.4f}".format(results['true_msve'][j]))
            self._print("Average MSVE: {0:.4f}".format(results['avg_msve'][j]))

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
    parser.add_argument('-tds', '--training_data_size', action='store', default=200000, type=int)
    parser.add_argument('-ssm', '--stepsize_method', action='store', default='sgd', type=str,
                        choices=['sgd', 'adam', 'idbd', 'autostep', 'rescaled_sgd', 'sidbd'])
    parser.add_argument('-fs', '--feature_noise', action='store', default=0.1, type=float)
    parser.add_argument('-tpv', '--tunable_parameter_values', action='store', nargs='+', type=float, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    exp_parameters = parser.parse_args()

    task_name = 'boyan_chain_task'
    results_path = os.path.join(os.getcwd(), 'results', 'parameter_tuning', task_name,
                                "feature_noise_" + str(exp_parameters.feature_noise),
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
    # SGD parameter values: stepsize in {0.5, 0.1, 0.05, 0.01, 0.005, 0.001}
    #   - 0.01 had the lowest true and sample MSVE
    # IDBD parameter values: theta in {10.0, 5.0, 1.0, 0.5, 0.1, 0.05}
    #   - 10 had the lowest true and sample MSVE, but also diverged many times when adding new features.
    #     Thus, we used 5.0 instead since it had the second lowest true and sample MSVE.
    # Adam parameter values: initial stepsize in {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005}
    #   - 0.001 had the lowest true and sample MSVE
    # AutoStep parameter values: mu in {0.5, 0.1, 0.05, 0.01, 0.005, 0.001}
    #   - 0.05 had the lowest true and sample MSVE
    # Rescaled SGD parameter values: stepsize in {0.5, 0.1, 0.05, 0.01, 0.005, 0.001}
    #   - 0.01 had the lowest true and sample MSVE
    # SIDBD parameter values: theta in {50.0 10.0 5.0 1.0 0.5 0.1}
    #   -  10.0 had the lowest true and sample MSVE
